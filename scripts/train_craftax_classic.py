"""Train the existing DQN agents on Craftax-Classic-Symbolic.

The runner deliberately keeps vanilla DQN and curious DQN as separate agent
classes. It changes only the environment plumbing, step-based schedules, and
evaluation required for the long Craftax episodes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import random
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curious_agent.agents.dqn import DQNAgent
from curious_agent.agents.dqn_curious import DNQCuriousAgent
from curious_agent.env.craftax_classic import (
    CRAFTAX_CLASSIC_SYMBOLIC,
    CraftaxClassicAdapter,
)

logger = logging.getLogger(__name__)

AGENT_NAMES = ("vanilla-dqn", "curious-dqn")


class Environment(Protocol):
    state_dim: int
    num_actions: int

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset the environment and return its initial observation."""
        ...

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Apply an action and return the resulting transition."""
        ...

    def close(self) -> None:
        """Release resources owned by the environment."""
        ...


EnvironmentFactory = Callable[..., Environment]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a YAML experiment configuration."""
    with Path(path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Craftax configuration must be a YAML mapping")
    return config


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for a repeatable run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def epsilon_at_step(config: dict[str, Any], step: int) -> float:
    """Linearly anneal epsilon by environment interaction, not episode."""
    agent_config = config.get("agent", {})
    start = float(agent_config.get("epsilon_start", 1.0))
    end = float(agent_config.get("epsilon_end", 0.05))
    decay_steps = int(agent_config.get("epsilon_decay_steps", 80_000))
    if decay_steps <= 0:
        raise ValueError("agent.epsilon_decay_steps must be positive")
    fraction = min(max(step, 0) / decay_steps, 1.0)
    return start + fraction * (end - start)


def compose_controller_reward(
    agent_name: str,
    external_reward: float,
    curiosity_reward: float,
    beta: float,
) -> float:
    """Keep external-only and curiosity-shaped controller rewards explicit."""
    if agent_name == "vanilla-dqn":
        return float(external_reward)
    if agent_name == "curious-dqn":
        return float(external_reward + beta * curiosity_reward)
    raise ValueError(f"Unknown agent: {agent_name}")


def _agent_config(config: dict[str, Any]) -> dict[str, Any]:
    """Translate the Craftax runner names to the existing agent API."""
    resolved = copy.deepcopy(config)
    training = resolved.setdefault("training", {})
    target = resolved.get("target_network", {})
    agent = resolved.setdefault("agent", {})

    training["buffer_size"] = int(
        training.get("replay_capacity", training.get("buffer_size", 20_000))
    )
    training["min_buffer_size"] = int(
        training.get("learning_starts", training.get("min_buffer_size", 2_000))
    )
    training["target_update_frequency"] = int(
        target.get(
            "update_frequency", training.get("target_update_frequency", 100)
        )
    )
    training["tau"] = float(target.get("tau", training.get("tau", 0.005)))
    agent["epsilon"] = float(agent.get("epsilon_start", 1.0))
    agent["epsilon_min"] = float(agent.get("epsilon_end", 0.05))
    return resolved


def _make_agent(
    agent_name: str,
    environment: Environment,
    config: dict[str, Any],
) -> DQNAgent | DNQCuriousAgent:
    """Construct the selected existing DQN implementation on CPU."""
    device_name = str(config.get("training", {}).get("device", "cpu"))
    if device_name != "cpu":
        raise ValueError("The Craftax-Classic integration currently supports CPU only")
    device = torch.device("cpu")
    resolved = _agent_config(config)

    if agent_name == "vanilla-dqn":
        return DQNAgent(
            state_dim=environment.state_dim,
            num_actions=environment.num_actions,
            config=resolved,
            device=device,
        )
    if agent_name == "curious-dqn":
        return DNQCuriousAgent(
            state_dim=environment.state_dim,
            num_actions=environment.num_actions,
            config=resolved,
            device=device,
        )
    raise ValueError(f"Unknown agent: {agent_name}")


def _achievement_flags(info: dict[str, Any]) -> dict[str, int]:
    """Extract binary achievement outcomes from terminal Craftax info."""
    prefix = "Achievements/"
    return {
        key[len(prefix) :]: int(float(value) > 0.0)
        for key, value in info.items()
        if key.startswith(prefix)
    }


def _greedy_action(
    agent: DQNAgent | DNQCuriousAgent,
    agent_name: str,
    state: np.ndarray,
) -> int:
    """Select an action from a frozen policy without exploration."""
    if agent_name == "vanilla-dqn":
        assert isinstance(agent, DQNAgent)
        return agent.select_action(state, evaluate=True)

    assert isinstance(agent, DNQCuriousAgent)
    state_tensor = agent._encode_state(state)
    return agent.q_network.get_action(state_tensor, epsilon=0.0)


def _write_episode_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """Write episode metrics and achievement flags to a CSV file."""
    base_fields = [
        "episode",
        "end_step",
        "episode_complete",
        "external_return",
        "curiosity_return",
        "controller_return",
        "episode_length",
        "score",
        "unique_achievements",
    ]
    achievement_names = sorted(
        {
            name
            for record in records
            for name in record.get("achievements", {})
        }
    )
    achievement_fields = [f"achievement_{name}" for name in achievement_names]

    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=base_fields + achievement_fields)
        writer.writeheader()
        for record in records:
            row = {field: record[field] for field in base_fields}
            flags = record.get("achievements", {})
            row.update({
                f"achievement_{name}": (
                    int(flags.get(name, 0))
                    if record["episode_complete"]
                    else ""
                )
                for name in achievement_names
            })
            writer.writerow(row)


def _crafter_score(achievement_rates: dict[str, float]) -> float:
    """Crafter score: geometric mean of achievement percentages."""
    if not achievement_rates:
        return 0.0
    percentages = np.asarray(list(achievement_rates.values()), dtype=np.float64)
    return float(np.exp(np.mean(np.log1p(percentages))) - 1.0)


def evaluate_agent(
    agent: DQNAgent | DNQCuriousAgent,
    agent_name: str,
    config: dict[str, Any],
    env_factory: EnvironmentFactory = CraftaxClassicAdapter,
) -> dict[str, Any]:
    """Evaluate a frozen greedy policy on shared, unseen procedural worlds."""
    evaluation = config.get("evaluation", {})
    episodes = int(evaluation.get("episodes", 10))
    if episodes <= 0:
        return {"episodes": 0, "evaluation_seeds": []}

    environment_config = config.get("environment", {})
    max_steps = int(
        evaluation.get(
            "max_episode_steps",
            environment_config.get("max_episode_steps", 10_000),
        )
    )
    seed_offset = int(evaluation.get("seed_offset", 10_000))
    environment = env_factory(
        seed=seed_offset,
        max_episode_steps=max_steps,
        env_name=environment_config.get("name", CRAFTAX_CLASSIC_SYMBOLIC),
    )

    returns: list[float] = []
    lengths: list[int] = []
    achievement_records: list[dict[str, int]] = []
    evaluation_seeds = [seed_offset + index for index in range(episodes)]

    try:
        for evaluation_seed in evaluation_seeds:
            state = environment.reset(seed=evaluation_seed)
            episode_return = 0.0
            final_info: dict[str, Any] = {}

            episode_length = 0
            for _ in range(max_steps):
                episode_length += 1
                action = _greedy_action(agent, agent_name, state)
                state, reward, done, final_info = environment.step(action)
                episode_return += reward
                if done:
                    break

            returns.append(episode_return)
            lengths.append(episode_length)
            achievement_records.append(_achievement_flags(final_info))
    finally:
        environment.close()

    achievement_names = sorted(
        {name for record in achievement_records for name in record}
    )
    achievement_rates = {
        name: 100.0
        * float(np.mean([record.get(name, 0) for record in achievement_records]))
        for name in achievement_names
    }
    unique_counts = [sum(record.values()) for record in achievement_records]

    return {
        "episodes": episodes,
        "evaluation_seeds": evaluation_seeds,
        "external_return_mean": float(np.mean(returns)),
        "external_return_std": float(np.std(returns)),
        "episode_length_mean": float(np.mean(lengths)),
        "unique_achievements_mean": float(np.mean(unique_counts)),
        "achievement_rates_percent": achievement_rates,
        "crafter_score": _crafter_score(achievement_rates),
    }


def train(
    agent_name: str,
    config: dict[str, Any],
    output_root: str | Path | None = None,
    env_factory: EnvironmentFactory = CraftaxClassicAdapter,
) -> dict[str, Any]:
    """Train one of the two DQN algorithms and write reproducible artifacts."""
    if agent_name not in AGENT_NAMES:
        raise ValueError(f"agent_name must be one of {AGENT_NAMES}")

    training = config.get("training", {})
    environment_config = config.get("environment", {})
    seed = int(training.get("seed", 0))
    total_steps = int(training.get("total_steps", 100_000))
    learning_starts = int(training.get("learning_starts", 2_000))
    train_frequency = int(training.get("train_frequency", 4))
    log_every = int(training.get("log_every_steps", 1_000))
    checkpoint_every = int(training.get("checkpoint_every_steps", 25_000))
    max_episode_steps = int(environment_config.get("max_episode_steps", 10_000))

    for name, value in {
        "total_steps": total_steps,
        "learning_starts": learning_starts,
        "train_frequency": train_frequency,
        "log_every_steps": log_every,
        "checkpoint_every_steps": checkpoint_every,
    }.items():
        if value <= 0:
            raise ValueError(f"training.{name} must be positive")

    thread_count = int(training.get("torch_num_threads", 8))
    torch.set_num_threads(max(1, thread_count))
    set_global_seed(seed)

    root = Path(
        output_root
        if output_root is not None
        else config.get("paths", {}).get("output_root", "runs/craftax_classic")
    )
    run_name = agent_name.replace("-", "_")
    run_dir = root / run_name / f"seed_{seed}"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as output:
        yaml.safe_dump(config, output, sort_keys=False)

    environment = env_factory(
        seed=seed,
        max_episode_steps=max_episode_steps,
        env_name=environment_config.get("name", CRAFTAX_CLASSIC_SYMBOLIC),
    )
    agent = _make_agent(agent_name, environment, config)
    beta = float(config.get("agent", {}).get("beta", 1.0))

    records: list[dict[str, Any]] = []
    completed_external_returns: list[float] = []
    curiosity_values: list[float] = []
    latest_q_loss: float | None = None
    latest_model_loss: float | None = None
    latest_confidence_loss: float | None = None
    episode_external = 0.0
    episode_curiosity = 0.0
    episode_controller = 0.0
    episode_length = 0
    update_count = 0
    started_at = time.perf_counter()
    state = environment.reset()

    logger.info(
        "Starting %s on %s: steps=%s seed=%s state_dim=%s actions=%s",
        agent_name,
        environment_config.get("name", CRAFTAX_CLASSIC_SYMBOLIC),
        total_steps,
        seed,
        environment.state_dim,
        environment.num_actions,
    )

    try:
        for step in range(1, total_steps + 1):
            agent.epsilon = epsilon_at_step(config, step - 1)
            action = agent.select_action(state)
            next_state, external_reward, done, info = environment.step(action)

            curiosity_reward = 0.0
            if agent_name == "curious-dqn":
                assert isinstance(agent, DNQCuriousAgent)
                confidence_before = agent.get_confidence_before(state, action)
                state_tensor = agent._encode_state(state).unsqueeze(0)
                action_tensor = agent._encode_action(action).unsqueeze(0)
                next_state_tensor = agent._encode_state(next_state).unsqueeze(0)
                actual_error = agent.world_model.compute_error(
                    state_tensor, action_tensor, next_state_tensor
                )
                latest_model_loss = agent.update_model(state, action, next_state)
                latest_confidence_loss = agent.update_confidence(
                    state, action, actual_error
                )
                confidence_after = agent.get_confidence_before(state, action)
                curiosity_reward = agent.compute_curiosity_reward(
                    confidence_before, confidence_after
                )
                curiosity_values.append(curiosity_reward)

            controller_reward = compose_controller_reward(
                agent_name,
                external_reward,
                curiosity_reward,
                beta,
            )
            agent.store_experience(
                state, action, controller_reward, next_state, done
            )

            if step >= learning_starts and step % train_frequency == 0:
                latest_q_loss = agent.update_q_network()
                if latest_q_loss is not None:
                    update_count += 1

            episode_external += external_reward
            episode_curiosity += curiosity_reward
            episode_controller += controller_reward
            episode_length += 1

            if done:
                flags = _achievement_flags(info)
                records.append(
                    {
                        "episode": len(records) + 1,
                        "end_step": step,
                        "episode_complete": True,
                        "external_return": episode_external,
                        "curiosity_return": episode_curiosity,
                        "controller_return": episode_controller,
                        "episode_length": episode_length,
                        "score": float(info.get("score", 0.0)),
                        "unique_achievements": sum(flags.values()),
                        "achievements": flags,
                    }
                )
                completed_external_returns.append(episode_external)
                state = environment.reset()
                episode_external = 0.0
                episode_curiosity = 0.0
                episode_controller = 0.0
                episode_length = 0
            else:
                state = next_state

            if step % checkpoint_every == 0:
                agent.save(checkpoint_dir / f"agent_step_{step}.pt")

            if step % log_every == 0 or step == total_steps:
                elapsed = time.perf_counter() - started_at
                return_10 = (
                    float(np.mean(completed_external_returns[-10:]))
                    if completed_external_returns
                    else math.nan
                )
                curiosity_mean = (
                    float(np.mean(curiosity_values[-log_every:]))
                    if curiosity_values
                    else 0.0
                )
                logger.info(
                    "step=%s/%s epsilon=%.3f replay=%s updates=%s "
                    "external_return_10=%.3f curiosity_mean=%.6f "
                    "q_loss=%s steps_per_second=%.1f",
                    step,
                    total_steps,
                    agent.epsilon,
                    len(agent.replay_buffer),
                    update_count,
                    return_10,
                    curiosity_mean,
                    "n/a" if latest_q_loss is None else f"{latest_q_loss:.6f}",
                    step / elapsed,
                )

        if episode_length > 0:
            records.append(
                {
                    "episode": len(records) + 1,
                    "end_step": total_steps,
                    "episode_complete": False,
                    "external_return": episode_external,
                    "curiosity_return": episode_curiosity,
                    "controller_return": episode_controller,
                    "episode_length": episode_length,
                    "score": None,
                    "unique_achievements": None,
                    "achievements": {},
                }
            )
    finally:
        environment.close()

    training_elapsed = time.perf_counter() - started_at
    agent.save(checkpoint_dir / "agent_final.pt")
    _write_episode_csv(run_dir / "episodes.csv", records)

    evaluation = evaluate_agent(agent, agent_name, config, env_factory)
    with (run_dir / "evaluation.json").open("w", encoding="utf-8") as output:
        json.dump(evaluation, output, indent=2)
        output.write("\n")

    summary: dict[str, Any] = {
        "agent": agent_name,
        "environment": environment_config.get(
            "name", CRAFTAX_CLASSIC_SYMBOLIC
        ),
        "seed": seed,
        "total_steps": total_steps,
        "completed_episodes": sum(
            int(record["episode_complete"]) for record in records
        ),
        "recorded_episodes": len(records),
        "q_updates": update_count,
        "training_seconds": training_elapsed,
        "steps_per_second": total_steps / training_elapsed,
        "mean_training_external_return": (
            float(np.mean(completed_external_returns))
            if completed_external_returns
            else None
        ),
        "mean_step_curiosity": (
            float(np.mean(curiosity_values)) if curiosity_values else 0.0
        ),
        "final_q_loss": latest_q_loss,
        "final_model_loss": latest_model_loss,
        "final_confidence_loss": latest_confidence_loss,
        "evaluation": evaluation,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)
        output.write("\n")

    logger.info("Completed %s; artifacts=%s", agent_name, run_dir)
    return summary


def _smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a small configuration for a fast end-to-end check."""
    smoke = copy.deepcopy(config)
    smoke["environment"]["max_episode_steps"] = 16
    smoke["training"].update(
        total_steps=64,
        replay_capacity=128,
        learning_starts=8,
        batch_size=8,
        train_frequency=2,
        log_every_steps=16,
        checkpoint_every_steps=64,
    )
    smoke["agent"]["epsilon_decay_steps"] = 48
    smoke["evaluation"].update(episodes=2, max_episode_steps=16)
    return smoke


def parse_args() -> argparse.Namespace:
    """Parse command-line options for one Craftax DQN run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=AGENT_NAMES, required=True)
    parser.add_argument(
        "--config", default="configs/craftax_classic_cpu.yaml"
    )
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Load configuration, run training, and print the final summary."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    config = load_config(args.config)
    if args.total_steps is not None:
        config["training"]["total_steps"] = args.total_steps
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.eval_episodes is not None:
        config["evaluation"]["episodes"] = args.eval_episodes
    if args.smoke_test:
        config = _smoke_config(config)

    summary = train(args.agent, config, output_root=args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
