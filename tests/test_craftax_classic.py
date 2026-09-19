"""Tests for the Craftax-Classic adapter and shared DQN runner."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from curious_agent.env.craftax_classic import (
    CRAFTAX_CLASSIC_SYMBOLIC,
    EXPECTED_NUM_ACTIONS,
    EXPECTED_STATE_DIM,
    CraftaxClassicAdapter,
)
from scripts.train_craftax_classic import (
    compose_controller_reward,
    train,
)


class FakeCraftaxEnvironment:
    """Small deterministic substitute for fast trainer tests."""

    state_dim = 4
    num_actions = 2

    def __init__(
        self,
        seed: int,
        max_episode_steps: int,
        env_name: str,
    ) -> None:
        """Initialize the deterministic test environment."""
        assert env_name == CRAFTAX_CLASSIC_SYMBOLIC
        self.seed = seed
        self.max_episode_steps = max_episode_steps
        self.episode_step = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset the short synthetic episode."""
        if seed is not None:
            self.seed = seed
        self.episode_step = 0
        return np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Advance one deterministic synthetic transition."""
        self.episode_step += 1
        done = self.episode_step >= min(3, self.max_episode_steps)
        reward = 1.0 if done else 0.0
        observation = np.asarray(
            [
                self.episode_step / max(self.max_episode_steps, 1),
                float(action),
                float(action == 0),
                float(action == 1),
            ],
            dtype=np.float32,
        )
        info = {
            "Achievements/collect_wood": 100.0 if done else 0.0,
            "Achievements/collect_stone": 0.0,
            "score": 1.0 if done else 0.0,
        }
        return observation, reward, done, info

    def close(self) -> None:
        """Close the resource-free test environment."""
        return None


def small_config() -> dict[str, Any]:
    """Build a minimal configuration for fast trainer tests."""
    return {
        "environment": {
            "name": CRAFTAX_CLASSIC_SYMBOLIC,
            "max_episode_steps": 3,
        },
        "networks": {
            "q_network": {
                "hidden_dims": [8],
                "learning_rate": 1e-3,
                "use_dueling": True,
                "value_hidden_dim": 4,
                "advantage_hidden_dim": 4,
            },
            "world_model": {"hidden_dims": [8], "learning_rate": 1e-3},
            "confidence_net": {"hidden_dims": [4], "learning_rate": 5e-4},
        },
        "agent": {
            "gamma": 0.99,
            "epsilon_start": 0.2,
            "epsilon_end": 0.0,
            "epsilon_decay_steps": 8,
            "beta": 1.0,
        },
        "training": {
            "total_steps": 12,
            "replay_capacity": 32,
            "learning_starts": 2,
            "batch_size": 2,
            "train_frequency": 1,
            "log_every_steps": 6,
            "checkpoint_every_steps": 12,
            "seed": 7,
            "device": "cpu",
            "torch_num_threads": 1,
        },
        "target_network": {"update_frequency": 2, "tau": 0.01},
        "evaluation": {
            "episodes": 2,
            "max_episode_steps": 3,
            "epsilon": 0.0,
            "seed_offset": 100,
        },
        "paths": {"output_root": "unused-in-test"},
    }


def test_controller_rewards_remain_separate() -> None:
    """Vanilla ignores curiosity while curious DQN adds its weighted value."""
    assert compose_controller_reward("vanilla-dqn", 2.0, 5.0, 0.5) == 2.0
    assert compose_controller_reward("curious-dqn", 2.0, 5.0, 0.5) == 4.5


def test_adapter_rejects_non_classic_environment_before_import() -> None:
    """The adapter rejects full Craftax and pixel variants."""
    with pytest.raises(ValueError, match="only supports"):
        CraftaxClassicAdapter(env_name="Craftax-Symbolic-v1")


@pytest.mark.parametrize("agent_name", ["vanilla-dqn", "curious-dqn"])
def test_shared_runner_writes_training_and_evaluation_artifacts(
    tmp_path: Path, agent_name: str
) -> None:
    """Both DQN variants train and write the expected result files."""
    summary = train(
        agent_name,
        small_config(),
        output_root=tmp_path,
        env_factory=FakeCraftaxEnvironment,
    )

    run_dir = tmp_path / agent_name.replace("-", "_") / "seed_7"
    assert summary["q_updates"] > 0
    assert summary["evaluation"]["episodes"] == 2
    assert summary["evaluation"]["external_return_mean"] == 1.0
    assert summary["evaluation"]["achievement_rates_percent"] == {
        "collect_stone": 0.0,
        "collect_wood": 100.0,
    }
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "checkpoints" / "agent_final.pt").is_file()
    assert json.loads((run_dir / "evaluation.json").read_text())["episodes"] == 2

    with (run_dir / "episodes.csv").open(newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    assert len(rows) == 4
    assert all(float(row["external_return"]) == 1.0 for row in rows)
    if agent_name == "vanilla-dqn":
        assert summary["mean_step_curiosity"] == 0.0
        assert all(float(row["curiosity_return"]) == 0.0 for row in rows)
    else:
        assert summary["final_model_loss"] is not None
        assert summary["final_confidence_loss"] is not None


@pytest.mark.skipif(
    importlib.util.find_spec("jax") is None
    or importlib.util.find_spec("craftax") is None,
    reason="optional Craftax dependencies are not installed",
)
def test_real_adapter_exposes_classic_symbolic_contract() -> None:
    """The real adapter exposes 1,345 features and 17 actions."""
    environment = CraftaxClassicAdapter(seed=3, max_episode_steps=2)
    try:
        observation = environment.reset()
        assert observation.shape == (EXPECTED_STATE_DIM,)
        assert observation.dtype == np.float32
        assert environment.num_actions == EXPECTED_NUM_ACTIONS

        next_observation, reward, done, info = environment.step(0)
        assert next_observation.shape == observation.shape
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "score" in info
    finally:
        environment.close()
