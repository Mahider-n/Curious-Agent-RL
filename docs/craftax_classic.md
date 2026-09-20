# Running the DQN comparison on Craftax-Classic-Symbolic

This integration compares the repository's two existing algorithms on fresh
procedural Craftax-Classic worlds:

- **Vanilla DQN** stores only Craftax's external reward in replay.
- **Curious DQN** stores external reward plus
  `beta * (C_before - C_after)` and continues training its world and confidence
  models online.

Both use the same dueling Q-network, replay settings, target updates,
step-based epsilon schedule, environment budget, and evaluation seeds. The
integration supports only `Craftax-Classic-Symbolic-v1` with 1,345 observation
features and 17 discrete actions.

## Installation

All required packages, including CPU JAX and Craftax, are listed in
`requirements.txt`. For a new environment, use:

```bash
cd Curious-Agent-RL
python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

## Verify the integration

Run the focused tests:

```bash
pytest tests/test_craftax_classic.py
```

Run a 64-step end-to-end smoke test for each algorithm:

```bash
python scripts/train_craftax_classic.py \
  --agent vanilla-dqn \
  --config configs/craftax_classic_cpu.yaml \
  --smoke-test \
  --output-dir runs/craftax_smoke

python scripts/train_craftax_classic.py \
  --agent curious-dqn \
  --config configs/craftax_classic_cpu.yaml \
  --smoke-test \
  --output-dir runs/craftax_smoke
```

## Real Vanilla DQN versus Curious DQN comparison

Activate the project environment and run both agents without `--smoke-test`:

```bash
source venv/bin/activate

python scripts/train_craftax_classic.py \
  --agent vanilla-dqn \
  --config configs/craftax_classic_cpu.yaml \
  --seed 0 \
  --output-dir runs/craftax_classic

python scripts/train_craftax_classic.py \
  --agent curious-dqn \
  --config configs/craftax_classic_cpu.yaml \
  --seed 0 \
  --output-dir runs/craftax_classic
```

The commands use the same 100,000-step budget and save the agents in separate
directories.

## Seed-0 result currently in `runs/craftax_classic`

The completed runs produced the following greedy evaluation results on the
same ten worlds.
| Metric | Vanilla DQN | Curious DQN |
| --- | ---: | ---: |
| Training steps | 100,000 | 100,000 |
| Q updates | 24,501 | 24,501 |
| Mean evaluation external return | 1.10 | 1.80 |
| Evaluation return standard deviation | 0.89 | 0.64 |
| Mean unique achievements | 2.0 | 2.7 |
| Crafter score | 1.416 | 1.540 |
| Mean evaluation episode length | 164.4 | 161.5 |
| Training throughput (steps/second) | 539.0 | 206.6 |

The non-zero achievement rates were:

| Achievement | Vanilla DQN | Curious DQN |
| --- | ---: | ---: |
| `collect_sapling` | 70% | 100% |
| `collect_wood` | 20% | 30% |
| `eat_cow` | 10% | 10% |
| `place_plant` | 70% | 100% |
| `place_table` | 20% | 10% |
| `wake_up` | 10% | 20% |

## Run analysis

Both agents were trained for 100,000 steps with 24,501 Q-network updates and evaluated on the same ten unseen Craftax worlds.
Curious DQN performed better during evaluation. It achieved a mean external return of 1.80, compared with 1.10 for Vanilla DQN, and discovered an average of 2.7 unique achievements versus 2.0. Its Crafter score was also slightly higher (1.540 versus 1.416). Curious DQN more frequently collected saplings and wood, placed plants, and completed wake_up, while Vanilla DQN performed better at placing tables.
However, Curious DQN was computationally more expensive. It processed approximately 206.6 steps per second, while Vanilla DQN achieved 539.0 steps per second. This slowdown is expected because Curious DQN updates the world model and confidence network at every interaction.

Overall, curiosity produced better evaluation results in this run, but the result is based on only one training seed. Multiple training seeds are required to determine whether the improvement is reliable or caused by random variation.

## Execution flow

In more detail, every training iteration performs these steps:

1. Compute epsilon from the global environment-step counter.
2. Select an epsilon-greedy action with the chosen agent.
3. Step Craftax and receive the next observation, external reward, terminal
   flag, and information dictionary.
4. For Curious DQN only, update the world and confidence models and calculate
   curiosity from the change in confidence prediction.
5. Form the appropriate controller reward and store the transition.
6. If replay is warm and the step matches the training frequency, perform one
   Q update. Periodically perform the configured soft target update.
7. On a terminal step, record returns, episode length, score, and achievements,
   then reset into a new procedural world.
8. At the configured intervals, save a checkpoint and print progress.
9. At the end, save the final model, evaluate it without exploration or
   learning, and write the result files.

## Output layout

```text
runs/craftax_classic/
├── vanilla_dqn/seed_0/
│   ├── config.yaml
│   ├── episodes.csv
│   ├── evaluation.json
│   ├── summary.json
│   └── checkpoints/agent_final.pt
└── curious_dqn/seed_0/
    └── ...
```

`evaluation.json` records the exact evaluation seeds, allowing both algorithms
to be compared on identical procedural worlds. If training ends in the middle
of an episode, `episodes.csv` records that row with `episode_complete=False`;
terminal-only score and achievement fields are left blank.

## Scope

This is a sequential PyTorch DQN experiment. The official Craftax baselines
use compiled PPO and many parallel JAX environments, so their speed and scores
are not directly comparable. The intended question is narrower: under equal
CPU and interaction budgets, does confidence-improvement curiosity change
external return or achievement discovery relative to vanilla DQN?
