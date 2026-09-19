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

## CPU pilot

The default configuration runs 100,000 environment interactions with one
environment, a 20,000-transition replay buffer, batch size 32, and one DQN
update every four steps:

```bash
python scripts/train_craftax_classic.py \
  --agent vanilla-dqn \
  --config configs/craftax_classic_cpu.yaml \
  --seed 0

python scripts/train_craftax_classic.py \
  --agent curious-dqn \
  --config configs/craftax_classic_cpu.yaml \
  --seed 0
```
 
## Evaluation protocol

After training, the runner automatically evaluates the frozen greedy policy on
fresh worlds beginning at seed 10,000. Evaluation uses:

- epsilon zero;
- no replay insertion;
- no Q-network updates;
- no world- or confidence-model updates; and
- the same world seeds for both algorithms.

It compare the algorithms using external return, episode length, unique
achievements, per-achievement discovery rates, and the Crafter geometric-mean
score. Curiosity reward is saved only as a diagnostic and must not be compared
against vanilla external return.

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
to be compared on identical procedural worlds.

## Scope

This is a sequential PyTorch DQN experiment. The official Craftax baselines
use compiled PPO and many parallel JAX environments, so their speed and scores
are not directly comparable. The intended question is narrower: under equal
CPU and interaction budgets, does confidence-improvement curiosity change
external return or achievement discovery relative to vanilla DQN?
