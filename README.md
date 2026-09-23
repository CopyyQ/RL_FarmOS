# farmosv1

Clean maintenance snapshot of the FarmOS / Kaggriculture reinforcement-learning agent.

## Current stack

- Macro policy with PPO for strategic route decisions.
- Winner / elite behavior cloning for route, market and horizon heads.
- State-aware micro-skill policy for care and farm operations.
- Two-stage ACT/KEEP gate before the conditional skill head.
- Skill curriculum currently designed around the 720 -> 696 -> 672 cutover sequence.
- Safety rollback based on margin, plant deaths and animal escapes.

## Repository layout

- `src/kaggrl/`: core policy, observation, training and FarmOS logic.
- `rollout/`: hybrid rollout agent.
- `vendor/`: vendored Kaggle environment pieces required by the current runtime.
- `assets/`: small runtime assets required by the maintained configuration.
- `winner_train.py`: main training entry point.
- `continuous_runtime.py`: actor/runtime implementation.
- `v45_skill_runtime.py`: learned skill takeover runtime.
- `v45_skill_runtime_actkeep.py`: ACT/KEEP gated runtime.
- `train_v4_6_econ_shadow.sh`: maintained training launcher.
- `diag_*.py`, `bench_*.py`, `check_*.py`: diagnostics and regression tools.

Generated runs, logs, large checkpoints, backups and imported V3 training artifacts are intentionally not versioned.

## Environment

Python 3.11 is recommended. Install the PyTorch build appropriate for the target CUDA/CPU environment first, then:

```bash
python -m pip install -r requirements.txt
```

The current workstation training path uses CUDA, while Kaggle submission/runtime validation must remain CPU-compatible.

## Training

Use a dedicated environment and point the launcher at its Python executable:

```bash
FARMOS_PYTHON=/path/to/python ./train_v4_6_econ_shadow.sh
```

Optional overrides:

```bash
FARMOS_RUN_DIR=/path/to/run \
FARMOS_ACT_KEEP_THRESHOLD=0.50 \
FARMOS_FREEZE_MICRO_REP=1 \
FARMOS_PYTHON=/path/to/python \
./train_v4_6_econ_shadow.sh
```

The launcher uses `assets/act_keep_gate_sweep_best.pt` by default.

## Live logs

```bash
tail -n 100 -f runs/v46_econ_shadow/train_v4_6_econ_shadow.log \
  | grep --line-buffered -E '\[TRAIN\]|\[EVAL \]|\[PPO\]|\[MICRO\]|\[SKILL-VAL\]|\[SKILL\]|\[SKILL-STAGE-SAFE\]|\[SKILL-ROLLBACK\]|\[WINNER-BC\]'
```

## Maintenance rules

Do not commit `runs/`, `output/`, large checkpoints, local datasets, credentials or environment folders. Keep new behavior behind deterministic diagnostics/A-B tests before promoting it into the maintained runtime.

See `MAINTENANCE.md` for the current source-cleaning and release workflow.
