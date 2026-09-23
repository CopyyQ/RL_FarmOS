# Maintenance workflow

## Source of truth

This repository is the clean maintenance copy named `farmosv1`. Experimental training may continue in a separate working directory, but maintainable changes should be copied or cherry-picked here only after regression checks.

## Before committing

1. Run Python compilation on maintained entry points.
2. Run action-alignment and determinism checks.
3. Run focused A/B tests for any policy/runtime behavior change.
4. Confirm no credentials, training logs, run directories or large generated checkpoints are staged.
5. Inspect `git status` and the staged diff before committing.

Recommended checks:

```bash
python -m py_compile \
  winner_train.py \
  continuous_runtime.py \
  v45_skill_runtime.py \
  v45_skill_runtime_actkeep.py

PYTHONPATH="$PWD/src:$PWD" python check_action_alignment.py
PYTHONPATH="$PWD/src:$PWD" python check_one_determinism.py
```

## Versioned runtime assets

Only these small runtime assets are intentionally tracked:

- `assets/parent_promoted_v2.pt`
- `assets/act_keep_gate_sweep_best.pt`

Training checkpoints such as `winner_v4_latest.pt`, `winner_v4_safe_best.pt` and `winner_v45_skill_stage_safe.pt` stay outside Git.

## Current safety expectations

For learned-skill takeover, preserve:

- animal escapes at zero on safety evals;
- plant deaths at or below the established baseline;
- takeover above the configured minimum;
- margin within the configured stage tolerance;
- winner anchor reproducibility;
- CPU compatibility for Kaggle runtime/submission.

## Git branch

The maintained branch is `farmosv1`.
