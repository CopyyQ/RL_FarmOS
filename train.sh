#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${FARMOS_RUN_DIR:-$ROOT/runs/v46_econ_shadow}"
PYTHON_BIN="${FARMOS_PYTHON:-python}"
LOG="$RUN/train_v4_6_econ_shadow.log"
PIDFILE="$RUN/winner_train.pid"
LOCK="$RUN/winner_train.lock"

mkdir -p "$RUN"
cd "$ROOT"
if pgrep -af "[p]ython.*winner_train.py" >/dev/null; then
  echo "ERROR: another winner_train.py is already running"
  pgrep -af "[p]ython.*winner_train.py"
  exit 2
fi

nohup bash -c "exec 9>'$LOCK'; flock -n 9 || exit 99; \
  export FARMOS_ACT_KEEP_GATE_PATH='${FARMOS_ACT_KEEP_GATE_PATH:-$ROOT/assets/act_keep_gate_sweep_best.pt}'; \
  export FARMOS_ACT_KEEP_THRESHOLD='${FARMOS_ACT_KEEP_THRESHOLD:-0.50}'; \
  export FARMOS_FREEZE_MICRO_REP='${FARMOS_FREEZE_MICRO_REP:-1}'; \
  exec '$PYTHON_BIN' -u winner_train.py \
  --output-dir '$RUN' \
  --workers 24 \
  --train-seeds-per-iter 32 \
  --worker-recycle-tasks 64 \
  --eval-every 10 \
  --temperature 1.30 \
  --min-temperature 1.00 \
  --entropy-coef 0.030 \
  --target-kl 0.010 \
  --winner-refresh-every 5 \
  --winner-refresh-epochs 1 \
  --migration-grace-iters 30 \
  --micro-learning-rate 0.000125 \
  --micro-ppo-epochs 1 \
  --micro-clip-ratio 0.08 \
  --micro-entropy-coef 0.010 \
  --micro-value-coef 0.20 \
  --micro-target-kl 0.010 \
  --skill-cutover-start 720 \
  --skill-cutover-min 672 \
  --skill-cutover-step-size 24 \
  --skill-shadow-start-step 0 \
  --skill-confidence-threshold 0.70 \
  --skill-min-takeover-ratio 0.08 \
  --skill-min-executed-samples 32 \
  --skill-keep-penalty 2.75 \
  --skill-bc-epochs 2 \
  --skill-teacher-per-class 2048 \
  --skill-prior-power 0.85 \
  --skill-bc-min-nonkeep-acc 0.72 \
  --skill-bc-min-core-acc 0.50 \
  --skill-val-min-core-f1 0.70 \
  --skill-stable-evals-required 5 \
  --skill-max-plant-deaths-per-game 20.0 \
  --skill-max-animal-escapes 0 \
  --skill-floor-eval-every 2 \
  --skill-stage-margin-tolerance 1000" >> "$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "STARTED pid=$(cat "$PIDFILE")"
echo "LOG=$LOG"
