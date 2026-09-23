#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

LOG="output/train_v4_5_skill_manager.log"
PIDFILE="output/winner_train.pid"
LOCK="output/winner_train.lock"
mkdir -p output

if ps -eo pid,args | grep -E '[.]venv311/bin/python( -u)? winner_train.py' | grep -v grep >/dev/null 2>&1; then
  echo "TRAINER_ALREADY_RUNNING"
  ps -eo pid,args | grep -E '[.]venv311/bin/python( -u)? winner_train.py' | grep -v grep || true
else
  nohup bash -c '
    exec 9>output/winner_train.lock
    if ! flock -n 9; then
      echo "TRAINER_LOCKED_ANOTHER_INSTANCE_RUNNING"
      exit 99
    fi
    exec ../farmos_rl/.venv311/bin/python -u winner_train.py \
      --workers 32 \
      --train-seeds-per-iter 32 \
      --worker-recycle-tasks 64 \
      --eval-every 10 \
      --temperature 1.30 \
      --min-temperature 1.00 \
      --entropy-coef 0.030 \
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
      --skill-cutover-min 696 \
      --skill-cutover-step-size 24 \
      --skill-shadow-start-step 0 \
      --skill-confidence-threshold 0.70 \
      --skill-keep-penalty 2.75 \
      --skill-bc-epochs 4 \
      --skill-teacher-per-class 2048 \
      --skill-bc-min-nonkeep-acc 0.72 \
      --skill-bc-min-core-acc 0.50 \
      --skill-stable-evals-required 2 \
      --skill-max-plant-deaths-per-game 20.0 \
      --skill-max-animal-escapes 0 \
      --skill-floor-eval-every 2 \
      --skill-stage-margin-tolerance 1000
  ' >> "$LOG" 2>&1 &
  TRAIN_PID=$!
  echo "$TRAIN_PID" > "$PIDFILE"
  echo "TRAIN_PID=$TRAIN_PID"
  sleep 2
fi

echo "TAILING $LOG (Ctrl+C only stops tail; trainer keeps running)"
exec tail -n 100 -f "$LOG"
