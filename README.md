# FarmOS V1

Bản mã nguồn bảo trì gọn của agent FarmOS/Kaggriculture hiện tại.

## Cấu trúc

```text
farmosv1/
├── assets/                         # parent model, ACT/KEEP gate, v51 reference
├── rollout/                        # hybrid rollout agent
├── src/kaggrl/                     # core V4.6 policy/game logic
├── tools/
│   ├── bench/                      # A/B benchmark quan trọng
│   └── diag/                       # chẩn đoán skill/ACT-KEEP
├── tests/                          # regression checks
├── vendor/                         # Kaggriculture engine cố định
├── continuous_runtime.py           # actor/runtime
├── v45_skill_runtime.py            # skill takeover runtime
├── v45_skill_runtime_actkeep.py    # ACT/KEEP gated runtime
├── winner_train.py                 # trainer chính
└── train.sh                        # launcher
```

Không lưu trong Git: `runs/`, `output/`, `backups/`, dataset, log, virtualenv và checkpoint train lớn.

## Môi trường

Khuyến nghị Python 3.11. Cài PyTorch phù hợp CUDA/CPU trước, sau đó:

```bash
python -m pip install -r requirements.txt
```

## Train

```bash
FARMOS_PYTHON=/path/to/python ./train.sh
```

Biến môi trường thường dùng:

```bash
FARMOS_RUN_DIR=/path/to/run
FARMOS_ACT_KEEP_GATE_PATH=/path/to/gate.pt
FARMOS_ACT_KEEP_THRESHOLD=0.50
FARMOS_FREEZE_MICRO_REP=1
```

## Xem log

```bash
tail -n 100 -f runs/v46_econ_shadow/train_v4_6_econ_shadow.log \
  | grep --line-buffered -E '\[TRAIN\]|\[EVAL \]|\[PPO\]|\[MICRO\]|\[SKILL\]|\[SKILL-VAL\]|\[SKILL-STAGE-SAFE\]|\[SKILL-ROLLBACK\]|\[WINNER-BC\]'
```

## Benchmark

Các benchmark cần checkpoint train bên ngoài Git:

```bash
export FARMOS_CHECKPOINT=/path/to/winner_v45_skill_stage_safe.pt
python tools/bench/cutover.py
python tools/bench/act_keep.py
```

## Diagnostic

```bash
export FARMOS_CHECKPOINT=/path/to/winner_v45_skill_stage_safe.pt
python tools/diag/skill_val.py
python tools/diag/skill_exec.py
python tools/diag/act_keep_gate.py
python tools/diag/act_keep_sweep.py
```

## Regression

```bash
export FARMOS_CHECKPOINT=/path/to/winner_v45_skill_stage_safe.pt
python tests/test_action_alignment.py
python tests/test_determinism.py
```

Quy trình bảo trì chi tiết nằm trong [MAINTENANCE.md](MAINTENANCE.md).
