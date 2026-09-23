# Maintenance

## Nguyên tắc

- `farmosv1` là branch mã nguồn bảo trì, không phải thư mục lưu experiment.
- Không commit `runs/`, `output/`, `backups/`, dataset, log, credential hoặc checkpoint train lớn.
- Chỉ đưa thay đổi policy/runtime vào branch sau khi compile, regression và A/B benchmark.
- Không giữ file thử nghiệm một lần trong repo. Git history đã là archive.

## Core source

Các entrypoint/runtime được bảo trì:

- `winner_train.py`
- `continuous_runtime.py`
- `v45_skill_runtime.py`
- `v45_skill_runtime_actkeep.py`
- `train.sh`

Core package nằm trong `src/kaggrl/`. Legacy V2/V3 và pipeline nghiên cứu cũ đã được loại khỏi branch này.

## QA trước commit

```bash
python -m compileall -q \
  winner_train.py \
  continuous_runtime.py \
  v45_skill_runtime.py \
  v45_skill_runtime_actkeep.py \
  rollout src tools tests

bash -n train.sh
git diff --check
```

Nếu có checkpoint:

```bash
export FARMOS_CHECKPOINT=/path/to/winner_v45_skill_stage_safe.pt
python tests/test_action_alignment.py
python tests/test_determinism.py
python tools/bench/cutover.py
python tools/bench/act_keep.py
```

## Safety gate

Khi thay đổi learned-skill takeover:

- animal escapes phải giữ 0 trên safety eval;
- plant deaths không vượt baseline đã chốt;
- takeover không thấp hơn ngưỡng cấu hình;
- margin không vượt quá stage tolerance theo hướng xấu;
- winner anchor vẫn reproduce;
- runtime Kaggle phải giữ CPU compatibility.

## Runtime assets được phép version

Chỉ giữ các artifact nhỏ cần cho cấu hình bảo trì:

- `assets/parent_promoted_v2.pt`
- `assets/act_keep_gate_sweep_best.pt`
- `assets/structural_gain_table_v4_2.json`
- `assets/v51_main.py`
