#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
if [ ! -d .venv ]; then python3 -m venv .venv; fi
. .venv/bin/activate
if ! python -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
  python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
fi
python -m pip install -q -r requirements_v4.txt
python -c "import sys,pathlib; sys.path.insert(0,'vendor'); import kaggle_environments as k; p=pathlib.Path(k.__file__).resolve(); assert 'vendor' in p.parts; print('Vendored Kaggriculture engine:',k.__version__)"
exec python winner_train.py "$@"
