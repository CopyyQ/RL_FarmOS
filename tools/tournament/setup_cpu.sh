#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "$UV_BIN" ]]; then
  echo "uv is required. Install uv first." >&2
  exit 2
fi

if [[ ! -x .venv/bin/python ]]; then
  "$UV_BIN" venv --python 3.11 .venv
fi

PY="$ROOT/.venv/bin/python"
if ! "$PY" -c 'import torch' >/dev/null 2>&1; then
  if ! "$UV_BIN" pip install --python "$PY"       --index-url https://download.pytorch.org/whl/cpu       'torch==2.11.0'; then
    "$UV_BIN" pip install --python "$PY"       --index-url https://download.pytorch.org/whl/cpu torch
  fi
fi

"$UV_BIN" pip install --python "$PY" -r requirements.txt

PYTHONPATH=vendor:src:. "$PY" tests/test_imports.py
PYTHONPATH=vendor:src:. "$PY" tests/test_tournament.py

echo "CPU tournament environment ready: $PY"
