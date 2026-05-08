#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

if command -v tmux >/dev/null 2>&1; then
    echo "Using tmux-backed launcher for SSH-safe training."
    exec "$ROOT_DIR/run_training_in_tmux.sh"
fi

echo "tmux not found, falling back to direct foreground launch."
echo "This mode is less robust to SSH disconnects."

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_EPOCHS_STAGE2="${NUM_EPOCHS_STAGE2:-100}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-30}"
MIN_EPOCHS_BEFORE_EARLY_STOP="${MIN_EPOCHS_BEFORE_EARLY_STOP:-40}"

export CUDA_VISIBLE_DEVICES NUM_EPOCHS_STAGE2 EARLY_STOPPING_PATIENCE MIN_EPOCHS_BEFORE_EARLY_STOP
"$PYTHON_BIN" train_parallel.py
