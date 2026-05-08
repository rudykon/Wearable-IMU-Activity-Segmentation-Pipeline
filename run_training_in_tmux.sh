#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SESSION_NAME="${SESSION_NAME:-imu-train}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi
GPU_LIST="${GPU_LIST:-0,1,2,3}"
NUM_EPOCHS_STAGE2="${NUM_EPOCHS_STAGE2:-100}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-30}"
MIN_EPOCHS_BEFORE_EARLY_STOP="${MIN_EPOCHS_BEFORE_EARLY_STOP:-40}"
LOG_DIR="$ROOT_DIR/saved_models"
SESSION_LOG="$LOG_DIR/tmux_${SESSION_NAME}.log"

mkdir -p "$LOG_DIR"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required but not found."
    exit 1
fi

if [ -S "${TMUX:-}" ]; then
    echo "Please launch this script outside an existing tmux session."
    exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists."
    echo "Attach with: tmux attach -t $SESSION_NAME"
    exit 1
fi

cat <<EOF
Launching training in tmux session '$SESSION_NAME'
  GPUs: $GPU_LIST
  epochs: $NUM_EPOCHS_STAGE2
  patience: $EARLY_STOPPING_PATIENCE
  min_epochs_before_early_stop: $MIN_EPOCHS_BEFORE_EARLY_STOP
  session log: $SESSION_LOG

Attach: tmux attach -t $SESSION_NAME
Detach: Ctrl+b then d
EOF

tmux new-session -d -s "$SESSION_NAME" \
    "cd '$ROOT_DIR' && export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES='$GPU_LIST' NUM_EPOCHS_STAGE2='$NUM_EPOCHS_STAGE2' EARLY_STOPPING_PATIENCE='$EARLY_STOPPING_PATIENCE' MIN_EPOCHS_BEFORE_EARLY_STOP='$MIN_EPOCHS_BEFORE_EARLY_STOP' && '$PYTHON_BIN' train_parallel.py 2>&1 | tee '$SESSION_LOG'"

tmux list-sessions | rg "$SESSION_NAME" || true
