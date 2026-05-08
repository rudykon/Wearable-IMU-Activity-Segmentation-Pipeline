#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_BIN="$PYTHON_BIN"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PY_BIN="$ROOT_DIR/.venv/bin/python"
else
  PY_BIN="python3"
fi

if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python interpreter not found: $PY_BIN" >&2
  exit 1
fi

echo "[1/7] run external-test saved-model evaluation suite"
"$PY_BIN" experiments/run_external_test_saved_model_evaluation_suite.py

echo "[2/7] run internal-eval single-3s baseline robustness checks"
"$PY_BIN" experiments/run_internal_eval_single_3s_baseline_robustness_checks.py

echo "[3/7] run internal-eval split-protocol policy-selection checks"
"$PY_BIN" experiments/run_internal_eval_split_protocol_policy_selection_checks.py

echo "[4/7] run PPG quality audit"
"$PY_BIN" experiments/analyze_ppg_signal_quality.py

echo "[5/7] regenerate representative timeline figures"
"$PY_BIN" experiments/make_external_test_representative_timeline_figures.py

echo "[6/7] run external unlabeled cohort stress test"
"$PY_BIN" experiments/run_external_unlabeled_cohort_analysis.py --max-ids 6 --max-hours-per-id 6 --min-files-per-id 3 --include-generic-ids

echo "[7/7] regenerate experiment summary figures"
"$PY_BIN" experiments/make_saved_model_evaluation_summary_figures.py

echo "Done. Artifacts are under:"
echo "  - experiments/results/"
echo "  - experiments/figures/"
