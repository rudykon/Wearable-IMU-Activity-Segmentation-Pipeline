"""Combined table builder for public TRL portability experiments.

Purpose:
    Collects available result rows from the public-dataset experiment runners and
    formats them into a compact comparison table.
Inputs:
    Reads result files emitted under `experiments/public_temporal_record_layer_checks/results/`.
Outputs:
    Writes a merged summary table for quick inspection.
"""
from __future__ import annotations


import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results"

SUMMARY_FILES = [
    ("HAR70+", "TCN", RESULT_DIR / "har70_tcn_trl_summary.json"),
    ("WISDM-phone", "MLP", RESULT_DIR / "wisdm_phone_mlp_trl_summary.json"),
    ("PAMAP2", "MLP", RESULT_DIR / "pamap2_mlp_trl_summary.json"),
    ("OPPORTUNITY", "MLP", RESULT_DIR / "opportunity_mlp_trl_summary.json"),
]

DIAGNOSTIC_SUMMARY_FILES = [
    ("WISDM-phone", "TCN", RESULT_DIR / "wisdm_phone_tcn_trl_summary.json"),
    ("PAMAP2", "TCN", RESULT_DIR / "pamap2_tcn_trl_summary.json"),
    ("PAMAP2", "GRU", RESULT_DIR / "pamap2_gru_trl_summary.json"),
    ("PAMAP2", "MLP-wide", RESULT_DIR / "pamap2_mlp_wide_trl_summary.json"),
    ("OPPORTUNITY", "TCN", RESULT_DIR / "opportunity_tcn_trl_summary.json"),
]


def load_rows(summary_files: list[tuple[str, str, Path]] | None = None) -> list[dict]:
    rows = []
    for dataset, model, path in summary_files or SUMMARY_FILES:
        if not path.exists():
            rows.append({"dataset": dataset, "model": model, "status": "missing", "summary_path": str(path)})
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        for setting, metrics in summary["test_metrics"].items():
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "status": "available",
                    "setting": setting,
                    **metrics,
                    "summary_path": str(path),
                }
            )
    return rows


def format_metric(value: object, digits: int = 3) -> str:
    if value is None or value == "":
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_table(rows: list[dict]) -> None:
    available = [row for row in rows if row.get("status") == "available"]
    lines = [
        "\\begin{center}",
        "\\begin{minipage}{\\columnwidth}",
        "\\centering",
        "\\footnotesize",
        "\\refstepcounter{table}\\label{tab:public_temporal_record_layer_checks}",
        "\\textsc{Table~\\thetable}\\\\[0.35ex]",
        "\\parbox{\\columnwidth}{\\centering\\scshape Additional public-corpus TRL portability checks. Rows use small neural window posterior generators followed by either direct argmax merging or dev-selected TRL-style record constants on held-out subjects. These rows are temporal-interface checks, not public leaderboard comparisons.}\\\\[0.8ex]",
        "{\\setlength{\\tabcolsep}{1.5pt}",
        "\\begin{tabular*}{\\columnwidth}{@{}lll@{\\extracolsep{\\fill}}ccccc@{}}",
        "\\toprule",
        "Dataset & Model & Setting & F1 & mIoU & B-MAE & FP/h & TP/FP/FN \\\\",
        "\\midrule",
    ]
    for row in available:
        counts = f"{int(row['tp'])}/{int(row['fp'])}/{int(row['fn'])}"
        setting_name = row["setting"].lower()
        setting = "TRL" if "trl" in setting_name else "Argmax"
        lines.append(
            f"{row['dataset']} & {row['model']} & {setting} & {format_metric(row['segment_f1'])} & "
            f"{format_metric(row['miou'])} & {format_metric(row['boundary_mae_s'], 2)} & "
            f"{format_metric(row['fp_per_hour'], 2)} & {counts} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular*}",
            "}",
            "\\end{minipage}",
            "\\end{center}",
            "",
        ]
    )
    (RESULT_DIR / "public_trl_extension_table.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    pd.DataFrame(rows).to_csv(RESULT_DIR / "public_trl_extension_summary.csv", index=False)
    sweep_rows = load_rows(SUMMARY_FILES + DIAGNOSTIC_SUMMARY_FILES)
    pd.DataFrame(sweep_rows).to_csv(RESULT_DIR / "public_neural_sweep_summary.csv", index=False)
    write_table(rows)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Wrote {RESULT_DIR / 'public_trl_extension_summary.csv'}")
    print(f"Wrote {RESULT_DIR / 'public_neural_sweep_summary.csv'}")
    print(f"Wrote {RESULT_DIR / 'public_trl_extension_table.tex'}")


if __name__ == "__main__":
    main()
