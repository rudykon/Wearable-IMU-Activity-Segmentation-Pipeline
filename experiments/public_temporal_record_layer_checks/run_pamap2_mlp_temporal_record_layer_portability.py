"""PAMAP2 feature-MLP TRL portability runner.

Purpose:
    Applies the shared feature-MLP training/evaluation utilities to PAMAP2
    hand-crafted window features.
Inputs:
    Reads PAMAP2 subject sequences via the PAMAP2 loader.
Outputs:
    Produces segment-level and aggregate metrics for public portability tables.
"""
from __future__ import annotations


from feature_mlp_temporal_record_layer_common import print_summary, run_mlp_portability_experiment
from run_pamap2_temporal_record_layer_portability import DATA_DIR, FS, RESULT_DIR, STEP_S, WINDOW_S, load_pamap2_sequences


def main() -> None:
    sequences, label_names, split = load_pamap2_sequences(DATA_DIR)
    caption = (
        "\\parbox{\\columnwidth}{\\centering\\scshape PAMAP2 portability check with a small "
        "feature MLP. The model replaces the linear window classifier with a nonlinear neural "
        "classifier over robust window statistics, while TRL constants are selected on held-out "
        "dev subjects. Inputs use accelerometer and gyroscope channels from hand, chest, and "
        "ankle IMUs with 5 s windows and 1 s step. This is a temporal-interface sanity check, "
        "not a public leaderboard comparison.}"
    )
    summary = run_mlp_portability_experiment(
        dataset_name="PAMAP2",
        source="UCI PAMAP2 Physical Activity Monitoring protocol files",
        sequences=sequences,
        split=split,
        label_names=label_names,
        window_s=WINDOW_S,
        step_s=STEP_S,
        fs_hz=FS,
        result_dir=RESULT_DIR,
        prefix="pamap2_mlp",
        caption=caption,
    )
    print_summary(summary, RESULT_DIR, "pamap2_mlp")


if __name__ == "__main__":
    main()
