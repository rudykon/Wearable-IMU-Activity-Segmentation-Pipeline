"""OPPORTUNITY locomotion feature-MLP TRL portability runner.

Purpose:
    Reuses the OPPORTUNITY loader with the shared feature-MLP training/evaluation
    utilities to test a compact neural baseline.
Inputs:
    Reads OPPORTUNITY subject sequences from the local public-data directory.
Outputs:
    Produces a result row for public portability comparison tables.
"""
from __future__ import annotations


from feature_mlp_temporal_record_layer_common import print_summary, run_mlp_portability_experiment
from run_opportunity_temporal_record_layer_portability import (
    BACKGROUND_LABEL,
    DATA_DIR,
    FS,
    RESULT_DIR,
    STEP_S,
    WINDOW_S,
    load_opportunity_sequences,
)


def main() -> None:
    sequences, label_names, split = load_opportunity_sequences(DATA_DIR)
    caption = (
        "\\parbox{\\columnwidth}{\\centering\\scshape OPPORTUNITY locomotion portability "
        "check with a small feature MLP. The model replaces the linear classifier with a "
        "nonlinear neural classifier over robust body-worn motion-window statistics. TRL "
        "constants are selected on subject S3 and evaluated on held-out subject S4 with "
        "2 s windows and 1 s step. Label 0 is treated as background and excluded from "
        "record matching. This is a temporal-interface sanity check, not a public "
        "leaderboard comparison.}"
    )
    summary = run_mlp_portability_experiment(
        dataset_name="OPPORTUNITY-locomotion",
        source="UCI OPPORTUNITY Activity Recognition locomotion ADL files",
        sequences=sequences,
        split=split,
        label_names=label_names,
        window_s=WINDOW_S,
        step_s=STEP_S,
        fs_hz=FS,
        result_dir=RESULT_DIR,
        prefix="opportunity_mlp",
        caption=caption,
        ignore_labels={BACKGROUND_LABEL},
    )
    print_summary(summary, RESULT_DIR, "opportunity_mlp")


if __name__ == "__main__":
    main()
