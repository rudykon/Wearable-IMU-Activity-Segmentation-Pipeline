"""WISDM-phone feature-MLP TRL portability runner.

Purpose:
    Applies the shared feature-MLP utilities to WISDM phone window features for a
    compact neural portability baseline.
Inputs:
    Reads WISDM phone subject sequences from the dataset loader.
Outputs:
    Produces summary metrics compatible with other public-corpus runs.
"""
from __future__ import annotations


from feature_mlp_temporal_record_layer_common import print_summary, run_mlp_portability_experiment
from temporal_record_layer_common import fixed_subject_split
from run_wisdm_phone_temporal_record_layer_portability import DATA_DIR, FS, RESULT_DIR, STEP_S, WINDOW_S, load_wisdm_phone_sequences


def main() -> None:
    sequences, label_names = load_wisdm_phone_sequences(DATA_DIR)
    split = fixed_subject_split(sequences.keys(), train_frac=0.6, dev_frac=0.2)
    caption = (
        "\\parbox{\\columnwidth}{\\centering\\scshape WISDM-phone portability check with a small "
        "feature MLP. The model replaces the linear window classifier with a nonlinear neural "
        "classifier over robust window statistics, while TRL constants are selected on held-out "
        "dev subjects. Phone accelerometer streams use 10 s windows with 1 s step. This is a "
        "temporal-interface sanity check, not a public leaderboard comparison.}"
    )
    summary = run_mlp_portability_experiment(
        dataset_name="WISDM-phone",
        source="WISDM phone accelerometer raw time-series",
        sequences=sequences,
        split=split,
        label_names=label_names,
        window_s=WINDOW_S,
        step_s=STEP_S,
        fs_hz=FS,
        result_dir=RESULT_DIR,
        prefix="wisdm_phone_mlp",
        caption=caption,
    )
    print_summary(summary, RESULT_DIR, "wisdm_phone_mlp")


if __name__ == "__main__":
    main()
