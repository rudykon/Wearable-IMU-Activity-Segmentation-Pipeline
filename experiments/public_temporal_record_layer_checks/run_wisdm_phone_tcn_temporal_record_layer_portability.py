"""WISDM-phone raw-window TCN TRL portability runner.

Purpose:
    Trains a compact temporal convolutional model on raw WISDM phone windows and
    evaluates decoded activity segments.
Inputs:
    Reads WISDM phone records and constructs fixed-length temporal windows.
Outputs:
    Writes metrics for the combined public portability comparison.
"""
from __future__ import annotations


from temporal_model_record_layer_common import print_summary, run_tcn_portability_experiment
from temporal_record_layer_common import fixed_subject_split
from run_wisdm_phone_temporal_record_layer_portability import DATA_DIR, FS, RESULT_DIR, STEP_S, WINDOW_S, load_wisdm_phone_sequences


def main() -> None:
    sequences, label_names = load_wisdm_phone_sequences(DATA_DIR)
    split = fixed_subject_split(sequences.keys(), train_frac=0.6, dev_frac=0.2)
    caption = (
        "\\parbox{\\columnwidth}{\\centering\\scshape WISDM-phone portability check with a small "
        "raw-window temporal CNN. TRL constants are selected on held-out dev subjects, and final "
        "subjects are reported under class-consistent segment matching. Phone accelerometer "
        "streams use 10 s windows with 1 s step. This is a temporal-interface sanity check, not "
        "a public leaderboard comparison.}"
    )
    summary = run_tcn_portability_experiment(
        dataset_name="WISDM-phone",
        source="WISDM phone accelerometer raw time-series",
        sequences=sequences,
        split=split,
        label_names=label_names,
        window_s=WINDOW_S,
        step_s=STEP_S,
        fs_hz=FS,
        result_dir=RESULT_DIR,
        prefix="wisdm_phone_tcn",
        caption=caption,
        model_name="tcn",
    )
    print_summary(summary, RESULT_DIR, "wisdm_phone_tcn")


if __name__ == "__main__":
    main()
