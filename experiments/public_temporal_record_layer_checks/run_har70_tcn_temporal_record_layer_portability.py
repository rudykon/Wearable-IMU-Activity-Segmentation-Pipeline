"""HAR70+ raw-window temporal-CNN TRL portability experiment.

Purpose:
    Tests whether a compact temporal CNN trained on raw HAR70+ windows benefits
    from the same temporal decoding and segment-evaluation protocol.
Inputs:
    Reads HAR70+ subject sequences from a local public dataset copy.
Outputs:
    Writes metrics compatible with the public TRL extension summary table.
"""
from __future__ import annotations


import numpy as np

from temporal_model_record_layer_common import SubjectSequence, print_summary, run_tcn_portability_experiment
from run_har70_temporal_record_layer_portability import (
    DATA_DIR,
    DEV_SUBJECTS,
    FS,
    LABEL_NAMES,
    RESULT_DIR,
    SIGNAL_COLS,
    STEP_S,
    TEST_SUBJECTS,
    TRAIN_SUBJECTS,
    WINDOW_S,
    load_subject,
)


def load_har70_sequences() -> dict[str, SubjectSequence]:
    sequences = {}
    for subject in TRAIN_SUBJECTS + DEV_SUBJECTS + TEST_SUBJECTS:
        df = load_subject(subject)
        signal = df[SIGNAL_COLS].to_numpy(dtype=np.float32)
        labels = df["label"].to_numpy(dtype=np.int32)
        timestamps_s = np.arange(len(df), dtype=np.float64) / FS
        sequences[subject] = SubjectSequence(subject, signal, labels, timestamps_s)
    return sequences


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing HAR70+ data directory: {DATA_DIR}")

    caption = (
        "\\parbox{\\columnwidth}{\\centering\\scshape HAR70+ portability check with a small "
        "raw-window temporal CNN. The model is trained on subjects 501--512, TRL constants "
        "are selected on subjects 513--515, and the table reports held-out subjects 516--518. "
        "Record-level matching excludes episodes shorter than the 4 s analysis window. This "
        "is a temporal-interface sanity check, not a public leaderboard comparison.}"
    )
    split = {
        "train_subjects": TRAIN_SUBJECTS,
        "dev_subjects": DEV_SUBJECTS,
        "test_subjects": TEST_SUBJECTS,
    }
    summary = run_tcn_portability_experiment(
        dataset_name="HAR70+",
        source="UCI Machine Learning Repository dataset 780",
        sequences=load_har70_sequences(),
        split=split,
        label_names=LABEL_NAMES,
        window_s=WINDOW_S,
        step_s=STEP_S,
        fs_hz=FS,
        result_dir=RESULT_DIR,
        prefix="har70_tcn",
        caption=caption,
    )
    print_summary(summary, RESULT_DIR, "har70_tcn")


if __name__ == "__main__":
    main()
