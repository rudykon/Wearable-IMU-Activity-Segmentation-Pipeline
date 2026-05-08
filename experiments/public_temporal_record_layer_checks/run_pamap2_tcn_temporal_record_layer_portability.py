"""PAMAP2 raw-window temporal-CNN TRL portability runner.

Purpose:
    Normalizes PAMAP2 subject channels, trains a compact temporal CNN on raw
    windows, and evaluates decoded activity segments.
Inputs:
    Reads PAMAP2 protocol files from a local public dataset copy.
Outputs:
    Produces public-portability metrics for comparison with feature baselines.
"""
from __future__ import annotations


import numpy as np

from temporal_record_layer_common import SubjectSequence
from run_pamap2_temporal_record_layer_portability import DATA_DIR, FS, RESULT_DIR, STEP_S, WINDOW_S, load_pamap2_sequences
from temporal_model_record_layer_common import print_summary, run_tcn_portability_experiment


def normalize_subject_channels(sequences: dict[str, SubjectSequence]) -> dict[str, SubjectSequence]:
    normalized = {}
    for subject, sequence in sequences.items():
        mean = sequence.signal.mean(axis=0, keepdims=True)
        std = sequence.signal.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        normalized[subject] = SubjectSequence(
            subject=sequence.subject,
            signal=((sequence.signal - mean) / std).astype(np.float32),
            labels=sequence.labels,
            timestamps_s=sequence.timestamps_s,
        )
    return normalized


def main() -> None:
    sequences, label_names, split = load_pamap2_sequences(DATA_DIR)
    sequences = normalize_subject_channels(sequences)
    caption = (
        "\\parbox{\\columnwidth}{\\centering\\scshape PAMAP2 portability check with a small "
        "raw-window temporal CNN. The model is trained on protocol subjects 101--105 when "
        "available, TRL constants are selected on the next held-out subjects, and the final "
        "subjects are reported. Inputs use accelerometer and gyroscope channels from hand, "
        "chest, and ankle IMUs with per-subject channel normalization, 5 s windows, and "
        "1 s step. This is a temporal-interface sanity check, not a public leaderboard "
        "comparison.}"
    )
    summary = run_tcn_portability_experiment(
        dataset_name="PAMAP2",
        source="UCI PAMAP2 Physical Activity Monitoring protocol files",
        sequences=sequences,
        split=split,
        label_names=label_names,
        window_s=WINDOW_S,
        step_s=STEP_S,
        fs_hz=FS,
        result_dir=RESULT_DIR,
        prefix="pamap2_tcn",
        caption=caption,
    )
    print_summary(summary, RESULT_DIR, "pamap2_tcn")


if __name__ == "__main__":
    main()
