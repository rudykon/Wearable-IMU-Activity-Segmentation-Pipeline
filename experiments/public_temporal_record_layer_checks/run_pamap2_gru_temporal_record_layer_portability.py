"""PAMAP2 raw-window GRU TRL portability runner.

Purpose:
    Evaluates a compact recurrent model on PAMAP2 raw windows using the shared
    temporal-decoding and segment-metric utilities.
Inputs:
    Reads normalized PAMAP2 subject windows from the dataset loader.
Outputs:
    Produces metrics for the public TRL extension comparison table.
"""
from __future__ import annotations


from temporal_model_record_layer_common import print_summary, run_tcn_portability_experiment
from run_pamap2_temporal_record_layer_portability import DATA_DIR, FS, RESULT_DIR, STEP_S, WINDOW_S, load_pamap2_sequences


def main() -> None:
    sequences, label_names, split = load_pamap2_sequences(DATA_DIR)
    caption = (
        "\\parbox{\\columnwidth}{\\centering\\scshape PAMAP2 portability check with a small "
        "raw-window bidirectional GRU. The model is trained on protocol subjects 101--105 when "
        "available, TRL constants are selected on the next held-out subjects, and final subjects "
        "are reported. Inputs use accelerometer and gyroscope channels from hand, chest, and "
        "ankle IMUs with 5 s windows and 1 s step. This is a temporal-interface sanity check, "
        "not a public leaderboard comparison.}"
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
        prefix="pamap2_gru",
        caption=caption,
        model_name="gru",
    )
    print_summary(summary, RESULT_DIR, "pamap2_gru")


if __name__ == "__main__":
    main()
