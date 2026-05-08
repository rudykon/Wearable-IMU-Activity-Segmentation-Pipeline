"""OPPORTUNITY locomotion raw-window TCN TRL portability runner.

Purpose:
    Evaluates a compact temporal convolutional model on raw OPPORTUNITY windows
    under the shared temporal-decoding protocol.
Inputs:
    Reads OPPORTUNITY subject sequences and constructs fixed-length temporal
    windows.
Outputs:
    Produces portability metrics for the public-dataset extension summary.
"""
from __future__ import annotations


from temporal_model_record_layer_common import print_summary, run_tcn_portability_experiment
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
        "check with a raw-window temporal CNN. The model consumes body-worn motion windows "
        "directly, selects sequence-only TRL constants on subject S3, and reports held-out "
        "subject S4. Label 0 is treated as background and excluded from record matching. "
        "This is a temporal-interface sanity check, not a public leaderboard comparison.}"
    )
    summary = run_tcn_portability_experiment(
        dataset_name="OPPORTUNITY-locomotion",
        source="UCI OPPORTUNITY Activity Recognition locomotion ADL files",
        sequences=sequences,
        split=split,
        label_names=label_names,
        window_s=WINDOW_S,
        step_s=STEP_S,
        fs_hz=FS,
        result_dir=RESULT_DIR,
        prefix="opportunity_tcn",
        caption=caption,
        model_name="tcn",
        ignore_labels={BACKGROUND_LABEL},
    )
    print_summary(summary, RESULT_DIR, "opportunity_tcn")


if __name__ == "__main__":
    main()
