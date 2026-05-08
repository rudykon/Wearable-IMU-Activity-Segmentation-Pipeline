"""PAMAP2 wider feature-MLP TRL portability runner.

Purpose:
    Tests a larger feature-MLP capacity on PAMAP2 to compare compact and wider
    feature-based neural baselines.
Inputs:
    Reads PAMAP2 subject sequences and hand-crafted window features.
Outputs:
    Writes metrics compatible with the public TRL extension summary.
"""
from __future__ import annotations


from feature_mlp_temporal_record_layer_common import print_summary, run_mlp_portability_experiment
from run_pamap2_temporal_record_layer_portability import DATA_DIR, FS, RESULT_DIR, STEP_S, WINDOW_S, load_pamap2_sequences


def main() -> None:
    sequences, label_names, split = load_pamap2_sequences(DATA_DIR)
    caption = (
        "\\parbox{\\columnwidth}{\\centering\\scshape PAMAP2 portability check with a wider "
        "feature MLP. The neural classifier operates on robust window statistics before the "
        "dev-selected TRL-style record layer. Inputs use accelerometer and gyroscope channels "
        "from hand, chest, and ankle IMUs with 5 s windows and 1 s step. This is a "
        "temporal-interface sanity check, not a public leaderboard comparison.}"
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
        prefix="pamap2_mlp_wide",
        caption=caption,
    )
    print_summary(summary, RESULT_DIR, "pamap2_mlp_wide")


if __name__ == "__main__":
    main()
