"""Default command-line entry point for inference.

Purpose:
    Runs the configured end-to-end inference workflow: load saved models, read
    raw signal files, predict activity segments, and write a workbook.
Inputs:
    Uses the configured canonical data split, `MODEL_DIR`, and other paths from
    `config.py`.
Outputs:
    Writes `predictions_<split>.xlsx` in the runtime directory and prints a
    short run summary with the number of generated segments.
"""
import os

from .config import BASE_DIR, DEFAULT_INFERENCE_SPLIT, SPLIT_DATA_DIRS
from .inference import run_inference


def main() -> None:
    """Run inference with the default configured input and output paths."""
    script_dir = BASE_DIR
    os.chdir(script_dir)
    split_name = DEFAULT_INFERENCE_SPLIT
    if split_name not in SPLIT_DATA_DIRS:
        valid = ", ".join(SPLIT_DATA_DIRS)
        raise ValueError(f"Unknown inference split '{split_name}'. Expected one of: {valid}")
    data_dir = SPLIT_DATA_DIRS[split_name]
    output_file = os.path.join(script_dir, f"predictions_{split_name}.xlsx")

    print("=" * 60)
    print("IMU Activity Segmentation - Inference Pipeline")
    print("=" * 60)
    print(f"Split: {split_name}")
    print(f"Signal data dir: {data_dir}")
    print(f"Output file: {output_file}")

    results = run_inference(data_dir, output_file)

    print(f"\nDone! Generated {len(results)} segment predictions.")
    print(f"Results saved to: {output_file}")
