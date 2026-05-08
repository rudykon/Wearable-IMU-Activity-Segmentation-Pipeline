#!/usr/bin/env python3
"""Compatibility wrapper for running the default inference pipeline.

Purpose:
    Keeps the repository-root command `python run_inference.py` available for users who
    run the project directly from a source checkout.
Inputs:
    Uses the configured canonical split signal directory and saved model
    directory.
Outputs:
    Writes `predictions_<split>.xlsx` with `user_id`, `category`, `start`, and
    `end` columns in the repository root.
"""
from pathlib import Path
import sys

if not getattr(sys, "frozen", False):
    src_dir = Path(__file__).resolve().parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from imu_activity_pipeline.inference_cli import main


if __name__ == "__main__":
    main()
