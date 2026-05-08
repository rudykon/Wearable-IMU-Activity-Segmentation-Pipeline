#!/usr/bin/env python3
"""Compatibility wrapper for running segment-level evaluation from a source checkout.

Purpose:
    Keeps the repository-root command `python evaluate.py` available after the
    implementation was moved into the `imu_activity_pipeline` package.
Inputs:
    Command-line arguments are forwarded to `imu_activity_pipeline.evaluate`.
Outputs:
    Prints per-user and average segmental F1 metrics for a prediction workbook.
"""
from pathlib import Path
import sys

if not getattr(sys, "frozen", False):
    src_dir = Path(__file__).resolve().parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from imu_activity_pipeline.evaluate import main


if __name__ == "__main__":
    main()
