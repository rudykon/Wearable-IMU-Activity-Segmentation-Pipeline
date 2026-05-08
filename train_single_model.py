#!/usr/bin/env python3
"""Compatibility wrapper for training one selected model.

Purpose:
    Keeps the repository-root command `python train_single_model.py` available for
    targeted retraining of a single window scale and random seed.
Inputs:
    Forwards CLI arguments such as `--suffix`, `--seed`, `--epochs`, and early
    stopping controls to `imu_activity_pipeline.train_single_model`.
Outputs:
    Writes the selected checkpoint and related normalization/cache files under
    the configured model directory.
"""
from pathlib import Path
import sys

if not getattr(sys, "frozen", False):
    src_dir = Path(__file__).resolve().parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from imu_activity_pipeline.train_single_model import main


if __name__ == "__main__":
    main()
