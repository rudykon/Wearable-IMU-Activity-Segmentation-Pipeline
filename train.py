#!/usr/bin/env python3
"""Compatibility wrapper for sequential model training.

Purpose:
    Keeps the repository-root command `python train.py` available while the real
    training implementation lives in `imu_activity_pipeline.train`.
Inputs:
    Reads training signals, annotations, and training configuration from
    `src/imu_activity_pipeline/config.py`.
Outputs:
    Produces model checkpoints, normalization parameters, ensemble metadata, and
    training artifacts under `saved_models/`.
"""
from pathlib import Path
import sys

if not getattr(sys, "frozen", False):
    src_dir = Path(__file__).resolve().parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from imu_activity_pipeline.train import main


if __name__ == "__main__":
    main()
