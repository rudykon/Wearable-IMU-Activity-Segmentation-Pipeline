#!/usr/bin/env python3
"""Compatibility wrapper for parallel multi-seed training.

Purpose:
    Keeps the repository-root command `python train_parallel.py` available while
    the implementation lives in `imu_activity_pipeline.train_parallel`.
Inputs:
    Reads configured training data and optional environment variables such as
    `CUDA_VISIBLE_DEVICES`, `NUM_EPOCHS_STAGE2`, and early stopping controls.
Outputs:
    Launches per-model training workers and writes final ensemble assets under
    `saved_models/`.
"""
from pathlib import Path
import sys

if not getattr(sys, "frozen", False):
    src_dir = Path(__file__).resolve().parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from imu_activity_pipeline.train_parallel import main


if __name__ == "__main__":
    main()
