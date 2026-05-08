"""Top-level package for the wearable IMU activity-segmentation pipeline.

Purpose:
    Marks `src/imu_activity_pipeline` as an importable package and defines the
    public package version for downstream tools.
Provides:
    Shared modules for configuration, data loading, model definitions, training,
    inference, evaluation, and prediction I/O.
Used by:
    Repository-root compatibility wrappers, experiment scripts, smoke tests, and
    external users who install the project with `pip install -e .`.
"""

__version__ = "0.1.0"
