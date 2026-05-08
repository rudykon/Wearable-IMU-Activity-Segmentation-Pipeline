"""Minimal health check for the public activity-segmentation pipeline.

Purpose:
    Verifies that the packaged repository can import core modules, resolve the
    unified data paths, read a tiny signal file, load annotations, and write a
    prediction workbook.
Inputs:
    Uses temporary files only; no real dataset or trained checkpoint is required.
Outputs:
    Prints `smoke test passed` when the basic public workflow is intact.
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from imu_activity_pipeline import config
from imu_activity_pipeline.sensor_data_processing import load_gold_labels
from imu_activity_pipeline.signal_file_reader import DataReader
from imu_activity_pipeline.prediction_writer import DataOutput


def assert_unified_data_paths() -> None:
    """Check that config paths use the unified data/ directory layout."""
    data_root = Path(config.DATA_ROOT)
    assert Path(config.SIGNALS_DIR) == data_root / "signals"
    assert Path(config.ANNOTATIONS_DIR) == data_root / "annotations"
    assert Path(config.SPLITS_DIR) == data_root / "splits"
    assert Path(config.METADATA_DIR) == data_root / "metadata"
    assert Path(config.TRAIN_DATA_DIR) == data_root / "signals" / "train"
    assert Path(config.INTERNAL_EVAL_DATA_DIR) == data_root / "signals" / "internal_eval"
    assert Path(config.EXTERNAL_TEST_DATA_DIR) == data_root / "signals" / "external_test"
    assert Path(config.TRAIN_ANNOTATIONS_FILE) == data_root / "annotations" / "train_annotations.csv"
    assert Path(config.INTERNAL_EVAL_GOLD_FILE) == data_root / "annotations" / "internal_eval_annotations.csv"
    assert Path(config.EXTERNAL_TEST_GOLD_FILE) == data_root / "annotations" / "external_test_annotations.csv"
    assert set(config.SPLIT_NAMES) == {"train", "internal_eval", "external_test"}
    assert Path(config.SPLIT_DATA_DIRS["train"]) == Path(config.TRAIN_DATA_DIR)
    assert Path(config.SPLIT_DATA_DIRS["internal_eval"]) == Path(config.INTERNAL_EVAL_DATA_DIR)
    assert Path(config.SPLIT_DATA_DIRS["external_test"]) == Path(config.EXTERNAL_TEST_DATA_DIR)
    assert Path(config.SPLIT_ANNOTATION_FILES["train"]) == Path(config.TRAIN_ANNOTATIONS_FILE)
    assert Path(config.SPLIT_ANNOTATION_FILES["internal_eval"]) == Path(config.INTERNAL_EVAL_GOLD_FILE)
    assert Path(config.SPLIT_ANNOTATION_FILES["external_test"]) == Path(config.EXTERNAL_TEST_GOLD_FILE)


def assert_data_reader(tmp: Path) -> None:
    """Verify that DataReader returns only text signal files keyed by file stem."""
    signal_dir = tmp / "signals"
    signal_dir.mkdir()
    (signal_dir / "HNU00001.txt").write_text("ACC_TIME\tACC_X\n1\t0.1\n", encoding="utf-8")
    (signal_dir / "notes.csv").write_text("ignored", encoding="utf-8")

    payload = DataReader(str(signal_dir)).read_data()
    assert sorted(payload) == ["HNU00001"]
    assert "ACC_TIME" in payload["HNU00001"]


def assert_annotation_loader(tmp: Path) -> None:
    """Verify annotation loading and timestamp dtype normalization."""
    annotation_path = tmp / "annotations.csv"
    with annotation_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "user_id", "category", "start", "end"])
        writer.writeheader()
        writer.writerow(
            {
                "split": "train",
                "user_id": "HNU00001",
                "category": "跑步",
                "start": "1000",
                "end": "2000",
            }
        )

    labels = load_gold_labels(str(annotation_path))
    assert labels.loc[0, "user_id"] == "HNU00001"
    assert labels.loc[0, "category"] == "跑步"
    assert str(labels["start"].dtype) == "int64"
    assert str(labels["end"].dtype) == "int64"


def assert_prediction_output(tmp: Path) -> None:
    """Verify that predictions can be written to a non-empty workbook."""
    output_path = tmp / "predictions_external_test.xlsx"
    rows = [["HNU00001", "跑步", 1000, 2000]]
    DataOutput(rows, output_file=str(output_path)).save_predictions()
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def main() -> None:
    """Run all smoke checks."""
    assert_unified_data_paths()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        assert_data_reader(tmp)
        assert_annotation_loader(tmp)
        assert_prediction_output(tmp)
    print("smoke test passed")


if __name__ == "__main__":
    main()
