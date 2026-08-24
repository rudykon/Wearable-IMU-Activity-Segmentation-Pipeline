"""Tests for checksum-verifying Hugging Face model downloads."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from imu_activity_pipeline.model_assets import (
    ModelAssetError,
    ensure_python_model_assets,
    verify_assets,
)


class ModelAssetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.remote = self.root / "remote"
        self.remote_assets = self.remote / "saved_models"
        self.remote_assets.mkdir(parents=True)

        self.payloads = {
            "saved_models/example.pth": b"public-checkpoint",
            "saved_models/norm_params.pkl": b"public-normalization",
        }
        assets = []
        for relative, payload in self.payloads.items():
            path = self.remote / relative
            path.write_bytes(payload)
            assets.append(
                {
                    "path": relative,
                    "kind": "test",
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )

        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "repo_id": "example/model",
                    "revision": "main",
                    "assets": assets,
                }
            ),
            encoding="utf-8",
        )
        self.destination = self.root / "local"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _environment(self, **extra: str):
        values = {
            "HLS_HAR_MODEL_BASE_URL": self.remote.as_uri(),
            "HLS_HAR_OFFLINE": "",
            "HLS_HAR_VERIFY_MODEL_ASSETS": "",
        }
        values.update(extra)
        return mock.patch.dict(os.environ, values, clear=False)

    def test_downloads_and_verifies_missing_assets(self) -> None:
        with self._environment():
            paths = ensure_python_model_assets(
                self.destination,
                manifest_path=self.manifest,
            )
        self.assertEqual(len(paths), 2)
        self.assertEqual((self.destination / "example.pth").read_bytes(), b"public-checkpoint")
        verified = verify_assets(
            self.destination,
            "saved_models/",
            manifest_path=self.manifest,
        )
        self.assertEqual(set(paths), set(verified))

    def test_keeps_an_existing_local_override_by_default(self) -> None:
        self.destination.mkdir()
        local = self.destination / "example.pth"
        local.write_bytes(b"locally-retrained")
        with self._environment():
            ensure_python_model_assets(self.destination, manifest_path=self.manifest)
        self.assertEqual(local.read_bytes(), b"locally-retrained")

    def test_strict_mode_rejects_a_mismatched_existing_file(self) -> None:
        self.destination.mkdir()
        (self.destination / "example.pth").write_bytes(b"wrong")
        with self._environment(HLS_HAR_VERIFY_MODEL_ASSETS="1"):
            with self.assertRaises(ModelAssetError):
                ensure_python_model_assets(self.destination, manifest_path=self.manifest)

    def test_offline_mode_reports_a_missing_asset(self) -> None:
        with self._environment(HLS_HAR_OFFLINE="1"):
            with self.assertRaisesRegex(ModelAssetError, "offline mode"):
                ensure_python_model_assets(self.destination, manifest_path=self.manifest)


if __name__ == "__main__":
    unittest.main()
