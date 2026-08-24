#!/usr/bin/env python3
"""Download public Python and Android model assets from Hugging Face."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from imu_activity_pipeline.model_assets import (  # noqa: E402
    ModelAssetError,
    ensure_android_model_assets,
    ensure_python_model_assets,
    verify_assets,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        choices=("python", "android", "all"),
        nargs="?",
        default="python",
        help="asset group to prepare (default: python)",
    )
    parser.add_argument("--force", action="store_true", help="redownload existing assets")
    parser.add_argument("--verify", action="store_true", help="verify only; do not download")
    args = parser.parse_args()

    groups: list[tuple[str, Path, str]] = []
    if args.target in {"python", "all"}:
        groups.append(("Python", ROOT / "saved_models", "saved_models/"))
    if args.target in {"android", "all"}:
        groups.append(
            (
                "Android",
                ROOT / "android_realtime_app" / "app" / "src" / "main" / "assets",
                "android_realtime_app/app/src/main/assets/",
            )
        )

    try:
        for label, destination, prefix in groups:
            if args.verify:
                paths = verify_assets(destination, prefix)
            elif label == "Python":
                paths = ensure_python_model_assets(destination, force=args.force)
            else:
                paths = ensure_android_model_assets(destination, force=args.force)
            print(f"{label}: {len(paths)} assets ready in {destination}")
    except ModelAssetError as exc:
        parser.exit(1, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
