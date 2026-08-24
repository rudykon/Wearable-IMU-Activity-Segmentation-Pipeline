"""Download and verify public model assets from Hugging Face.

Binary checkpoints live in the public Hugging Face Model repository. Source
checkouts keep only the small manifest and configuration files in Git. Missing
assets are downloaded atomically and verified against the published SHA-256
manifest before inference uses them.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_MODEL_REPO_ID = "config-h/Wearable-IMU-Activity-Segmentation-Pipeline"
DEFAULT_MODEL_REVISION = "main"
_TRUTHY = {"1", "true", "yes", "on"}


class ModelAssetError(RuntimeError):
    """Raised when a required public model asset cannot be prepared safely."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _manifest_path(path: str | os.PathLike[str] | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    configured = os.getenv("HLS_HAR_MODEL_MANIFEST")
    if configured:
        return Path(configured).expanduser().resolve()
    return _project_root() / "model-assets.json"


def load_asset_manifest(path: str | os.PathLike[str] | None = None) -> dict:
    """Load the tracked model-asset manifest."""

    manifest_path = _manifest_path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelAssetError(f"Could not read model manifest: {manifest_path}") from exc

    if not isinstance(manifest.get("assets"), list) or not manifest.get("repo_id"):
        raise ModelAssetError(f"Invalid model manifest: {manifest_path}")
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matches_manifest(path: Path, asset: dict) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(asset["size"])
        and _sha256(path) == asset["sha256"]
    )


def _asset_url(repo_id: str, revision: str, asset_path: str) -> str:
    base_url = os.getenv("HLS_HAR_MODEL_BASE_URL")
    encoded_path = quote(asset_path, safe="/")
    if base_url:
        return f"{base_url.rstrip('/')}/{encoded_path}"
    return (
        "https://huggingface.co/"
        f"{quote(repo_id, safe='/')}/resolve/{quote(revision, safe='')}/{encoded_path}"
    )


def _download_asset(url: str, destination: Path, asset: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".download",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            request = Request(url, headers={"User-Agent": "wearable-imu-model-assets/1.0"})
            with urlopen(request, timeout=120) as response:
                shutil.copyfileobj(response, temporary, length=1024 * 1024)

        if not _matches_manifest(temporary_path, asset):
            raise ModelAssetError(f"Checksum mismatch for {asset['path']}")
        os.replace(temporary_path, destination)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise ModelAssetError(
            f"Could not download {asset['path']} from Hugging Face: {exc}"
        ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _selected_assets(manifest: dict, prefix: str) -> list[dict]:
    selected = [asset for asset in manifest["assets"] if asset["path"].startswith(prefix)]
    if not selected:
        raise ModelAssetError(f"No assets with prefix {prefix!r} in model manifest")
    return selected


def _ensure_assets(
    assets: Iterable[dict],
    destination_dir: Path,
    *,
    repo_id: str,
    revision: str,
    force: bool,
) -> tuple[Path, ...]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    verify_existing = os.getenv("HLS_HAR_VERIFY_MODEL_ASSETS", "").lower() in _TRUTHY
    offline = os.getenv("HLS_HAR_OFFLINE", "").lower() in _TRUTHY
    prepared: list[Path] = []

    for asset in assets:
        destination = destination_dir / Path(asset["path"]).name
        if destination.exists() and not force:
            if verify_existing and not _matches_manifest(destination, asset):
                raise ModelAssetError(
                    f"Local model asset differs from the public manifest: {destination}"
                )
            prepared.append(destination)
            continue

        if offline:
            raise ModelAssetError(
                f"Missing model asset in offline mode: {destination}. "
                "Run scripts/download_model_assets.py while online first."
            )

        _download_asset(_asset_url(repo_id, revision, asset["path"]), destination, asset)
        prepared.append(destination)

    return tuple(prepared)


def ensure_python_model_assets(
    model_dir: str | os.PathLike[str],
    *,
    manifest_path: str | os.PathLike[str] | None = None,
    force: bool = False,
) -> tuple[Path, ...]:
    """Ensure the PyTorch checkpoints and normalization files are available."""

    manifest = load_asset_manifest(manifest_path)
    repo_id = os.getenv("HLS_HAR_MODEL_REPO_ID", manifest.get("repo_id", DEFAULT_MODEL_REPO_ID))
    revision = os.getenv(
        "HLS_HAR_MODEL_REVISION", manifest.get("revision", DEFAULT_MODEL_REVISION)
    )
    return _ensure_assets(
        _selected_assets(manifest, "saved_models/"),
        Path(model_dir),
        repo_id=repo_id,
        revision=revision,
        force=force,
    )


def ensure_android_model_assets(
    asset_dir: str | os.PathLike[str],
    *,
    manifest_path: str | os.PathLike[str] | None = None,
    force: bool = False,
) -> tuple[Path, ...]:
    """Ensure the Android ONNX exports and normalization files are available."""

    manifest = load_asset_manifest(manifest_path)
    repo_id = os.getenv("HLS_HAR_MODEL_REPO_ID", manifest.get("repo_id", DEFAULT_MODEL_REPO_ID))
    revision = os.getenv(
        "HLS_HAR_MODEL_REVISION", manifest.get("revision", DEFAULT_MODEL_REVISION)
    )
    return _ensure_assets(
        _selected_assets(manifest, "android_realtime_app/app/src/main/assets/"),
        Path(asset_dir),
        repo_id=repo_id,
        revision=revision,
        force=force,
    )


def verify_assets(
    destination_dir: str | os.PathLike[str],
    prefix: str,
    *,
    manifest_path: str | os.PathLike[str] | None = None,
) -> tuple[Path, ...]:
    """Verify one published asset group without downloading or replacing files."""

    manifest = load_asset_manifest(manifest_path)
    checked: list[Path] = []
    for asset in _selected_assets(manifest, prefix):
        path = Path(destination_dir) / Path(asset["path"]).name
        if not _matches_manifest(path, asset):
            raise ModelAssetError(f"Missing or mismatched model asset: {path}")
        checked.append(path)
    return tuple(checked)
