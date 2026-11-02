from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from ml_project.data import SPLITS, default_data_root

DATASET_VERSION = "v1"
DATASET_ARCHIVE = "vla_takehome_v1.zip"
DATASET_URL = (
    "https://infinite-hands--ml-takehome-dataset-serve.modal.run/"
    f"{DATASET_VERSION}/{DATASET_ARCHIVE}"
)
DATASET_SHA256 = "243ca75fe9d6ea4830e8bc25477e40b11f483be5d5bc9f81c05674d0b4b53887"


def download_dataset(
    *,
    destination: Path | None = None,
    url: str = DATASET_URL,
    expected_sha256: str = DATASET_SHA256,
    local_archive: Path | None = None,
) -> Path:
    """Download, verify, and extract the immutable dataset without overwriting data."""

    destination = destination or default_data_root()
    if destination.exists():
        _verify_extracted(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary_directory:
        temporary_root = Path(temporary_directory)
        archive_path = temporary_root / DATASET_ARCHIVE
        if local_archive is None:
            with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
                with archive_path.open("wb") as output:
                    shutil.copyfileobj(response, output)
        else:
            shutil.copy2(local_archive, archive_path)
        actual_sha256 = _sha256(archive_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "Dataset archive checksum mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        extracted = temporary_root / "extracted"
        extracted.mkdir()
        _safe_extract(archive_path, extracted)
        _verify_extracted(extracted)
        extracted.rename(destination)
    return destination


def _safe_extract(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Unsafe archive member: {info.filename}")
        archive.extractall(destination)


def _verify_extracted(data_root: Path) -> None:
    manifest_path = data_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing dataset manifest in {data_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("dataset_version") != DATASET_VERSION:
        raise ValueError("Unsupported dataset manifest")
    for split in SPLITS:
        info = manifest["splits"][split]
        split_path = data_root / info["file"]
        if not split_path.is_file() or _sha256(split_path) != info["sha256"]:
            raise ValueError(f"Missing or corrupt dataset split: {split}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
