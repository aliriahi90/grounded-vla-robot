from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

SPLITS = ("train", "validation", "deployment_week_1", "deployment_week_2")
VOCABULARY_TOKENS = (
    "and",
    "block",
    "blue",
    "bowl",
    "green",
    "in",
    "inside",
    "into",
    "it",
    "move",
    "pick",
    "place",
    "put",
    "red",
    "the",
    "to",
    "transfer",
    "up",
    "yellow",
)


@dataclasses.dataclass(frozen=True)
class VLADataConfig:
    image_size: int = 64
    max_text_length: int = 16
    action_horizon: int = 8
    state_dim: int = 7
    action_dim: int = 7

    def __post_init__(self) -> None:
        if self.image_size < 16:
            raise ValueError("image_size must be at least 16")
        if self.action_horizon < 4:
            raise ValueError("action_horizon must be at least 4")
        if self.state_dim != 7 or self.action_dim != 7:
            raise ValueError("The frozen v1 dataset uses 7D state and actions")


class Vocabulary:
    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"

    def __init__(self, tokens: list[str]) -> None:
        ordered = [self.PAD, self.UNK, self.BOS, self.EOS]
        ordered.extend(token for token in tokens if token not in ordered)
        self.token_to_id = {token: index for index, token in enumerate(ordered)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}

    @classmethod
    def default(cls) -> Vocabulary:
        return cls(list(VOCABULARY_TOKENS))

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def encode(self, text: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = [self.BOS, *self.tokenize(text), self.EOS][:max_length]
        ids = [self.token_to_id.get(token, self.token_to_id[self.UNK]) for token in tokens]
        mask = [True] * len(ids)
        padding = max_length - len(ids)
        ids.extend([self.token_to_id[self.PAD]] * padding)
        mask.extend([False] * padding)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

    def decode(self, ids: torch.Tensor) -> list[str]:
        return [self.id_to_token.get(int(index), self.UNK) for index in ids]

    def __len__(self) -> int:
        return len(self.token_to_id)


class FrozenVLADataset(Dataset[dict[str, object]]):
    """Read a verified, immutable split from the downloaded v1 dataset."""

    def __init__(
        self,
        split: str,
        size: int | None = None,
        *,
        data_root: str | Path | None = None,
        config: VLADataConfig | None = None,
        vocabulary: Vocabulary | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
        self.split = split
        self.config = config or VLADataConfig()
        self.vocabulary = vocabulary or Vocabulary.default()
        self.data_root = Path(data_root) if data_root is not None else default_data_root()
        manifest = load_and_verify_manifest(self.data_root)
        _validate_schema(manifest, self.config)
        split_info = manifest["splits"][split]
        split_path = self.data_root / split_info["file"]
        self._arrays = np.load(split_path, allow_pickle=False)
        available = int(split_info["samples"])
        self.size = available if size is None else size
        if self.size <= 0 or self.size > available:
            raise ValueError(f"size must be in [1, {available}] for {split}")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += self.size
        if index < 0 or index >= self.size:
            raise IndexError(index)
        prompt = str(self._arrays["prompt"][index])
        input_ids, attention_mask = self.vocabulary.encode(prompt, self.config.max_text_length)
        front = self._image_tensor(self._arrays["front"][index])
        wrist = self._image_tensor(self._arrays["wrist"][index])
        return {
            "sample_id": str(self._arrays["sample_id"][index]),
            "images": {"front": front, "wrist": wrist},
            "image_masks": {
                "front": torch.tensor(bool(self._arrays["front_available"][index])),
                "wrist": torch.tensor(bool(self._arrays["wrist_available"][index])),
            },
            "prompt": prompt,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "state": torch.from_numpy(self._arrays["state"][index].astype(np.float32)),
            "actions": torch.from_numpy(self._arrays["actions"][index].astype(np.float32)),
            "action_mask": torch.tensor(bool(self._arrays["action_mask"][index])),
            "metadata": self._metadata(index),
        }

    def _image_tensor(self, image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)
        if tensor.shape[-1] != self.config.image_size:
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(self.config.image_size, self.config.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return tensor

    def _metadata(self, index: int) -> dict[str, object]:
        string_fields = (
            "robot_id",
            "site_id",
            "camera_id",
            "task",
            "target_color",
            "destination_color",
            "model_checkpoint",
            "control_source",
        )
        metadata: dict[str, object] = {
            "split": self.split,
            "timestamp": float(self._arrays["timestamp"][index]),
            "intervention": bool(self._arrays["intervention"][index]),
            "success": float(self._arrays["success"][index]),
            "label_available": bool(self._arrays["label_available"][index]),
        }
        metadata.update({field: str(self._arrays[field][index]) for field in string_fields})
        return metadata


def default_data_root() -> Path:
    configured = os.environ.get("ML_PROJECT_DATA_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[2] / "data" / "vla_takehome_v1"


@functools.lru_cache(maxsize=8)
def load_and_verify_manifest(data_root: Path) -> dict[str, object]:
    manifest_path = data_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Frozen dataset not found at {data_root}. Run `python scripts/download_data.py`."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("dataset_version") != "v1":
        raise ValueError("Unsupported dataset manifest")
    for split in SPLITS:
        info = manifest["splits"][split]
        split_path = data_root / info["file"]
        if not split_path.is_file():
            raise FileNotFoundError(split_path)
        if _sha256(split_path) != info["sha256"]:
            raise ValueError(f"Checksum mismatch for {split_path}")
    return manifest


def _validate_schema(manifest: dict[str, object], config: VLADataConfig) -> None:
    expected = {
        "action_horizon": config.action_horizon,
        "state_dim": config.state_dim,
        "action_dim": config.action_dim,
    }
    mismatched = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatched:
        raise ValueError(f"Dataset/config schema mismatch: {mismatched}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_default_datasets(
    *,
    image_size: int = 64,
    max_text_length: int = 16,
    train_size: int = 512,
    validation_size: int = 128,
    deployment_size: int = 128,
    data_root: str | Path | None = None,
) -> dict[str, FrozenVLADataset]:
    config = VLADataConfig(image_size=image_size, max_text_length=max_text_length)
    vocabulary = Vocabulary.default()
    sizes = {
        "train": train_size,
        "validation": validation_size,
        "deployment_week_1": deployment_size,
        "deployment_week_2": deployment_size,
    }
    return {
        split: FrozenVLADataset(
            split,
            size,
            data_root=data_root,
            config=config,
            vocabulary=vocabulary,
        )
        for split, size in sizes.items()
    }
