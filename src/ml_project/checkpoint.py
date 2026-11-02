from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path

import torch

from ml_project.model import TinyVLA, TinyVLAConfig

CHECKPOINT_FORMAT_VERSION = 1


def save_model_checkpoint(
    path: str | Path,
    model: TinyVLA,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Save the model configuration and weights needed for later analysis."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_config": dataclasses.asdict(model.config),
            "model_state_dict": model.state_dict(),
            "metadata": dict(metadata or {}),
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_model_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[TinyVLA, dict[str, object]]:
    """Reconstruct a TinyVLA from a trusted checkpoint produced by this project."""

    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Invalid checkpoint payload")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format")
    model_config = payload.get("model_config")
    model_state_dict = payload.get("model_state_dict")
    metadata = payload.get("metadata", {})
    if not isinstance(model_config, dict) or not isinstance(model_state_dict, dict):
        raise ValueError("Checkpoint is missing its model configuration or weights")
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be a mapping")

    model = TinyVLA(TinyVLAConfig(**model_config)).to(device)
    model.load_state_dict(model_state_dict)
    return model, metadata
