"""Train-only VLA observation augmentation; never modifies targets or source batches."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AugmentConfig:
    camera_dropout: float = 0.15
    brightness: float = 0.12
    contrast: float = 0.12

    def __post_init__(self):
        if not 0 <= self.camera_dropout <= 1:
            raise ValueError("camera_dropout must be in [0,1]")
        if self.brightness < 0 or self.contrast < 0:
            raise ValueError("brightness and contrast must be nonnegative")


class AugmentedBatches:
    """Wrapper for train DataLoader; seeded RNG independent of training RNG.

    Augmentation is generated on CPU and transferred with batch by train_steps.
    A new iterator continues RNG state rather than restarting augmentation patterns.
    """

    def __init__(
        self, batches: Iterable[Mapping[str, object]], config: AugmentConfig, *, seed: int = 1007
    ):
        self.batches = batches
        self.config = config
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

    def __iter__(self) -> Iterator[dict[str, object]]:
        for batch in self.batches:
            yield augment_batch(batch, self.config, self.rng)


def augment_batch(
    batch: Mapping[str, object], cfg: AugmentConfig, rng: torch.Generator
) -> dict[str, object]:
    original_images = batch["images"]
    original_masks = batch["image_masks"]
    if not isinstance(original_images, Mapping) or not isinstance(original_masks, Mapping):
        raise TypeError("Expected images and image_masks to be mappings")
    camera_names = tuple(original_images)
    if not camera_names:
        raise ValueError("No cameras")
    images = {name: original_images[name].clone() for name in camera_names}
    masks = {name: original_masks[name].clone().bool() for name in camera_names}
    n = images[camera_names[0]].shape[0]
    if any(images[name].ndim != 4 or images[name].shape[0] != n for name in camera_names):
        raise ValueError("Images must have shape [batch, channels, height, width]")
    # Sample new availability only from physically present cameras.
    if cfg.camera_dropout > 0:
        for name in camera_names:
            dropped = torch.rand(n, generator=rng) < cfg.camera_dropout
            masks[name] = masks[name] & ~dropped
        # Restore one ORIGINAL available camera for samples where dropout hid all.
        available = torch.stack(
            [original_masks[name].reshape(n).bool() for name in camera_names], dim=1
        )
        current = torch.stack([masks[name].reshape(n) for name in camera_names], dim=1)
        restore = available.any(dim=1) & ~current.any(dim=1)
        if restore.any():
            # Random choice among originally available cameras.
            scores = torch.rand((n, len(camera_names)), generator=rng)
            scores[~available] = -1.0
            chosen = scores.argmax(dim=1)
            for index, name in enumerate(camera_names):
                masks[name][restore & (chosen == index)] = True
    # Per-sample and per-camera photometric jitter, shared across RGB channels.
    if cfg.brightness > 0 or cfg.contrast > 0:
        for name in camera_names:
            x = images[name]
            if cfg.contrast > 0:
                factor = 1 + (torch.rand((n, 1, 1, 1), generator=rng) * 2 - 1) * cfg.contrast
            else:
                factor = torch.ones((n, 1, 1, 1))
            if cfg.brightness > 0:
                delta = (torch.rand((n, 1, 1, 1), generator=rng) * 2 - 1) * cfg.brightness
            else:
                delta = torch.zeros((n, 1, 1, 1))
            images[name] = ((x - 0.5) * factor + 0.5 + delta).clamp(0, 1)
    result = dict(batch)
    result["images"] = images
    result["image_masks"] = masks
    return result
