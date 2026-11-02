from __future__ import annotations

import torch
from torch.nn import functional as F

from ml_project.model import TinyVLA


def language_to_image_attention(
    model: TinyVLA,
    attentions: dict[str, torch.Tensor],
    *,
    layer: str,
    language_token_index: int,
    camera_name: str,
) -> torch.Tensor:
    """Extract head-averaged attention from one language token to one camera."""

    if layer not in attentions:
        raise KeyError(f"Unknown attention layer {layer!r}")
    if camera_name not in model.config.camera_names:
        raise KeyError(f"Unknown camera {camera_name!r}")
    if not 0 <= language_token_index < model.config.max_text_length:
        raise IndexError(language_token_index)
    token_slices = model.token_slices()
    query_index = token_slices["text"].start + language_token_index
    camera_slice = token_slices[camera_name]
    # [batch, heads, query, key] -> [batch, camera_patches]
    return attentions[layer][:, :, query_index, camera_slice].mean(dim=1)


def attention_heatmap(model: TinyVLA, patch_attention: torch.Tensor) -> torch.Tensor:
    grid_size = model.vision_encoder.grid_size
    if patch_attention.shape[-1] != grid_size**2:
        raise ValueError("Attention width does not match the model patch grid")
    return patch_attention.reshape(*patch_attention.shape[:-1], grid_size, grid_size)


def mask_top_attention_patches(
    images: torch.Tensor,
    heatmaps: torch.Tensor,
    *,
    fraction: float = 0.15,
    fill: float = 0.0,
) -> torch.Tensor:
    """Mask the highest-attention patches in a batch of images."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if images.shape[0] != heatmaps.shape[0]:
        raise ValueError("Images and heatmaps must have the same batch size")
    flat = heatmaps.flatten(1)
    count = max(1, round(flat.shape[1] * fraction))
    indices = flat.topk(count, dim=1).indices
    patch_mask = torch.zeros_like(flat, dtype=torch.bool)
    patch_mask.scatter_(1, indices, True)
    patch_mask = patch_mask.reshape_as(heatmaps).unsqueeze(1).float()
    pixel_mask = F.interpolate(patch_mask, size=images.shape[-2:], mode="nearest").bool()
    return images.masked_fill(pixel_mask, fill)


def action_change(reference: torch.Tensor, perturbed: torch.Tensor) -> torch.Tensor:
    if reference.shape != perturbed.shape:
        raise ValueError("Action tensors must have identical shapes")
    return torch.linalg.vector_norm((reference - perturbed).flatten(1), dim=-1)
