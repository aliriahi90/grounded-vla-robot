from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch

from ml_project.device import move_to_device
from ml_project.model import TinyVLA


def action_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    label_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return per-example offline action metrics."""

    if predictions.shape != targets.shape:
        raise ValueError(f"Prediction shape {predictions.shape} != target shape {targets.shape}")
    squared_error = (predictions - targets).square()
    mse = squared_error.mean((1, 2))
    predicted_endpoint = predictions[..., :3].sum(dim=1)
    target_endpoint = targets[..., :3].sum(dim=1)
    endpoint_error = torch.linalg.vector_norm(predicted_endpoint - target_endpoint, dim=-1)
    if predictions.shape[1] > 1:
        smoothness = predictions.diff(dim=1).square().mean((1, 2))
    else:
        smoothness = torch.zeros_like(mse)
    valid = (
        label_mask.bool()
        if label_mask is not None
        else torch.ones(predictions.shape[0], dtype=torch.bool, device=predictions.device)
    )
    return {
        "mse": mse[valid],
        "endpoint_error": endpoint_error[valid],
        "smoothness": smoothness[valid],
    }


@torch.no_grad()
def evaluate_model(
    model: TinyVLA,
    batches: Iterable[Mapping[str, object]],
    *,
    device: torch.device,
    flow_steps: int = 6,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    values: dict[str, list[torch.Tensor]] = {
        "mse": [],
        "endpoint_error": [],
        "smoothness": [],
    }
    for index, batch in enumerate(batches):
        if max_batches is not None and index >= max_batches:
            break
        device_batch = move_to_device(batch, device)
        if not isinstance(device_batch, Mapping):
            raise TypeError("Expected a mapping batch")
        output = model(
            images=device_batch["images"],  # type: ignore[arg-type]
            image_masks=device_batch["image_masks"],  # type: ignore[arg-type]
            input_ids=device_batch["input_ids"],  # type: ignore[arg-type]
            attention_mask=device_batch["attention_mask"],  # type: ignore[arg-type]
            state=device_batch["state"],  # type: ignore[arg-type]
            flow_steps=flow_steps,
            sample_seed=index,
        )
        metrics = action_metrics(
            output.actions,
            device_batch["actions"],  # type: ignore[arg-type]
            device_batch.get("action_mask"),  # type: ignore[arg-type]
        )
        for name, tensor in metrics.items():
            if tensor.numel():
                values[name].append(tensor.cpu())

    return {
        name: float(torch.cat(tensors).mean()) if tensors else float("nan")
        for name, tensors in values.items()
    }
