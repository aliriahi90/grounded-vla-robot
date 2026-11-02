from __future__ import annotations

import dataclasses
import time
from collections.abc import Iterable, Mapping

import torch
from torch import nn

from ml_project.device import move_to_device
from ml_project.model import TinyVLA


@dataclasses.dataclass
class TrainHistory:
    losses: list[float]
    elapsed_seconds: float


def train_steps(
    model: TinyVLA,
    batches: Iterable[Mapping[str, object]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    steps: int,
    max_grad_norm: float = 1.0,
) -> TrainHistory:
    """Train for a fixed number of steps, cycling a finite loader as needed."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    model.train()
    iterator = iter(batches)
    losses: list[float] = []
    started = time.perf_counter()

    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(batches)
            batch = next(iterator)
        device_batch = move_to_device(batch, device)
        if not isinstance(device_batch, Mapping):
            raise TypeError("Expected a mapping batch")
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(device_batch)["total"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return TrainHistory(losses=losses, elapsed_seconds=time.perf_counter() - started)
