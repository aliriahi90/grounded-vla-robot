from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance


def dataset_frame(dataset: Iterable[Mapping[str, object]], max_samples: int = 512) -> pd.DataFrame:
    """Create a flat monitoring table without storing raw pixels."""

    rows: list[dict[str, object]] = []
    for index, sample in enumerate(dataset):
        if index >= max_samples:
            break
        images = sample["images"]
        image_masks = sample["image_masks"]
        metadata = sample["metadata"]
        if not isinstance(images, Mapping) or not isinstance(image_masks, Mapping):
            raise TypeError("Expected image and image-mask mappings")
        if not isinstance(metadata, Mapping):
            raise TypeError("Expected metadata mapping")
        state = sample["state"]
        actions = sample["actions"]
        if not isinstance(state, torch.Tensor) or not isinstance(actions, torch.Tensor):
            raise TypeError("Expected tensor state and actions")
        row = dict(metadata)
        row.update(
            {
                "sample_id": sample["sample_id"],
                "prompt_length": len(str(sample["prompt"]).split()),
                "state_norm": float(torch.linalg.vector_norm(state)),
                "state_x": float(state[0]),
                "state_y": float(state[1]),
                "action_abs_mean": (
                    float(actions.abs().mean()) if bool(sample["action_mask"]) else float("nan")
                ),
            }
        )
        for camera_name, image in images.items():
            if not isinstance(image, torch.Tensor):
                raise TypeError("Expected tensor images")
            row[f"{camera_name}_available"] = bool(image_masks[camera_name])
            row[f"{camera_name}_mean"] = float(image.mean())
            row[f"{camera_name}_std"] = float(image.std())
        rows.append(row)
    return pd.DataFrame(rows)


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    reference = _finite(reference)
    current = _finite(current)
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    reference_counts = np.histogram(reference, bins=edges)[0].astype(np.float64)
    current_counts = np.histogram(current, bins=edges)[0].astype(np.float64)
    reference_rate = np.clip(reference_counts / reference_counts.sum(), 1e-6, None)
    current_rate = np.clip(current_counts / current_counts.sum(), 1e-6, None)
    return float(np.sum((current_rate - reference_rate) * np.log(current_rate / reference_rate)))


def continuous_shift(reference: np.ndarray, current: np.ndarray) -> dict[str, float]:
    reference = _finite(reference)
    current = _finite(current)
    if len(reference) == 0 or len(current) == 0:
        return {"ks_statistic": float("nan"), "wasserstein": float("nan"), "psi": float("nan")}
    return {
        "ks_statistic": float(ks_2samp(reference, current).statistic),
        "wasserstein": float(wasserstein_distance(reference, current)),
        "psi": population_stability_index(reference, current),
    }


def categorical_shift(reference: Iterable[object], current: Iterable[object]) -> dict[str, float]:
    reference_series = pd.Series(list(reference), dtype="string").fillna("<missing>")
    current_series = pd.Series(list(current), dtype="string").fillna("<missing>")
    categories = sorted(set(reference_series) | set(current_series))
    reference_rate = reference_series.value_counts(normalize=True).reindex(
        categories, fill_value=0.0
    )
    current_rate = current_series.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    reference_values = reference_rate.to_numpy(dtype=np.float64)
    current_values = current_rate.to_numpy(dtype=np.float64)
    total_variation = 0.5 * np.abs(reference_values - current_values).sum()
    js_distance = jensenshannon(reference_values, current_values, base=2.0)
    return {
        "total_variation": float(total_variation),
        "jensen_shannon_divergence": float(js_distance**2),
    }


def compare_frames(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    continuous_columns: Iterable[str],
    categorical_columns: Iterable[str],
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        "continuous": {
            column: continuous_shift(reference[column].to_numpy(), current[column].to_numpy())
            for column in continuous_columns
        },
        "categorical": {
            column: categorical_shift(reference[column], current[column])
            for column in categorical_columns
        },
    }


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]
