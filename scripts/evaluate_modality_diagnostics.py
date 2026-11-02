"""Frozen-checkpoint modality shuffling and trajectory diagnostics. Seed fixed at 7.

Run from repository root: uv run python scripts/evaluate_modality_diagnostics.py
No training, no changes to original data. Only labeled examples enter accuracy metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ml_project.cross_attention_model import load_cross_attention_checkpoint
from ml_project.data import make_default_datasets
from ml_project.device import move_to_device, seed_everything
from ml_project.model import TinyVLA, TinyVLAConfig

CONDITIONS = ("clean", "shuffle_images", "shuffle_language", "shuffle_both")
SPLITS = ("validation", "deployment_week_1", "deployment_week_2")


def derangement(n: int, seed: int, salt: int = 0) -> np.ndarray:
    """Reproducible bijection with no fixed points; distinct image/text permutations."""
    if n < 2:
        raise ValueError("Need at least two examples to shuffle")
    rng = np.random.default_rng(seed + salt)
    p = rng.permutation(n)
    # Resolve fixed points by rotating the permuted index order until none remain.
    for offset in range(n):
        candidate = np.roll(p, offset)
        if not np.any(candidate == np.arange(n)):
            return candidate
    # Guaranteed derangement fallback, with randomized ordering.
    order = rng.permutation(n)
    result = np.arange(n)
    result[order] = np.roll(order, 1)
    return result


class ShuffledModalities(Dataset):
    """Keep state/target/sample_id fixed, substitute only specified modalities."""

    def __init__(self, dataset, condition: str, seed: int = 7):
        if condition not in CONDITIONS:
            raise ValueError(condition)
        self.dataset = dataset
        self.condition = condition
        self.image_perm = derangement(len(dataset), seed, 101)
        self.language_perm = derangement(len(dataset), seed, 202)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        if self.condition in ("shuffle_images", "shuffle_both"):
            donor = self.dataset[int(self.image_perm[index])]
            item["images"] = {k: v.clone() for k, v in donor["images"].items()}
            item["image_masks"] = {k: v.clone() for k, v in donor["image_masks"].items()}
        if self.condition in ("shuffle_language", "shuffle_both"):
            donor = self.dataset[int(self.language_perm[index])]
            item["input_ids"] = donor["input_ids"].clone()
            item["attention_mask"] = donor["attention_mask"].clone()
        return item


def load_baseline(path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    model = TinyVLA(TinyVLAConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model.eval()


def evaluate(model, loader, split, condition, device, flow_steps):
    sample_rows, dim_rows, temporal_rows = [], [], []
    with torch.inference_mode():
        for batch_index, raw in enumerate(loader):
            batch = move_to_device(raw, device)
            pred = (
                model(
                    images=batch["images"],
                    image_masks=batch["image_masks"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    state=batch["state"],
                    flow_steps=flow_steps,
                    sample_seed=batch_index,
                )
                .actions.detach()
                .cpu()
            )
            target = batch["actions"].detach().cpu()
            mask = batch["action_mask"].detach().cpu().bool().reshape(-1)
            ids = raw["sample_id"]
            for i in range(len(mask)):
                if not bool(mask[i]):
                    continue
                a, b = pred[i], target[i]
                sq = (a - b).square()
                dp, dt = a.diff(dim=0), b.diff(dim=0)
                delta_err = (dp - dt).square()
                endpoint = (a[:, :3] - b[:, :3]).sum(dim=0)
                corrected = a.clone()
                corrected[:, 6] *= -1
                sample_rows.append(
                    {
                        "split": split,
                        "condition": condition,
                        "sample_id": str(ids[i]),
                        "mse": sq.mean().item(),
                        "endpoint_error": endpoint.norm().item(),
                        "pred_smoothness": dp.square().mean().item(),
                        "target_smoothness": dt.square().mean().item(),
                        "temporal_difference_mse": delta_err.mean().item(),
                        "dim6_mse": sq[:, 6].mean().item(),
                        "pred_dim6_mean": a[:, 6].mean().item(),
                        "target_dim6_mean": b[:, 6].mean().item(),
                        "offline_flip_dim6_mse": (corrected - b).square().mean().item()
                        if split == "deployment_week_2" and condition == "clean"
                        else np.nan,
                    }
                )
                for d in range(a.shape[1]):
                    dim_rows.append(
                        {
                            "split": split,
                            "condition": condition,
                            "sample_id": str(ids[i]),
                            "dimension": d,
                            "action_mse": sq[:, d].mean().item(),
                            "pred_smoothness": dp[:, d].square().mean().item(),
                            "target_smoothness": dt[:, d].square().mean().item(),
                            "temporal_difference_mse": delta_err[:, d].mean().item(),
                        }
                    )
                for t in range(a.shape[0]):
                    temporal_rows.append(
                        {
                            "split": split,
                            "condition": condition,
                            "sample_id": str(ids[i]),
                            "timestep": t,
                            "mse": sq[t].mean().item(),
                        }
                    )
    return sample_rows, dim_rows, temporal_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cross-checkpoint", default="results/cross_full/model.pt")
    p.add_argument("--baseline-checkpoint", default="checkpoints/baseline.pt")
    p.add_argument("--output-dir", default="results/modality_diagnostics")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--flow-steps", type=int, default=6)
    p.add_argument("--skip-baseline", action="store_true")
    args = p.parse_args()
    seed_everything(7)
    device = torch.device(args.device)
    datasets = make_default_datasets(train_size=512, validation_size=128, deployment_size=128)
    models = {"cross_full": load_cross_attention_checkpoint(args.cross_checkpoint, device=device)}
    if not args.skip_baseline:
        models["baseline"] = load_baseline(args.baseline_checkpoint, device)
    samples, dims, steps = [], [], []
    for model_name, model in models.items():
        model.eval()
        for split in SPLITS:
            for condition in CONDITIONS:
                dataset = ShuffledModalities(datasets[split], condition)
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
                r, d, t = evaluate(model, loader, split, condition, device, args.flow_steps)
                for group in (r, d, t):
                    for record in group:
                        record["model"] = model_name
                samples.extend(r)
                dims.extend(d)
                steps.extend(t)
                print(f"{model_name:12s} {split:20s} {condition:17s} n={len(r)}", flush=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sf, df, tf = pd.DataFrame(samples), pd.DataFrame(dims), pd.DataFrame(steps)
    sf.to_csv(output / "per_sample.csv", index=False)
    df.to_csv(output / "per_dimension.csv", index=False)
    tf.to_csv(output / "per_timestep.csv", index=False)
    summary = sf.groupby(["model", "split", "condition"], as_index=False).agg(
        n_labeled=("mse", "size"),
        mse=("mse", "mean"),
        endpoint_error=("endpoint_error", "mean"),
        pred_smoothness=("pred_smoothness", "mean"),
        target_smoothness=("target_smoothness", "mean"),
        temporal_difference_mse=("temporal_difference_mse", "mean"),
        dim6_mse=("dim6_mse", "mean"),
        offline_flip_dim6_mse=("offline_flip_dim6_mse", "mean"),
    )
    summary.to_csv(output / "summary.csv", index=False)
    clean_dims = (
        df[df.condition == "clean"]
        .groupby(["model", "split", "dimension"], as_index=False)[
            ["action_mse", "pred_smoothness", "target_smoothness", "temporal_difference_mse"]
        ]
        .mean()
    )
    clean_dims.to_csv(output / "clean_dimension_summary.csv", index=False)
    clean_time = (
        tf[tf.condition == "clean"]
        .groupby(["model", "split", "timestep"], as_index=False)["mse"]
        .mean()
    )
    clean_time.to_csv(output / "clean_timestep_summary.csv", index=False)
    paired = summary.pivot(index=["model", "split"], columns="condition", values="mse")
    for condition in CONDITIONS[1:]:
        paired[f"{condition}_relative_mse_change_pct"] = 100 * (
            paired[condition] / paired["clean"] - 1
        )
    paired.reset_index().to_csv(output / "modality_effects.csv", index=False)
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "seed": 7,
                "flow_steps": args.flow_steps,
                "conditions": CONDITIONS,
                "splits": SPLITS,
                "batch_size": args.batch_size,
                "note": (
                    "Global deterministic derangements; keep state and action targets "
                    "unchanged; labels only for metrics."
                ),
            },
            indent=2,
        )
    )
    print("\nMODALITY EFFECTS (MSE % vs clean)")
    print(paired[[f"{c}_relative_mse_change_pct" for c in CONDITIONS[1:]]].round(2).to_string())
    print("\nCLEAN PER-DIMENSION RESULTS")
    print(clean_dims.to_string(index=False))
    print("\nSaved to", output.resolve())


if __name__ == "__main__":
    main()
