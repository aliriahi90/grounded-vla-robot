"""Paired offline VLA robustness and trajectory diagnostics (no retraining).

Run from repository root. Uses the existing frozen dataset and fixed seed 7.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ml_project.cross_attention_model import load_cross_attention_checkpoint
from ml_project.data import make_default_datasets
from ml_project.device import move_to_device, seed_everything
from ml_project.model import TinyVLA, TinyVLAConfig

CONDITIONS = ("clean", "no_wrist", "no_front", "dark", "bright")
SPLITS = ("validation", "deployment_week_1", "deployment_week_2")


def load_baseline(path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    config = TinyVLAConfig(**payload["model_config"])
    model = TinyVLA(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model.eval()


def perturb(batch, condition, delta):
    # Make independent copies: never mutate dataset tensors or another condition's inputs.
    images = {k: v.clone() for k, v in batch["images"].items()}
    masks = {k: v.clone() for k, v in batch["image_masks"].items()}
    if condition in ("no_wrist", "no_front"):
        camera = "wrist" if condition == "no_wrist" else "front"
        masks[camera] = torch.zeros_like(masks[camera], dtype=torch.bool)
        images[camera].zero_()
    elif condition in ("dark", "bright"):
        shift = -delta if condition == "dark" else delta
        for camera in images:
            images[camera] = (images[camera] + shift).clamp(0, 1)
    return images, masks


def metadata_value(meta, field, i):
    if not isinstance(meta, dict) or field not in meta:
        return None
    value = meta[field]
    if isinstance(value, torch.Tensor):
        value = value[i].item()
    elif isinstance(value, (list, tuple)):
        value = value[i]
    return value


def run_split(model, loader, split, condition, device, flow_steps, delta):
    rows, dimensions, timesteps = [], [], []
    with torch.no_grad():
        for batch_index, raw in enumerate(loader):
            batch = move_to_device(raw, device)
            images, masks = perturb(batch, condition, delta)
            # Same random sample seed for corresponding batches in every condition.
            output = model(
                images=images,
                image_masks=masks,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                state=batch["state"],
                flow_steps=flow_steps,
                sample_seed=batch_index,
            )
            prediction = output.actions.detach().cpu()
            target = batch["actions"].detach().cpu()
            labels = batch["action_mask"].detach().cpu().bool().reshape(-1)
            error = (prediction - target).square()
            endpoint_delta = (prediction[:, :, :3] - target[:, :, :3]).sum(1)
            pred_smooth = prediction.diff(dim=1).square().mean((1, 2))
            true_smooth = target.diff(dim=1).square().mean((1, 2))
            for i in range(prediction.shape[0]):
                row = {
                    "split": split,
                    "condition": condition,
                    "sample_id": str(raw["sample_id"][i]),
                    "labeled": bool(labels[i]),
                    "front_available": bool(batch["image_masks"]["front"][i].item()),
                    "wrist_available": bool(batch["image_masks"]["wrist"][i].item()),
                    "task": metadata_value(raw.get("metadata"), "task", i),
                    "robot_id": metadata_value(raw.get("metadata"), "robot_id", i),
                    "site_id": metadata_value(raw.get("metadata"), "site_id", i),
                    "intervention": metadata_value(raw.get("metadata"), "intervention", i),
                    "pred_smoothness": float(pred_smooth[i]),
                    "pred_dim6_mean": float(prediction[i, :, 6].mean()),
                }
                if labels[i]:
                    row.update(
                        mse=float(error[i].mean()),
                        endpoint_error=float(torch.linalg.vector_norm(endpoint_delta[i])),
                        endpoint_bias_x=float(endpoint_delta[i, 0]),
                        endpoint_bias_y=float(endpoint_delta[i, 1]),
                        endpoint_bias_z=float(endpoint_delta[i, 2]),
                        target_smoothness=float(true_smooth[i]),
                        smoothness_gap=float(pred_smooth[i] - true_smooth[i]),
                        target_dim6_mean=float(target[i, :, 6].mean()),
                        dim6_mse=float(error[i, :, 6].mean()),
                    )
                    for d in range(error.shape[2]):
                        dimensions.append(
                            {
                                "split": split,
                                "condition": condition,
                                "sample_id": row["sample_id"],
                                "dimension": d,
                                "mse": float(error[i, :, d].mean()),
                            }
                        )
                    for t in range(error.shape[1]):
                        timesteps.append(
                            {
                                "split": split,
                                "condition": condition,
                                "sample_id": row["sample_id"],
                                "timestep": t,
                                "mse": float(error[i, t].mean()),
                            }
                        )
                    if split == "deployment_week_2" and condition == "clean":
                        corrected = prediction[i].clone()
                        corrected[:, 6] *= -1
                        row["offline_flip_dim6_mse"] = float(
                            (corrected - target[i]).square().mean()
                        )
                rows.append(row)
    return rows, dimensions, timesteps


def summarize(frame):
    output = []
    for (split, condition), group in frame.groupby(["split", "condition"], sort=False):
        labeled = group[group.labeled]
        row = {
            "split": split,
            "condition": condition,
            "n_total": len(group),
            "n_labeled": len(labeled),
            "pred_smoothness_all": group.pred_smoothness.mean(),
        }
        for metric in (
            "mse",
            "endpoint_error",
            "pred_smoothness",
            "target_smoothness",
            "smoothness_gap",
            "endpoint_bias_x",
            "endpoint_bias_y",
            "endpoint_bias_z",
            "dim6_mse",
            "offline_flip_dim6_mse",
        ):
            row[metric] = labeled[metric].mean() if metric in labeled else np.nan
        output.append(row)
    return pd.DataFrame(output)


def benchmark(model, loader, device, flow_steps, repeats):
    raw = next(iter(loader))
    batch = move_to_device(raw, device)
    kwargs = dict(
        images=batch["images"],
        image_masks=batch["image_masks"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        state=batch["state"],
        flow_steps=flow_steps,
        sample_seed=0,
    )
    with torch.inference_mode():
        for _ in range(3):
            model(**kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(**kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000)
    return {
        "batch_size": int(batch["input_ids"].shape[0]),
        "median_batch_ms": float(np.median(samples)),
        "mean_batch_ms": float(np.mean(samples)),
        "repeats": repeats,
        "note": "Batched forward-pass wall time; not single-robot closed-loop latency.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-checkpoint", default="results/cross_full/model.pt")
    parser.add_argument("--baseline-checkpoint", default="checkpoints/baseline.pt")
    parser.add_argument("--output-dir", default="results/robustness_analysis")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--flow-steps", type=int, default=6)
    parser.add_argument("--brightness-delta", type=float, default=0.12)
    parser.add_argument("--latency-repeats", type=int, default=15)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.brightness_delta <= 1:
        parser.error("--brightness-delta must be between 0 and 1")
    seed_everything(7)  # Assessment-required fixed seed.
    device = torch.device(args.device)
    datasets = make_default_datasets(train_size=512, validation_size=128, deployment_size=128)
    loaders = {
        s: DataLoader(datasets[s], batch_size=args.batch_size, shuffle=False) for s in SPLITS
    }
    models = {"cross_full": load_cross_attention_checkpoint(args.cross_checkpoint, device=device)}
    if not args.skip_baseline:
        models["baseline"] = load_baseline(args.baseline_checkpoint, device)
    all_rows, all_dims, all_steps, latencies = [], [], [], {}
    for name, model in models.items():
        model.eval()
        print("Evaluating", name, flush=True)
        latencies[name] = benchmark(
            model, loaders["validation"], device, args.flow_steps, args.latency_repeats
        )
        for split in SPLITS:
            for condition in CONDITIONS:
                rows, dims, steps = run_split(
                    model,
                    loaders[split],
                    split,
                    condition,
                    device,
                    args.flow_steps,
                    args.brightness_delta,
                )
                for collection in (rows, dims, steps):
                    for row in collection:
                        row["model"] = name
                all_rows.extend(rows)
                all_dims.extend(dims)
                all_steps.extend(steps)
                print(
                    f"  {split} {condition}: {sum(r['labeled'] for r in rows)}/{len(rows)} labeled",
                    flush=True,
                )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples = pd.DataFrame(all_rows)
    samples.to_csv(output / "per_sample.csv", index=False)
    pd.DataFrame(all_dims).to_csv(output / "per_dimension.csv", index=False)
    pd.DataFrame(all_steps).to_csv(output / "per_timestep.csv", index=False)
    summary = pd.concat(
        [summarize(group).assign(model=name) for name, group in samples.groupby("model")],
        ignore_index=True,
    )
    summary.to_csv(output / "summary.csv", index=False)
    with (output / "latency.json").open("w") as f:
        json.dump(
            {
                "seed": 7,
                "flow_steps": args.flow_steps,
                "brightness_delta": args.brightness_delta,
                "models": latencies,
            },
            f,
            indent=2,
        )
    for field in ("task", "wrist_available", "intervention"):
        grouped = (
            samples[samples.labeled & (samples.condition == "clean")]
            .groupby(["model", "split", field], dropna=False)
            .agg(n=("mse", "size"), mse=("mse", "mean"), endpoint_error=("endpoint_error", "mean"))
            .reset_index()
        )
        grouped.to_csv(output / f"group_{field}.csv", index=False)
    print(
        "\n"
        + summary[
            [
                "model",
                "split",
                "condition",
                "n_labeled",
                "mse",
                "endpoint_error",
                "pred_smoothness",
                "target_smoothness",
            ]
        ].to_string(index=False)
    )
    print("\nSaved results to", output.resolve())


if __name__ == "__main__":
    main()
