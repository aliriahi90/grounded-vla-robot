"""Train-only statistical baselines for the frozen VLA v1 dataset.

Run from repo root: uv run python scripts/evaluate_simple_baselines.py
Does not modify dataset, train neural networks, or use deployment labels for fitting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "validation", "deployment_week_1", "deployment_week_2")


def extract(ds):
    states, actions, tasks, ids, labeled = [], [], [], [], []
    for i in range(len(ds)):
        x = ds[i]
        states.append(x["state"].numpy())
        actions.append(x["actions"].numpy())
        tasks.append(str(x["metadata"]["task"]))
        ids.append(str(x["sample_id"]))
        labeled.append(bool(x["action_mask"]))
    return dict(
        state=np.stack(states),
        actions=np.stack(actions),
        task=np.asarray(tasks),
        sample_id=np.asarray(ids),
        mask=np.asarray(labeled),
    )


def fit_models(train, ridge_alpha=10.0):
    m = train["mask"]
    if not m.any():
        raise ValueError("Training split has no labeled actions")
    y = train["actions"][m]
    x = train["state"][m].astype(np.float64)
    global_mean = y.mean(axis=0)
    task_means = {
        task: y[train["task"][m] == task].mean(axis=0) for task in np.unique(train["task"][m])
    }
    mu = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    z = (x - mu) / scale
    y_flat = y.reshape(len(y), -1).astype(np.float64)
    # Center X and Y so the intercept is never regularized.
    z_mean = z.mean(axis=0)
    y_mean = y_flat.mean(axis=0)
    zc = z - z_mean
    yc = y_flat - y_mean
    weights = np.linalg.solve(zc.T @ zc + ridge_alpha * np.eye(z.shape[1]), zc.T @ yc)
    return dict(
        global_mean=global_mean,
        task_means=task_means,
        state_mu=mu,
        state_scale=scale,
        z_mean=z_mean,
        y_mean=y_mean,
        weights=weights,
    )


def predict(fitted, data, model):
    n = len(data["state"])
    if model == "global_mean":
        return np.broadcast_to(fitted["global_mean"], (n, *fitted["global_mean"].shape)).copy()
    if model == "task_mean":
        return np.stack([fitted["task_means"].get(t, fitted["global_mean"]) for t in data["task"]])
    if model == "state_ridge":
        z = (data["state"].astype(np.float64) - fitted["state_mu"]) / fitted["state_scale"]
        flat = (z - fitted["z_mean"]) @ fitted["weights"] + fitted["y_mean"]
        return flat.reshape(n, *fitted["global_mean"].shape).astype(np.float32)
    raise ValueError(model)


def summarize(pred, data, split, model):
    m = data["mask"]
    if not m.any():
        return None, [], []
    p, a = pred[m], data["actions"][m]
    err = (p - a) ** 2
    end = np.linalg.norm((p[:, :, :3] - a[:, :, :3]).sum(axis=1), axis=1)
    diff = np.diff(p, axis=1)
    tdiff = np.diff(a, axis=1)
    row = dict(
        model=model,
        split=split,
        n_labeled=int(m.sum()),
        mse=float(err.mean()),
        mse_motion_0_5=float(err[:, :, :6].mean()),
        mse_dim6=float(err[:, :, 6].mean()),
        endpoint_error=float(end.mean()),
        pred_smoothness=float((diff**2).mean()),
        target_smoothness=float((tdiff**2).mean()),
        temporal_difference_mse=float(((diff - tdiff) ** 2).mean()),
        task_coverage=float(np.mean(np.isin(data["task"][m], list(data.get("known_tasks", [])))))
        if "known_tasks" in data
        else np.nan,
    )
    dims = [
        dict(
            model=model,
            split=split,
            dimension=d,
            action_mse=float(err[:, :, d].mean()),
            pred_smoothness=float((diff[:, :, d] ** 2).mean()),
            target_smoothness=float((tdiff[:, :, d] ** 2).mean()),
        )
        for d in range(a.shape[2])
    ]
    steps = [
        dict(model=model, split=split, timestep=t, mse=float(err[:, t].mean()))
        for t in range(a.shape[1])
    ]
    return row, dims, steps


def main():
    from ml_project.data import make_default_datasets

    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results/simple_baselines")
    ap.add_argument(
        "--ridge-alpha",
        type=float,
        default=10.0,
        help="Fixed ridge penalty, not tuned using deployment labels",
    )
    args = ap.parse_args()
    if args.ridge_alpha < 0:
        ap.error("--ridge-alpha must be nonnegative")
    datasets = make_default_datasets(train_size=512, validation_size=128, deployment_size=128)
    data = {name: extract(datasets[name]) for name in SPLITS}
    fitted = fit_models(data["train"], args.ridge_alpha)
    known = set(fitted["task_means"])
    rows, dim_rows, step_rows = [], [], []
    for split in SPLITS:
        ds = data[split]
        ds["known_tasks"] = known
        for model in ("global_mean", "task_mean", "state_ridge"):
            pred = predict(fitted, ds, model)
            row, dims, steps = summarize(pred, ds, split, model)
            if row is not None:
                rows.append(row)
                dim_rows.extend(dims)
                step_rows.extend(steps)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(dim_rows).to_csv(out / "per_dimension.csv", index=False)
    pd.DataFrame(step_rows).to_csv(out / "per_timestep.csv", index=False)
    pd.DataFrame(
        [
            {
                "task": t,
                "train_labeled_count": int(
                    ((data["train"]["task"] == t) & data["train"]["mask"]).sum()
                ),
            }
            for t in sorted(known)
        ]
    ).to_csv(out / "task_counts.csv", index=False)
    print(
        pd.DataFrame(rows)
        .query("split != 'train'")[
            [
                "model",
                "split",
                "n_labeled",
                "mse",
                "mse_motion_0_5",
                "mse_dim6",
                "endpoint_error",
                "task_coverage",
            ]
        ]
        .to_string(index=False)
    )
    print("\nSaved to", out.resolve())


if __name__ == "__main__":
    main()
