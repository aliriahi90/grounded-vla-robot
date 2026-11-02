"""Train deterministic controls and evaluate on frozen splits without deployment fitting.

Examples:
 uv run python scripts/run_regression_comparison.py
    --model both --steps 200 --device cpu
 uv run python scripts/run_regression_comparison.py
    --model both --steps 1000 --output-dir results/regression_1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml_project.augmentation import AugmentConfig, AugmentedBatches
from ml_project.checkpoint import save_model_checkpoint
from ml_project.config import load_yaml_config
from ml_project.data import make_default_datasets
from ml_project.device import get_device, seed_everything
from ml_project.evaluate import evaluate_model
from ml_project.model import TinyVLAConfig
from ml_project.regression_models import CrossAttentionRegressionVLA, StateOnlyMLP
from ml_project.train import train_steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["both", "regression_vla", "state_mlp"], default="both")
    p.add_argument("--config", default="configs/cross_attention.yaml")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--mlp-width", type=int, default=64)
    p.add_argument("--no-augmentation", action="store_true")
    p.add_argument("--output-dir", default="results/regression_comparison")
    args = p.parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        p.error("steps and batch-size must be positive")
    cfg = load_yaml_config(args.config)
    d, m, t = cfg["data"], cfg["model"], cfg["training"]
    ds = make_default_datasets(
        image_size=int(d["image_size"]),
        max_text_length=int(d["max_text_length"]),
        train_size=int(d["train_size"]),
        validation_size=int(d["validation_size"]),
        deployment_size=int(d["validation_size"]),
    )
    config = TinyVLAConfig(
        vocab_size=len(ds["train"].vocabulary),
        image_size=int(d["image_size"]),
        max_text_length=int(d["max_text_length"]),
        hidden_dim=int(m["hidden_dim"]),
        patch_size=int(m["patch_size"]),
        num_heads=int(m["num_heads"]),
        num_fusion_layers=int(m["num_fusion_layers"]),
        num_action_layers=int(m["num_action_layers"]),
        action_horizon=int(m["action_horizon"]),
        action_dim=int(m["action_dim"]),
    )
    device = get_device(args.device)
    names = ["regression_vla", "state_mlp"] if args.model == "both" else [args.model]
    for name in names:
        seed_everything(args.seed)
        loaders = {
            k: DataLoader(v, batch_size=args.batch_size, shuffle=k == "train")
            for k, v in ds.items()
        }
        model = (
            CrossAttentionRegressionVLA(config)
            if name == "regression_vla"
            else StateOnlyMLP(config, width=args.mlp_width)
        ).to(device)
        # Augmentation only changes observations; state-only MLP intentionally ignores it.
        batches = loaders["train"]
        if name == "regression_vla" and not args.no_augmentation:
            batches = AugmentedBatches(
                batches,
                AugmentConfig(camera_dropout=0.15, brightness=0.12, contrast=0.12),
                seed=args.seed + 1000,
            )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        history = train_steps(model, batches, optimizer, device=device, steps=args.steps)
        metrics = {
            "architecture": name,
            "seed": args.seed,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "parameters": model.parameter_count(),
            "elapsed_seconds": history.elapsed_seconds,
            "training_seconds": history.elapsed_seconds,
            "final_train_loss": history.losses[-1],
            "augmentation": name == "regression_vla" and not args.no_augmentation,
        }
        for split in ("validation", "deployment_week_1", "deployment_week_2"):
            metrics[split] = evaluate_model(
                model, loaders[split], device=device, flow_steps=int(t["flow_steps"])
            )
        out = Path(args.output_dir) / name
        out.mkdir(parents=True, exist_ok=True)
        save_model_checkpoint(
            out / "model.pt",
            model,
            metadata={
                "architecture": name,
                "seed": args.seed,
                "steps": args.steps,
                "mlp_width": args.mlp_width,
            },
        )
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        print("Saved:", out)


if __name__ == "__main__":
    main()
