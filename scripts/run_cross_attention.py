"""Run from repo root: uv run python scripts/run_cross_attention.py --device cpu"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml_project.augmentation import AugmentConfig, AugmentedBatches
from ml_project.checkpoint import save_model_checkpoint
from ml_project.config import load_yaml_config
from ml_project.cross_attention_model import CrossAttentionVLA
from ml_project.data import make_default_datasets
from ml_project.device import get_device, seed_everything
from ml_project.evaluate import evaluate_model
from ml_project.model import TinyVLAConfig
from ml_project.train import train_steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/cross_attention.yaml")
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--no-language-cross-attention", action="store_true")
    p.add_argument("--no-action-queries", action="store_true")
    p.add_argument("--no-augmentation", action="store_true")
    p.add_argument("--output-dir", default="results/cross_attention")
    args = p.parse_args()
    cfg = load_yaml_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    seed_everything(seed)
    device = get_device(args.device)
    d, m, t = cfg["data"], cfg["model"], cfg["training"]
    ds = make_default_datasets(
        image_size=int(d["image_size"]),
        max_text_length=int(d["max_text_length"]),
        train_size=int(d["train_size"]),
        validation_size=int(d["validation_size"]),
        deployment_size=int(d["validation_size"]),
    )
    loaders = {
        k: DataLoader(v, batch_size=int(t["batch_size"]), shuffle=k == "train")
        for k, v in ds.items()
    }
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
    model = CrossAttentionVLA(
        config,
        use_language_cross_attention=not args.no_language_cross_attention,
        use_action_queries=not args.no_action_queries,
    ).to(device)
    train_loader = (
        loaders["train"]
        if args.no_augmentation
        else AugmentedBatches(
            loaders["train"],
            AugmentConfig(camera_dropout=0.15, brightness=0.12, contrast=0.12),
            seed=seed + 1000,
        )
    )
    steps = args.steps if args.steps is not None else int(t["steps"])
    lr = float(t["learning_rate"])
    history = train_steps(
        model,
        train_loader,
        torch.optim.AdamW(model.parameters(), lr=lr),
        device=device,
        steps=steps,
    )
    metrics = {
        "architecture": "cross_attention",
        "seed": seed,
        "steps": steps,
        "parameters": model.parameter_count(),
        "language_cross_attention": model.use_language_cross_attention,
        "action_queries": model.use_action_queries,
        "augmentation": not args.no_augmentation,
        "train_seconds": history.elapsed_seconds,
        "final_train_loss": history.losses[-1],
        "validation": evaluate_model(
            model, loaders["validation"], device=device, flow_steps=int(t["flow_steps"])
        ),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_model_checkpoint(
        output / "model.pt",
        model,
        metadata={
            k: metrics[k]
            for k in (
                "architecture",
                "seed",
                "steps",
                "language_cross_attention",
                "action_queries",
                "augmentation",
            )
        },
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
