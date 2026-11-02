"""Controlled baseline/FiLM x augmentation ablation. Run from repository root."""

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
from ml_project.film_model import FiLMTinyVLA
from ml_project.model import TinyVLA, TinyVLAConfig
from ml_project.train import train_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["baseline", "film"], required=True)
    parser.add_argument(
        "--augmentation", choices=["none", "dropout", "jitter", "both"], required=True
    )
    parser.add_argument("--config", default="configs/robustness.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--camera-dropout", type=float, default=0.15)
    parser.add_argument("--brightness", type=float, default=0.12)
    parser.add_argument("--contrast", type=float, default=0.12)
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    seed_everything(seed)
    device = get_device(args.device)
    d, m, t = cfg["data"], cfg["model"], cfg["training"]
    datasets = make_default_datasets(
        image_size=int(d["image_size"]),
        max_text_length=int(d["max_text_length"]),
        train_size=int(d["train_size"]),
        validation_size=int(d["validation_size"]),
        deployment_size=int(d["validation_size"]),
    )
    train_loader = DataLoader(datasets["train"], batch_size=int(t["batch_size"]), shuffle=True)
    val_loader = DataLoader(datasets["validation"], batch_size=int(t["batch_size"]), shuffle=False)
    augment = AugmentConfig(
        camera_dropout=args.camera_dropout if args.augmentation in ("dropout", "both") else 0,
        brightness=args.brightness if args.augmentation in ("jitter", "both") else 0,
        contrast=args.contrast if args.augmentation in ("jitter", "both") else 0,
    )
    train_batches = (
        AugmentedBatches(train_loader, augment, seed=seed + 10000)
        if args.augmentation != "none"
        else train_loader
    )
    model_cfg = TinyVLAConfig(
        vocab_size=len(datasets["train"].vocabulary),
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
    model = (FiLMTinyVLA if args.architecture == "film" else TinyVLA)(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(t["learning_rate"]))
    steps = args.steps if args.steps is not None else int(t["steps"])
    history = train_steps(model, train_batches, optimizer, device=device, steps=steps)
    run_name = f"{args.architecture}_{args.augmentation}_seed{seed}"
    output_dir = Path(args.output_dir or f"results/robustness/{run_name}")
    checkpoint = Path(args.checkpoint or f"checkpoints/{run_name}.pt")
    metrics = dict(
        architecture=args.architecture,
        augmentation=args.augmentation,
        augment_config=augment.__dict__,
        seed=seed,
        device=str(device),
        steps=steps,
        learning_rate=float(t["learning_rate"]),
        parameters=model.parameter_count(),
        elapsed_seconds=history.elapsed_seconds,
        final_train_loss=history.losses[-1],
        validation=evaluate_model(
            model, val_loader, device=device, flow_steps=int(t["flow_steps"])
        ),
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    saved = save_model_checkpoint(
        checkpoint,
        model,
        metadata={
            "architecture": args.architecture,
            "augmentation": args.augmentation,
            "seed": seed,
            "steps": steps,
            "augmentation_config": augment.__dict__,
        },
    )
    metrics["checkpoint"] = str(saved)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
