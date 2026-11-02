"""Matched FiLM vs baseline experiment. Run from repository root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml_project.checkpoint import save_model_checkpoint
from ml_project.config import load_yaml_config
from ml_project.data import make_default_datasets
from ml_project.device import get_device, seed_everything
from ml_project.evaluate import evaluate_model
from ml_project.film_model import FiLMTinyVLA
from ml_project.model import TinyVLAConfig
from ml_project.train import train_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/film.yaml")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="results/film")
    parser.add_argument("--checkpoint", default="checkpoints/film.pt")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    seed = int(cfg["seed"])
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
    loaders = {
        split: DataLoader(ds, batch_size=int(t["batch_size"]), shuffle=(split == "train"))
        for split, ds in datasets.items()
    }
    model = FiLMTinyVLA(
        TinyVLAConfig(
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
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(t["learning_rate"]))
    steps = args.steps if args.steps is not None else int(t["steps"])
    history = train_steps(model, loaders["train"], optimizer, device=device, steps=steps)
    metrics = {
        "architecture": "film",
        "seed": seed,
        "device": str(device),
        "steps": steps,
        "learning_rate": float(t["learning_rate"]),
        "parameters": model.parameter_count(),
        "elapsed_seconds": history.elapsed_seconds,
        "final_train_loss": history.losses[-1],
        "validation": evaluate_model(
            model, loaders["validation"], device=device, flow_steps=int(t["flow_steps"])
        ),
    }
    checkpoint = save_model_checkpoint(
        args.checkpoint,
        model,
        metadata={
            "architecture": "film",
            "seed": seed,
            "steps": steps,
            "learning_rate": float(t["learning_rate"]),
        },
    )
    metrics["checkpoint"] = str(checkpoint)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
