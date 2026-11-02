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
from ml_project.model import TinyVLA, TinyVLAConfig
from ml_project.train import train_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the compact VLA baseline")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--steps", type=int, help="Override the configured training steps")
    parser.add_argument("--device", help="Explicit torch device, for example cpu or cuda")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--checkpoint", default="checkpoints/baseline.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    device = get_device(args.device)

    data_config = config["data"]
    datasets = make_default_datasets(
        image_size=int(data_config["image_size"]),
        max_text_length=int(data_config["max_text_length"]),
        train_size=int(data_config["train_size"]),
        validation_size=int(data_config["validation_size"]),
        deployment_size=int(data_config["validation_size"]),
    )
    training_config = config["training"]
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=int(training_config["batch_size"]),
            shuffle=split == "train",
        )
        for split, dataset in datasets.items()
    }

    model_config = config["model"]
    model = TinyVLA(
        TinyVLAConfig(
            vocab_size=len(datasets["train"].vocabulary),
            image_size=int(data_config["image_size"]),
            max_text_length=int(data_config["max_text_length"]),
            hidden_dim=int(model_config["hidden_dim"]),
            patch_size=int(model_config["patch_size"]),
            num_heads=int(model_config["num_heads"]),
            num_fusion_layers=int(model_config["num_fusion_layers"]),
            num_action_layers=int(model_config["num_action_layers"]),
            action_horizon=int(model_config["action_horizon"]),
            action_dim=int(model_config["action_dim"]),
        )
    ).to(device)
    learning_rate = float(training_config["learning_rate"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    steps = args.steps or int(training_config["steps"])
    history = train_steps(
        model,
        loaders["train"],
        optimizer,
        device=device,
        steps=steps,
    )
    flow_steps = int(training_config["flow_steps"])
    metrics: dict[str, object] = {
        "seed": seed,
        "device": str(device),
        "steps": steps,
        "learning_rate": learning_rate,
        "parameters": model.parameter_count(),
        "elapsed_seconds": history.elapsed_seconds,
        "final_train_loss": history.losses[-1],
    }
    metrics["validation"] = evaluate_model(
        model,
        loaders["validation"],
        device=device,
        flow_steps=flow_steps,
    )

    checkpoint_path = save_model_checkpoint(
        args.checkpoint,
        model,
        metadata={
            "seed": seed,
            "steps": steps,
            "learning_rate": learning_rate,
        },
    )
    metrics["checkpoint"] = str(checkpoint_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote results to {output_dir}")
    print(f"Wrote checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
