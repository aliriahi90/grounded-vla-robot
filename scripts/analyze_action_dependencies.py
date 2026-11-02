"""Dataset-only checks: does state, instruction, or task explain target actions?"""

from pathlib import Path

import pandas as pd

from ml_project.data import make_default_datasets


def main():
    datasets = make_default_datasets(train_size=512, validation_size=128, deployment_size=128)
    rows = []
    for split, dataset in datasets.items():
        for i in range(len(dataset)):
            x = dataset[i]
            if not bool(x["action_mask"]):
                continue
            a = x["actions"].numpy()
            row = {
                "split": split,
                "sample_id": x["sample_id"],
                "prompt": x["prompt"],
                "task": x["metadata"]["task"],
                "robot_id": x["metadata"]["robot_id"],
            }
            row.update({f"state_{j}": float(v) for j, v in enumerate(x["state"].numpy())})
            row.update({f"action_mean_{j}": float(a[:, j].mean()) for j in range(7)})
            row.update({f"action_start_{j}": float(a[0, j]) for j in range(7)})
            row.update({f"action_end_{j}": float(a[-1, j]) for j in range(7)})
            rows.append(row)
    df = pd.DataFrame(rows)
    out = Path("results/state_dependency")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "labeled_action_features.csv", index=False)
    summary = (
        df.groupby(["split", "robot_id"], dropna=False)
        .agg(
            n=("sample_id", "size"),
            dim6_mean=("action_mean_6", "mean"),
            dim6_std=("action_mean_6", "std"),
        )
        .reset_index()
    )
    summary.to_csv(out / "dim6_by_robot.csv", index=False)
    print("ACTION DIMENSION 6 BY ROBOT AND SPLIT\n", summary.to_string(index=False))
    print(
        "\nTRAINING PROMPT/TASK COUNTS\n",
        df[df.split == "train"]
        .groupby(["task", "prompt"])
        .size()
        .sort_values(ascending=False)
        .head(30)
        .to_string(),
    )
    print("\nACTION MEAN STANDARD DEVIATION ACROSS LABELED SAMPLES")
    print(df.groupby("split")[[f"action_mean_{j}" for j in range(7)]].std().to_string())
    print("\nSaved to", out.resolve())


if __name__ == "__main__":
    main()
