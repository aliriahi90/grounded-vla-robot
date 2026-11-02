import json
from pathlib import Path

import pandas as pd

root = Path.cwd()
if root.name == "notebooks":
    root = root.parent

rows = []

for path in sorted((root / "results").rglob("metrics.json")):
    metrics = json.loads(path.read_text())

    if "validation" not in metrics:
        continue

    rows.append(
        {
            "experiment": str(path.parent.relative_to(root / "results")),
            "parameters": metrics.get("parameters"),
            "mse": metrics["validation"]["mse"],
            "endpoint_error": metrics["validation"]["endpoint_error"],
            "smoothness": metrics["validation"]["smoothness"],
            "training_seconds": metrics.get("elapsed_seconds"),
        }
    )

df = pd.DataFrame(rows).sort_values("mse")
print(df)
