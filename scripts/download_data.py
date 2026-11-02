from __future__ import annotations

import argparse
from pathlib import Path

from ml_project.download import DATASET_SHA256, DATASET_URL, download_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify the frozen take-home dataset")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--url", default=DATASET_URL)
    parser.add_argument("--sha256", default=DATASET_SHA256)
    parser.add_argument(
        "--local-archive",
        type=Path,
        help="Verify and extract a local archive instead of downloading (maintainer use)",
    )
    args = parser.parse_args()
    destination = download_dataset(
        destination=args.destination,
        url=args.url,
        expected_sha256=args.sha256,
        local_archive=args.local_archive,
    )
    print(f"Dataset ready at {destination}")


if __name__ == "__main__":
    main()
