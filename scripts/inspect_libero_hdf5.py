"""Inspect an HDF5 demonstration file without assuming a LIBERO schema.

Usage: python scripts/inspect_libero_hdf5.py /path/to/demo.hdf5
Install: python -m pip install h5py
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--max-items", type=int, default=120)
    args = parser.parse_args()
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit("Install optional dependency: pip install -e '.[libero-data]'") from exc
    if not args.file.is_file():
        parser.error(f"File not found: {args.file}")
    with h5py.File(args.file, "r") as f:
        print("Root attributes:", {k: str(v)[:250] for k, v in f.attrs.items()})
        count = 0
        def visit(name, obj):
            nonlocal count
            if count >= args.max_items:
                return
            if isinstance(obj, h5py.Dataset):
                print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
            else:
                print(f"{name}/ [group]")
            count += 1
        f.visititems(visit)
        if count >= args.max_items:
            print("Output truncated; increase --max-items to inspect more.")


if __name__ == "__main__":
    main()
