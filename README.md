# Robust VLA Research

Research extension of the original TinyVLA assessment, aiming to evaluate multimodal fusion, FiLM conditioning, and action-query decoding on established robot-manipulation benchmarks.

## Status
- **Preserved:** Original models, scripts, notebooks, configurations and `REPORT.md` from the supplied project archive.
- **Added:** Research planning documents, a LIBERO HDF5 inspection utility, dependency manifest and repository hygiene.
- **Not yet implemented:** LIBERO episode dataset adapter, model dimension adaptation, closed-loop LIBERO rollout evaluation, and benchmark training. Do not interpret original synthetic metrics as LIBERO results.

## Quick start
```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .
python scripts/run_baseline.py --help
```

Original experiments require their original dataset (see `scripts/download_data.py`). Avoid committing datasets or checkpoints.

## LIBERO setup (separate environment recommended)
Official project: https://github.com/Lifelong-Robot-Learning/LIBERO

Install LIBERO using its own documented environment instructions. Dataset file formats, observation keys and action conventions should be inspected before connecting them to this project's models. Run:
```bash
python scripts/inspect_libero_hdf5.py /path/to/demo.hdf5
```

Read `docs/RESEARCH_PLAN.md` and `docs/LIBERO_INTEGRATION.md` before implementing the adapter.

## Reproducibility
Keep benchmark train/validation splits disjoint by episode/task as appropriate; record seeds, exact dataset versions, robot action conventions, model parameter counts, and simulation settings. Report closed-loop task success alongside offline metrics.
