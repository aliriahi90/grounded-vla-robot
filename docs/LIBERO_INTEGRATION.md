# LIBERO integration design (not implemented yet)

LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO

1. Download official demonstration files and use `scripts/inspect_libero_hdf5.py` to enumerate keys, shapes and attributes.
2. Implement `src/ml_project/datasets/libero.py` returning the existing training batch contract: `images` (front/wrist), `image_masks`, `prompt`, `input_ids`, `attention_mask`, `state`, `actions`, `action_mask`, `metadata`.
3. Use official task descriptions and episode boundaries; create action chunks of configurable horizon. Never allow chunks to cross episode boundaries; pad and mask final chunks.
4. Resolve camera-name mapping (e.g. agentview/eye-in-hand), channel order, image scaling, and state composition explicitly. Preserve raw and normalized action representations.
5. Do not reuse the assessment's 19-token vocabulary; fit a training-only vocabulary or use a pretrained tokenizer, consistently across all models.
6. Modify runners to select dataset and to infer `state_dim`, `action_dim`, vocabulary and camera configuration. Validate policy output against LIBERO's action interface.
7. Add rollout evaluator with official success detection and consistent environment resets. Offline HDF5 evaluation alone is not a closed-loop benchmark.

**Caution:** Simulator dependencies and demonstration schemas can vary by LIBERO version. This repository deliberately does not claim a working LIBERO integration yet.
