# Final Report

## Training and validation data

The frozen data contain **512 training** and **128 validation** observation–action records. Each record has a language instruction, front and wrist camera images, a seven-dimensional robot state, and an eight-step, seven-dimensional future action chunk. Records are **independent samples, not episodes or video trajectories**.

Both splits contain all 12 source/destination color tasks; task frequencies vary (32–56 in training, 4–16 in validation). Both cameras are available in the training and validation sets, training labels are complete, and the audit found neither duplicate sample IDs within a split nor overlapping IDs between splits. Distributions of robot-state magnitude, action magnitude, and front-camera intensity broadly overlap. These checks make the splits usable for **controlled offline model comparisons**, but do not establish independence of underlying trajectories or cover real deployment conditions. Uneven task frequencies and the modest validation size can conceal slice-level failures. See `notebooks/01_data_exploration.ipynb`.

A consequential limitation is the strong action prior. Training-only fitted diagnostic predictors achieve validation action MSE **0.004931** (global mean), **0.005009** (task mean), and **0.004089** (state-only ridge). These outperform the generative VLA's aggregate action MSE, so low error need not imply visual grounding or language understanding. I therefore retain these simple controls, inspect per-channel and task-level behavior, and do not use deployment labels for architecture selection.

## Baseline results

The original TinyVLA was trained using the unchanged `configs/baseline.yaml`, with seed **7** and **200 training steps**. The baseline evaluation is:

| Metric | Original TinyVLA |
|---|---:|
| Validation action MSE | **0.032230** |
| Validation endpoint error | **0.923881** |
| Validation action-chunk smoothness | **0.104621** |
| Trainable parameters | **184,391** |
| Training wall time | **378.31 s** |
| Evaluation runtime / throughput | **See measured forward-pass benchmark in Notebook 2 and `results/metrics.json`** |

Action MSE averages squared errors over the 8 × 7 action chunk, but can be dominated by a single channel and is not task-success rate. Endpoint error measures final-step discrepancy according to the repository evaluator; it does not establish physical end-effector success without verified action semantics. Smoothness summarizes adjacent-step action changes; lower is not automatically better when ground-truth actions include necessary transitions. Parameter count describes model size, not memory use or latency. Training wall time depends on hardware and environment. The notebook's forward-pass benchmark measures inference compute under its stated protocol and **excludes** data loading, checkpoint I/O, and other end-to-end deployment costs. See `notebooks/02_vla_architecture.ipynb` and `results/metrics.json` for the locally measured runtime/throughput value.

## Architecture change

**Hypothesis.** The baseline's shared multimodal fusion may not adequately select camera-specific evidence for a language instruction or construct representations specific to each of the eight predicted action steps.

**Implementation.** I added language-guided visual cross-attention and eight learned action queries, alongside FiLM conditioning and training augmentations, under `src/ml_project`. The principal comparison uses the same training split, validation split, seed (7), and 200-step budget as the baseline. Supporting ablations separate action-query-only and language-cross-attention-only settings. The modified model has more parameters, so this is **not** a parameter-matched or perfectly isolated causal test.

| Validation metric | Original flow VLA | Cross-attention + action-query flow VLA |
|---|---:|---:|
| Action MSE | 0.032230 | **0.021902** |
| Endpoint error | 0.923881 | **0.872208** |
| Smoothness | 0.104621 | 0.130312 |
| Trainable parameters | 184,391 | 243,721 |

The combined model improves validation action MSE by approximately **32.0%** relative to the original flow VLA. Action queries alone improved MSE (0.024281); language cross-attention alone worsened it (0.036293). Thus the combined configuration is promising, but the result does **not** prove that language-conditioned image selection caused the improvement. Single-seed comparisons, parameter-count differences, and the weak modality-sensitivity results limit that interpretation. The notebook's file-backed comparison also includes training wall time from the experiment metrics; comparable improved-model inference timing remains an outstanding deployment benchmark.

An additional objective/decoder diagnostic found that a direct-regression VLA reached validation MSE **0.005870** and a state-only MLP reached **0.004711** after 200 steps. Along with the train-fitted statistical controls, this indicates that the generative action decoder is not justified by **aggregate offline MSE alone** on this benchmark. These are supplementary analyses, not replacements for the required controlled architecture comparison.

## Attention and perturbation findings

For **two validation examples**, Notebook 2 visualizes language-to-image attention for **both front and wrist cameras**. The inspected query is **language token position 3** (its token identity is printed for each example), from transformer layer **`fusion.layer_1`**, with attention **averaged over four heads**. The analysis uses the trained baseline checkpoint.

For each camera, the experiment masks the **top 15% of image patches** ranked by attention and compares the action change with **20 random masks** of the same size. The other camera, language, and state are held fixed; sampling seed **7** is shared across clean and perturbed flow predictions. The observed average action changes are close for front-camera top-attention and random masks (approximately **0.0377** versus **0.0376**), while wrist-camera top-attention masking changes actions **less** than random masking in the reported aggregate (approximately **0.0370** versus **0.0419**). There is no consistent evidence that high-attention patches are uniquely causally important.

Attention weights describe internal routing, not a faithful causal explanation. The patch perturbation measures local prediction sensitivity under a masking intervention, not task success or semantic understanding. Separately, image/language/state shuffling changes aggregate validation MSE by less than 0.1% in the reported tests; this raises concern about an action-prior shortcut but does not prove that predictions are invariant or that visual information is unused. See attention overlays and perturbation tables in `notebooks/02_vla_architecture.ipynb`.

## Deployment shift

Each deployment week is compared **independently** with training as the reference. The drift analysis examines continuous image/state features, categorical task/site/robot metadata, camera availability, and missing-input rates. Placeholder images for unavailable cameras are excluded from image brightness/contrast summaries. Performance claims use **labeled records only**: **29** in week 1 and **33** in week 2. Labeled/unlabeled comparisons are included to assess observable selection bias, but cannot establish missing-at-random labels. Methods and measured feature shifts are in `notebooks/03_deployment_shift.ipynb` and `results/drift_report.json`.

**Week 1.** The labeled original flow VLA has action MSE approximately **0.031059**, comparable to its validation MSE of 0.032230. The modified flow VLA has labeled week-1 MSE approximately **0.020923**. Camera and metadata changes warrant monitoring, but these observed labeled metrics do **not** demonstrate aggregate performance degradation. **Response:** continue monitored deployment, collect more representative labels, and report performance by task, robot, camera availability, and action dimension.

**Week 2.** The original flow VLA's labeled action MSE increases to approximately **0.327124**, and the modified flow VLA's to **0.398693**. This is **demonstrated offline prediction degradation on the labeled subset**, not proof of lower closed-loop task success. The largest contribution comes from **action dimension 6**: original baseline dimension-6 MSE is approximately **2.152460**, compared with much smaller errors on dimensions 0–5. The action-dimension-6 **chunk mean** shifts from **+0.375** in training/validation/week 1 (`robot_01`) to **−0.375** in week 2 (`robot_03`). This does not imply the value is constant at every step within a chunk. Robot identity and week are confounded; the data do **not** verify a physical actuator-sign convention change. **Response:** escalate week-2 investigation; verify robot-specific action schema, normalization, and adapters; audit camera availability and metadata; collect representative labeled `robot_03` cases; compare motion-only and dimension-6 errors separately; consider targeted recalibration/retraining or rollback **only after** verifying operational semantics and safety impact. Do not flip command signs based on this statistical pattern alone.

## Limitations and next step

The primary limitations are small and potentially correlated offline splits; independent observation–action records without rollout outcomes; limited deployment labels with possible selection bias; a single controlled training seed; model-size and augmentation confounds; attention maps that are not causal explanations; and robot identity confounded with week 2. Aggregate action MSE also hides channel-specific failures. None of the offline results establish real-robot success, safety, or robust visual grounding.

**Next step:** first validate the week-2 robot/action interface and label semantics, then obtain balanced labeled examples across robots, tasks, and camera-availability conditions. Repeat architecture comparisons across seeds and report per-task, per-channel, and closed-loop success metrics. Before claiming deployment efficiency, benchmark the original and modified models with the same end-to-end evaluation protocol on target hardware.
