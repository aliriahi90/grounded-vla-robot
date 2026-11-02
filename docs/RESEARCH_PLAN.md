# Research plan

## Primary question
When do language-conditioned fusion and step-specific action queries improve robot manipulation success and robustness compared with a matched TinyVLA baseline?

## Milestones
1. **Preserve assessment baseline:** rerun original models on the original data; save seed/config/metrics.
2. **LIBERO adapter:** parse episodes, two camera views, proprioception, language and actions. Validate gripper convention, camera order, action units, and padding.
3. **Architecture compatibility:** make state dimension, action dimension, text vocabulary, camera names and horizon configurable end-to-end. Existing `VLADataConfig` explicitly restricts original dataset to 7D; do not silently change this behavior.
4. **Closed-loop evaluator:** connect trained policy to LIBERO environment, perform receding-horizon action execution, measure task success.
5. **Ablations:** baseline; FiLM; cross-attention; action queries; combined; state-only and image-only controls.
6. **Robustness:** camera dropout, background/lighting shifts, object layouts, instruction swaps; test both clean and perturbed rollouts.
7. **External validation:** consider DROID and/or ALOHA only after LIBERO results are reliable.

## Evaluation protocol
- Multiple training seeds and multiple episodes per task; uncertainty intervals.
- Primary: closed-loop success; secondary: offline action error, inference latency, memory and parameter count.
- No leakage across episode-level splits; record benchmark suite and evaluation protocol.
- Compare to contemporary published baselines under matched data and compute where possible.
- Treat synthetic results as diagnostics, not evidence of real-world manipulation performance.

## Paper structure
Introduction; Related Work; Problem Formulation; Proposed Method; Experimental Setup; Ablations; Robustness; Limitations; Conclusion.

## Candidate venues
ICRA, IROS, CoRL (check actual deadlines and submission policies before selecting).
