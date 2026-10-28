# Paper-to-implementation traceability

Source: *StateLoop: Persistent State for Looped Transformers*, v1, supplied PDF; [paper record](https://arxiv.org/abs/2610.35544). This document records the frozen implementation and interpretation limits. It does not change the registered experiment.

| Paper evidence | Local implementation | Verification or limit |
|---|---|---|
| Sections 3.1–3.2, equations 2–3: normalize the backbone output; maintain a separate state per token across loops | `StateBranch.h_norm`; FP32 state initialized to zero in each `features` invocation | State persists along the loop axis within a forward pass. It is not memory across examples or optimizer steps. |
| Equation 4: direct backbone output plus projected state read forms the next backbone input | `h = h + state.feedback(state.s_norm(s))` before the next shared backbone call | The direct hidden-state path remains present; disabling all reads recovers the paired vanilla model. |
| Equations 5–6: selective exponential retention and complementary write gates | `write`, `delta`, `A_log`; `expm1` computes the complementary write gate stably | The old state remains in the autograd graph. Gate initialization follows the paper: step size 0.3; channel rates log-spaced from 0.01 to 4. |
| Equation 7: identity token-axis mixer | No additional token mixing in the state branch | Token interactions still occur in causal backbone attention. Optional attention/Mamba-2 state mixers are outside this reproduction. |
| Equations 10–12: separate final state projection and autoregressive cross entropy | `state.terminal`; final `head`; next-token loss | Feedback and terminal projection weights are separate. Read normalization is shared. Evaluation masks padding and weights NLL by prediction tokens. |
| Section 5.2, equation 14 and Table 4: reset versus persistent state with matched parameter counts | `reset` removes only the old-state contribution; the branch and initialization are otherwise identical | This is the primary comparison. Vanilla has fewer parameters and is an auxiliary comparison. |
| Section 5.3: mask projected reads | `feedback_off`, `terminal_off`, `all_off` | These are group interventions. They do not reproduce individual-read Shapley attribution or the paper's 63.70%/36.30% shares. |
| Section 5.3, equation 15: channel-wise calibrated state reuse | `single_S1..S3`, `frozen_S1..S3`; epsilon 1e-6 | Statistics come from the separate calibration split. Single replacement resumes normal updates; frozen reuse substitutes the saved state at every later boundary. Both continue backbone computation. |

## What the comparison can establish

The paper's persistence ablation reports PPL 9.25 for reset state and 9.04 for persistent state at 410M scale, seven loops and 25B training tokens (Table 4). Our primary contrast tests the same architectural distinction under a substantially smaller, explicitly registered setting. It does not target those absolute perplexities.

Matched parameter counts, initial weights, data order and token budgets make the reset-versus-persistent comparison informative about the effect of the persistent-state design in this setting. An observed gain would not by itself identify whether its cause is storage, gradient propagation, or an interaction with optimization. Read and reuse interventions measure dependence of an already trained model on its state pathway; they do not replace the training comparison and do not uniquely identify stored information.

## Explicit implementation choices and deviations

The frozen architecture has eight shared backbone blocks, width/state width 384, four loops and 512-token sequences. Vanilla has 52,795,776 parameters; reset/persistent each have 53,387,136. Each run receives 99,991,552 prediction tokens. The data are a pinned, sampled two-shard subset of deduplicated Pile with document-based splits, not the paper's complete data/evaluation setting.

Backbone and numerical details chosen locally are documented in `stateloop/model.py` and `configs/protocol.json`: bias-free projections, pre-RMSNorm blocks and final backbone RMSNorm, standard RoPE base 10000, independent embeddings/head, normal weight initialization with standard deviation 0.02, embedding scaling by the square root of width, and FP16 autocast with FP32 state recurrence and loss. The optimizer and schedule are explicit local choices. These are implementation assumptions, not claims that all were specified by the paper.

The registered decision rule requires positive reset-minus-persistent test NLL for both seeds and mean gain at least 0.005 nat/token. This is an exploratory criterion chosen for this reproduction, not a threshold from the paper. Two seeds and approximately two training tokens per parameter limit conclusions about training variability and convergence. The basic learning gate does not certify convergence; document-bootstrap intervals do not quantify seed uncertainty.

## Artifact interpretation

`training.jsonl` records training prediction-token counts. In the original `validation.jsonl` schema, `tokens` denotes the 65,536 evaluated development tokens; reconstruct training progress from `step * batch_sequences * seq_len`. Plotting uses that reconstruction. This logging ambiguity does not change the model updates, evaluation denominators, or frozen budgets.

The execution commit in each run report identifies the version actually used. Subsequent reporting and audit commits must not be substituted for it. Raw data, weights and execution logs remain local; public results contain scalar metrics, configurations, provenance hashes and reviewed figures.
