# StateLoop local mechanism reproduction

Independent, scaled reproduction of [StateLoop: Persistent State for Looped Transformers](https://arxiv.org/abs/2610.35544), v1. This is not the authors' implementation.

## Status

Preparation started 2026-10-28 11:38:32 UTC. Absolute deadline: **2026-10-30 11:38:32 UTC**.

**Completed and reviewed within 48 hours.** All six runs completed 99,991,552 prediction tokens each. Held-out evaluation and 18 state interventions finished about 7.1 hours after preparation began. There were no formal training failures or scientific retries. See the [final scientific review (中文)](experiments/stateloop-20261028-local48/review.md), [independent numerical audit](experiments/stateloop-20261028-local48/completion-audit.json), and [resource report](experiments/stateloop-20261028-local48/resources.json).

| Model | Seed 17 test PPL | Seed 42 test PPL |
|---|---:|---:|
| Vanilla Looped | 61.2985 | 62.1433 |
| Reset state | 58.8136 | 59.2412 |
| Persistent StateLoop | **58.0440** | **58.2648** |

The matched persistence effect was +0.013172 and +0.016620 nat/token, mean **+0.014896**. Both seeds satisfy the registered exploratory rule. This is preliminary support in the scaled setting, not a full-scale replication or a strong claim about training-seed uncertainty.

![Primary comparison](experiments/stateloop-20261028-local48/figures/primary-comparison.png)

Structural tests, FP16/tiny-fit checks, real-data smoke and the [input audit](experiments/stateloop-20261028-local48/input-audit.json) passed. [matrix.json](experiments/stateloop-20261028-local48/matrix.json) records the final reviewed status. Original automatic measurements are preserved in `assessment.json`; its historical pending-review field is superseded by `review.md` and `review.json`. Stage results and progress snapshots were committed and pushed during execution. Weights and raw data remain local.

## Registered comparison

Vanilla Looped, reset-state, and persistent StateLoop, each with seeds 17 and 42. The primary comparison is held-out token NLL(reset) minus NLL(persistent). All runs use matched original-token budgets, paired backbone initialization and data order. Reset and persistent variants have identical parameter counts and initial weights. Both seed differences must be positive and their mean at least 0.005 nat/token for preliminary support; two seeds do not establish strong statistical significance.

Read `configs/protocol.json` for the 48-hour budget, correctness gates, candidate models, data split and stopping conditions. Throughput and smoke measurements selected the approximately 53M preferred model before formal training. `configs/frozen.json` records the fixed settings; the smaller predeclared fallback was not used. [Method traceability](experiments/stateloop-20261028-local48/method-traceability.md) maps the implementation to paper equations and distinguishes local choices from paper specifications.

## Original execution workflow and local re-audit

Use Python 3.10 with the pinned requirements and a PyTorch 2.2.2 CUDA 12.1 build that supports V100. The original local run reuses an existing isolated environment; it does not modify global packages.

```bash
python -m pytest -q tests
python scripts/prepare.py
python -m stateloop.preflight benchmark --candidate preferred --output experiments/stateloop-20261028-local48/benchmark-preferred.json
python -m stateloop.preflight overfit --candidate preferred --output experiments/stateloop-20261028-local48/overfit-preferred.json
# Generate and commit a smoke configuration after data integrity checks:
python scripts/freeze.py --stage smoke
# Commit configs/smoke.json before execution, then:
python -m stateloop.train --variant persistent --seed 123 --run-id stateloop-20261028-smoke --config configs/smoke.json
# After reviewing smoke/report.json, generate and commit the formal configuration:
python scripts/freeze.py --stage formal
# Commit configs/frozen.json and configs/protocol.json before execution, then:
python -m stateloop.supervise
```

The supervisor is configured to publish to the repository owner used by the original run. A different user should run the individual trainer/evaluator commands or explicitly configure their own publication destination. Existing run directories are not silently overwritten. The original absolute deadline is deliberately enforced; reruns require a newly registered protocol with a new deadline.

The commands above document the original workflow. A new run also needs separate run IDs/output paths and an explicitly configured publication destination; cloning this completed repository does not constitute a fresh experiment. On the original machine, with retained local data, checkpoints and per-document evaluation arrays, audit and regenerate the standalone figures with:

```bash
python scripts/review_results.py
python scripts/plot_results.py
```

Published aggregate metrics alone cannot reproduce the document bootstrap. Each audit records its execution HEAD; regenerating reviewed artifacts creates a new revision and requires a new review attestation.

Data preparation downloads pinned text parquet shards, validates their SHA256, assigns normalized-document hashes to disjoint splits, and excludes held-out documents with matching aligned 512-token blocks. This detects exact overlap, not all possible near-duplicate or semantic leakage. Training packs documents separated by EOS; evaluation uses document-bounded windows with masked padding. Raw data and per-document evaluation arrays remain local.

## Scope and limitations

Local V100 32GB, four loops, identity state mixer, 512-token contexts, sampled deduplicated Pile data, training from scratch. This tests the persistent-state mechanism at reduced scale; it does not reproduce the paper's absolute perplexities or 25B/30B-token training. Two seeds and roughly 1.87 tokens per parameter do not establish convergence or general validity. Document-bootstrap intervals condition on the trained models. Group read masks are not Shapley attribution, and disabling reads in a trained model is not a retrained baseline.

This public repository is explicitly authorized by the user. Only original implementation, reproducible configurations, provenance hashes and small aggregate metrics are published. Data text, weights, environments, credentials and raw execution logs are excluded.

## Version policy

Code and configuration are committed before execution; fixes receive a new commit before rerunning. Each run records its actual execution commit separately from later result commits and current HEAD. Stage results are committed and pushed without rewriting history. Failed runs are retained.
