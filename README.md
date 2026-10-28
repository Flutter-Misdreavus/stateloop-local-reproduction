# StateLoop local mechanism reproduction

Independent, scaled reproduction of [StateLoop: Persistent State for Looped Transformers](https://arxiv.org/abs/2610.35544), v1. This is not the authors' implementation.

## Status

Preparation started 2026-10-28 11:38:32 UTC. Absolute deadline: **2026-10-30 11:38:32 UTC**.

The preferred 52.8M/53.4M models passed structural checks, FP16 checks and tiny-data fitting. See [preflight review](experiments/stateloop-20261028-local48/preflight-review.md) and its linked JSON evidence. These checks are not language-model results.

The local managed service is active. [launch.json](experiments/stateloop-20261028-local48/launch.json) tracks data preparation and the real-data smoke gate before the formal matrix begins. The full workflow stops on a failed gate and preserves the failure; it does not silently change settings or retry scientific results.

When formal execution starts, [matrix.json](experiments/stateloop-20261028-local48/matrix.json) records live stage status. At completion, [assessment.json](experiments/stateloop-20261028-local48/assessment.json) and [results.md](experiments/stateloop-20261028-local48/results.md) contain automated measurements. These links are created by the relevant stage; automated completion is not scientific review. Stage results and hourly progress snapshots are pushed to this repository.

## Registered comparison

Vanilla Looped, reset-state, and persistent StateLoop, each with seeds 17 and 42. The primary comparison is held-out token NLL(reset) minus NLL(persistent). All runs use matched original-token budgets, paired backbone initialization and data order. Reset and persistent variants have identical parameter counts and initial weights. Both seed differences must be positive and their mean at least 0.005 nat/token for preliminary support; two seeds do not establish strong statistical significance.

Read `configs/protocol.json` for the 48-hour budget, correctness gates, candidate models, data split and stopping conditions. Model size and token budget will be frozen from throughput measurements before formal training. The preferred model is approximately 53M parameters; a predeclared approximately 29M fallback is permitted only before formal results.

## Reproduction commands

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

Data preparation downloads pinned text parquet shards, validates their SHA256, assigns normalized-document hashes to disjoint splits, and excludes held-out documents with matching aligned 512-token blocks. This detects exact overlap, not all possible near-duplicate or semantic leakage. Training packs documents separated by EOS; evaluation uses document-bounded windows with masked padding. Raw data and per-document evaluation arrays remain local.

## Scope and limitations

Local V100 32GB, four loops, identity state mixer, 512-token contexts, sampled deduplicated Pile data, training from scratch. This tests the persistent-state mechanism at reduced scale; it does not reproduce the paper's absolute perplexities or 25B/30B-token training. Training sufficiency and seed uncertainty will be reported even if correctness gates pass.

This public repository is explicitly authorized by the user. Only original implementation, reproducible configurations, provenance hashes and small aggregate metrics are published. Data text, weights, environments, credentials and raw execution logs are excluded.

## Version policy

Code and configuration are committed before execution; fixes receive a new commit before rerunning. Each run records its actual execution commit separately from later result commits and current HEAD. Stage results are committed and pushed without rewriting history. Failed runs are retained.
