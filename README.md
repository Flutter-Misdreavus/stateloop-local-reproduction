# StateLoop local mechanism reproduction

Independent, scaled reproduction of [StateLoop: Persistent State for Looped Transformers](https://arxiv.org/abs/2610.35544), v1. This is not the authors' implementation.

## Status

Preparation started 2026-10-28 11:38:32 UTC. Absolute deadline: **2026-10-30 11:38:32 UTC**. Training has not started. Results are not yet available.

## Registered comparison

Vanilla Looped, reset-state, and persistent StateLoop, each with seeds 17 and 42. The primary comparison is held-out token NLL(reset) minus NLL(persistent). All runs use matched original-token budgets, paired backbone initialization and data order. Reset and persistent variants have identical parameter counts and initial weights. Both seed differences must be positive and their mean at least 0.005 nat/token for preliminary support; two seeds do not establish strong statistical significance.

Read `configs/protocol.json` for the 48-hour budget, correctness gates, candidate models, data split and stopping conditions. Model size and token budget will be frozen from throughput measurements before formal training. The preferred model is approximately 53M parameters; a predeclared approximately 29M fallback is permitted only before formal results.

## Scope and limitations

Local V100 32GB, four loops, identity state mixer, 512-token contexts, sampled deduplicated Pile data, training from scratch. This tests the persistent-state mechanism at reduced scale; it does not reproduce the paper's absolute perplexities or 25B/30B-token training. Training sufficiency and seed uncertainty will be reported even if correctness gates pass.

This public repository is explicitly authorized by the user. Only original implementation, reproducible configurations, provenance hashes and small aggregate metrics are published. Data text, weights, environments, credentials and raw execution logs are excluded.

## Version policy

Code and configuration are committed before execution; fixes receive a new commit before rerunning. Each run records its actual execution commit separately from later result commits and current HEAD. Stage results are committed and pushed without rewriting history. Failed runs are retained.
