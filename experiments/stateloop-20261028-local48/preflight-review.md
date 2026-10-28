# Preflight review

The independent FP32 tests cover paired initialization, K=1 persistent/reset equivalence, zero-read reduction to vanilla, causal masking, cross-loop gradients, state gate initialization, save/restore of the next optimizer update, masked NLL, data order, gradient accumulation, document aggregation and bootstrap arithmetic. **10 tests passed** on 2026-10-28.

GPU benchmark and tiny-fit execution commit: `858b99e831e8d5c6b669bc89ab473da930fd5f4f`.

Preferred backbone: 52,795,776 parameters. Reset and persistent models: 53,387,136 parameters each. Short synthetic benchmark (microbatch 8, sequence 512): 30,894--31,688 original tokens/s; measured peak allocated memory 5.999--6.295 GB. This is a short benchmark, not a guarantee of sustained real-data throughput.

Tiny-fit final NLL: vanilla 0.06814 (27 steps), reset 0.07980 (20), persistent 0.08079 (18). All are below the preregistered 0.1 gate. These are correctness checks on four fixed synthetic sequences, not language-model quality results. FP16-vs-FP32 initial maximum logit errors were below 0.0018.

Decision: accept the model preflight and retain the preferred architecture. Freeze formal runs at 99,991,552 prediction tokens each (6,103 optimizer steps, effective batch 32x512), subject to data integrity and a real-data training-path smoke check. The token count is the largest full optimizer-step multiple not exceeding the predeclared 100M cap. Six runs require approximately 5.4 hours of pure training at the slowest short benchmark rate, leaving substantial room within the 34-hour training limit. All quality conclusions remain pending.

Data provenance, partition integrity and full training/evaluation smoke verification are still pending. Formal execution must not begin until these gates pass and `configs/frozen.json` is committed with `ready: true`.

Operational incidents: first data preparation at commit c742638 failed inside the sandbox due to DNS restrictions before obtaining data; the authorized network-enabled retry used the same data preparation implementation. A slow transfer was stopped and resumed with a 7200-second timeout and byte-range resume at commit 0ec2716. A transfer retry reported an SSL interruption. These are preparation incidents, not failed scientific hypotheses. The console progress log before the first resume was overwritten; the original DNS traceback and subsequent transfer log remain local. No training results have been omitted.
