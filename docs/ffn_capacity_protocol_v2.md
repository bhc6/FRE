# FFN capacity control following the normalized readout screen

Written 2026-11-01, before this control is trained or scored. This is a generic adapter baseline, not a proposed novel method. The normalized first-stage screen found identity 17/40, normalized time at radius .3 18/40, and the other five candidates 17/40. Its option-rotation run has completed but has not yet been summarized here. New 32-source scores remain unopened.

The training-only label-assisted reachability diagnostic achieved 84 -> 89/142 with normalized time directions, 84 -> 90 with normalized spectral directions, and 84 -> 103 with arbitrary head-context directions under radius .3. These finite-optimization BF16 replay results motivate testing a less restricted learned space; they are not accuracy claims, proof of an optimum, or equal actuator comparisons with an FFN adapter.

At the final decoder layer and only the last prefill text position, let x be the actual post-attention RMSNorm output, y = FFN(x). A rank-4 MLP (2048 -> 4 -> 2048, SiLU, zero-initialized output layer, 18,436 parameters) proposes d_raw = ||y|| tanh(MLP(x)) / sqrt(2048). Project onto ||d|| <= rho ||y|| and replace y by y + d in the original computation. The original y, residual connection, final norm, and language head remain unchanged. No label enters the forward module. It is a general feature-conditioned FFN residual, with no spectral or memory-specific inductive bias.

Fixed initial candidates: rho .1 and .3, seed 20261101, 200 AdamW steps, lr .001, weight decay .0001. Use the same 32 training sources, 142 real questions, 64 static distillation cases, paired batch stream, and loss as normalized_development_protocol_v2.md. Frozen-tail replay is approximate; full-model forwards are authoritative. Save paired native/zero/final replay logits and full training provenance.

The parameter count is close to but not equal to 18,584. The norm cap is relative to FFN output, not attention head context. Therefore neither perturbation strength nor hypothesis class is matched; this is a broader capacity baseline, not an isolated comparison of temporal versus spectral representations. A gain cannot establish that the diagnosed attention mass caused errors.

Evaluate both candidates on the same eight-source validation questions, original order first and all cyclic option orders before any retained gain claim. Show source macro, micro, NLL, repairs/regressions and static prediction changes. Later seeds or unseen-source confirmation require frozen choices. All V1 sources and results remain unchanged. No new-source test inspection or arbitrary rank/radius/step search is allowed.

Use the unchanged ResourceGuard, a single GPU process, maximum 300 seconds, two CPU threads, .65 allocator limit, and original memory/temperature thresholds.
