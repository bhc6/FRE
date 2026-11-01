# Endpoint routing CPU review

All 68 checks passed. CPU threads=2, interop threads=1; no GPU or model loading.

- Independently enumerated None/bool/additive edge tables confirm suffix-only deletion, retained endpoint keys, unchanged text keys and unchanged prefix queries.
- BF16 finite-minimum masks are supported; noncausal masks, NaN/+inf and malformed positions are rejected.
- Actual CPU SDPA agrees with a dense reference; a GQA native causal call agrees with explicit-mask sham within float64 tolerance.
- Mocked 28-layer lifecycle verifies Tensor-valued decoder output hooks, exact layer visits, untouched vision dispatch and restoration on success/exception.
- The existing 32-input hash equality check is retained; real-input equality and visual-state hashes still require the pending model run.

- Selectivity requires causal_sham, first_gate and last_gate each to pass all eight static controls; failures in non-target static gates also block the claim.

The runner now records and checks the structural invariants promised by the protocol. Only the final case receives one extra unhooked restoration forward (161 total).

These checks do not certify CUDA numerical equality, resource completion, static selectivity or endpoint recovery. Those remain runtime gates.
