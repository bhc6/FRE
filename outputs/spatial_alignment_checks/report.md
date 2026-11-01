# Spatial alignment CPU counterexamples

All 10 checks passed. One NumPy process, two library threads, no torch/model/GPU.

- A constant object crossing patches gives fixed-patch non-DC energy 1.23046875.
- Object alignment reduces non-DC energy to 0 without reducing total energy.
- Opposite motion labels become identical aligned features: difference 0.
- Retaining the warp/position sequence preserves the two opposite direction labels in this construction.
- Temporal DFT/time-varying warp commutator norm: 6.2749502; constant-warp control: 0.
- Frequency-domain warp operator off-diagonal norm: 15.4919334.

This is a periodic translation construction with oracle alignment, not evidence that tracking is feasible or memory improves.
Exact criteria are in the source check conditions; metrics and source hash are in checks.json.
