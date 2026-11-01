# Controller smoke artifact audit

130 / 130 CPU artifact checks passed. Original artifacts were unchanged.

The recorded 3 conditions × 4 methods are complete. Every row records full-vocabulary zero-controller equality, zero option-logit online/replay error for a nonzero context delta, and finite nonzero controller gradients. The saved tail's tensor hashes and configuration match its manifest; runtime source hashes still match; resource telemetry agrees with its completion summary and frozen bounds.

This smoke is an engineering check on one seen training source. Raw logit/gradient arrays are not saved, so scalar equality evidence cannot be independently rescored. All nonzero controllers receive fixed bias gains; learned content-dependent gains and batch-eight replay need their own checks. Nonzero context deltas may still quantize away in native logits. Static original labels serve only to obtain a gradient, with no optimizer update and no static accuracy claim.
