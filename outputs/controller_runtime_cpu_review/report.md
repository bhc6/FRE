# Controller runtime CPU review

73 / 73 checks passed. No GPU or pretrained model was used.

The expanded GQA reference agrees with independent dense attention and native CPU SDPA for final-token none, boolean, and additive masks. Visual attention mass is not renormalized.

FrozenTail matches an independent BF16 expression, including a rounding-sensitive fixture which detects incorrect residual reassociation. Gradients propagate through BF16 casts and frozen norms/MLP to the residual and all four controllers. Three CPU optimizer steps produce nonzero bounded residuals. Frozen backbone parameters receive no gradients.

A synthetic 28-layer wrapper using native CPU SDPA verifies passive and zero-controller bitwise equality, exact last-token delta projection, unchanged earlier tokens, tuple preservation, and registry/hook restoration after success, injected errors, and rejected repeated dispatch.

This validates structure and differentiability, not pretrained GPU online/replay equality or task quality. The cached query is the actual post-normalization, post-RoPE SDPA input. Cached-tail replay is a numerical training proxy, whose error must be reported and whose learned checkpoints require independent full online evaluation.
