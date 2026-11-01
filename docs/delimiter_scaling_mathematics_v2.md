# Delimiter scaling as residual update damping

Date: 2026-11-01. This is a mathematical mechanism hypothesis and an equivalence control for a published delimiter-scaling baseline. It is not a claim of a new method, improved task accuracy, or a result already measured on the new confirmation sources.

## Exact normalization relation

Let R(x) = gamma * x / sqrt(mean(x^2) + epsilon), with elementwise learned gamma and lambda > 0. Then

R(lambda x) = gamma * x / sqrt(mean(x^2) + epsilon/lambda^2).

With epsilon = 0 and nonzero x, R(lambda x) = R(x). For positive epsilon, their ratio is sqrt((mean(x^2)+epsilon)/(mean(x^2)+epsilon/lambda^2)), not lambda. Multiplying a pre-norm residual token by lambda therefore does not imply multiplying its next Q/K/V by lambda. Finite precision, small-norm tokens and subsequent residual updates need separate treatment. Projection bias does not break the exact equivalence when normalized inputs are already identical. Qwen's additional query/key head normalization further changes any remaining scalar effects.

## Proposition: fixed token scaling and reciprocal updates

Assume a pre-norm decoder whose attention and FFN depend on residual states only through exactly scale-invariant per-token normalization. Fix a set of token positions S and lambda > 0. No dropout, cache changes, token mixing outside these sublayers, or extra residual injections are introduced. Attention may mix tokens arbitrarily after normalization and respects the same causal mask in both models.

Path A multiplies the residual at positions S by lambda immediately after layer ell, then performs every subsequent block normally. Path B leaves the residual unchanged at ell and, at every layer j > ell, multiplies both the attention update and the FFN update at positions S by 1/lambda before residual addition. Updates and states at all other positions remain ordinary.

Claim: after each following sublayer, h_A[t] = lambda h_B[t] for t in S and h_A[t] = h_B[t] otherwise. In particular, normalized final states and final logits are equal when the last normalization is scale-invariant. If the answer position is outside S, its pre-final-normalization residual is equal as well.

Proof by induction. The relation holds immediately after the intervention. Per-token normalization makes the attention inputs identical everywhere, so the two attention computations produce the same update a. For a selected token, lambda h + a = lambda (h + a/lambda); for an unselected token, both paths add the same a to equal states. The relation therefore holds after attention. The FFN receives equal normalized inputs and produces the same update f. Again lambda h + f = lambda (h + f/lambda), completing the sublayer induction. The argument repeats across all remaining layers. The final normalization removes the residual scale.

This concerns a fixed selector S and constant lambda. It does not automatically cover selectors recomputed from modified hidden states, nonuniform time-varying scale decisions, post-norm models, additive cross-layer visual features after ell, or stochastic generation paths with differing inputs. For Qwen3-VL, the chosen middle layer is after the early DeepStack additions; for the pinned SmolVLM2 language backbone the architecture is ordinary pre-norm Llama. The real model still has epsilon and BF16 arithmetic, so full-vocabulary comparison is required.

## Interpretation and falsification

An observed benefit of scaling a delimiter may reflect smaller subsequent updates relative to its accumulated state. It does not by itself establish stronger immediate Q/K/V, greater visual attention, or preserved temporal evidence. The first following attention is predicted to be nearly unchanged; later updates can differ because addition changes token directions before subsequent normalization.

The runtime comparison is native, identity scaling, delimiter scaling at 0.5/2/4, and reciprocal update damping at 0.5 corresponding to scaling by 2. Save next-layer delimiter normalization and Q/K/V statistics, whole-vocabulary differences, predictions and paired task results. Equality between the two nontrivial intervention paths is a mechanism control, not an extra accuracy gain. Unexpected differences should first be traced to epsilon, dtype, extra residual injection, or an implementation mismatch. Powers of two have particularly favorable binary floating-point arithmetic; empirical equality at lambda=2 does not establish equality for arbitrary lambda.

The 18 CPU checks in outputs/delimiter_gauge_cpu_v2 cover the algebraic RMSNorm relation and tiny actual Qwen/Llama text architectures. Finite-epsilon answer-state differences of approximately 1e-4 to 1e-3 remain in those checks. A threshold-based test pass is evidence of the declared numerical tolerance, not exact equivalence for positive epsilon or evidence of checkpoint task performance.

## Connection to frequency analysis

Scaling every time-indexed delimiter state by a common lambda multiplies every Fourier coefficient by lambda and every power by lambda^2. Normalized power ratios are unchanged at that intervention instant. Consequently a frequency-ratio plot alone cannot distinguish direct scaling from temporal selectivity. Frequency-dependent or selective update effects must be measured later and compared with spatial/token and temporal controls. Delimiter indices are frame-associated only after actual processor layout verification; a single video-boundary token pair has no frame-time frequency axis.

No frequency-based adaptive rule is selected in this document. It first separates a possible causal mechanism from a misleading amplitude interpretation and provides a stronger, deployable baseline for later research.
