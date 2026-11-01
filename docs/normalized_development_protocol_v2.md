# Internal readout normalization: bounded V2 development

Date: 2026-11-01. This protocol is written before any V2 training or model scoring. V1 source, checkpoints, and conclusions remain frozen. V1 test is now seen development replication, never an unseen test for V2.

## Observation and hypothesis

On the 142 original real training cases, the median head visual attention mass is 0.001694. The exact maximum norm over the eight-gain box is only 0.159% (time) or 0.185% (spectral) of native head context at the median; the 10% cap cannot activate. This limits the V1 intervention and weakens the visual conditioner. Low final-layer direct visual attention does not imply that earlier layers failed to encode vision.

For basis vectors B_r and native context c, define u_r = B_r / max(||B_r||, 1e-6 ||c||), with exactly zero numerator remaining zero. Conditioner inputs are [q, sqrt(128) u_1, ..., sqrt(128) u_8]. The intervention basis is ||c|| u_r / sqrt(8). Compute gains g = .25 tanh(MLP(features)) and project sum_r g_r ||c|| u_r / sqrt(8) onto the per-head ball of radius rho ||c||. All coordinates, including signed spectral phase coordinates, remain available. The original context and remaining frozen model computation are retained.

This changes both conditioner and actuator scaling. A benefit of this combined change alone cannot distinguish their causal contributions. It is a normalized internal visual readout, not established memory emergence or a novel spectral mechanism. Normalization can amplify noisy directions; the epsilon and norm cap remain mandatory.

## Fixed first-stage candidate set

- Methods: unit_time_mlp, unit_spectral_mlp, unit_time_constant.
- Residual radii: 0.1 and 0.3 of each native head context norm.
- MLP: 1152 -> 16 -> 8, SiLU, 18,584 parameters. Constant: 8 trainable gains.
- Original 32-source train split: 142 real QA and 64 null-label static distillation conditions.
- Fixed seeds: 20261101, 20261102, 20261103. First development screen uses 20261101; further seeds follow if a candidate is useful.
- Fixed 200 steps; final checkpoint only. AdamW lr .001, weight decay .0001; six real and two static cases per batch sampled with replacement using identical paired RNG streams.
- CE + .1 real-teacher KL + 1 static-teacher KL + .001 mean relative delta squared. Gradient norm clip 1.0. Original BF16 frozen-tail replay for training; full native model forwards determine evaluation accuracy.
- Retain training logs, sample IDs, initial/final checkpoints, optimizer state, before/after full-train replay metrics, hashes, and resource telemetry for all candidates.

## Development and option controls

Evaluate all six initial candidates on the original eight-source, 40-QA validation split plus 16 static conditions. Original-order accuracy is a screen, not a publication result. Candidate selection uses source-macro accuracy, then micro accuracy, then lower radius and fewer parameters; never unseen-source labels or scores.

An apparently useful candidate must be evaluated on all three cyclic option rotations (0,1,2), mapping predictions back to semantic answer indices. Every comparison uses identical prompts and full model forwards. Static cases have no retained QA gold. A train-only two-parameter letter-logit bias is a scoring-calibration diagnostic, never an internal-memory intervention. Report gains and regressions against baseline per source and semantic question, including ties. V1's 16-source test is available only as explicitly seen development replication. Do not make extra hyperparameter choices from a new confirmation score.

No mandatory success is assumed. If normalization changes outputs but does not improve held-out development accuracy, reject the sufficiency of the amplitude hypothesis and record the result. Next mechanism candidates require a separate development addendum before execution: a generic low-rank residual control or earlier-layer attention routing. No unrestricted layer/seed/hyperparameter search.

## Fresh confirmation and claims

The 32-source selection at data/internal_routing_v2/selection.json is already frozen with SHA256 c4f06cfd0088de99202c3fb607a18d36ba2a0376c66e2ec5b6903c416211ceca. It excludes all 64 seen source IDs. No model score from those sources may be inspected before final method, all checkpoints, rotations, contrasts, and analysis code are frozen. Content hashes and near-duplicate limitations must be audited. The final protocol will state all evaluated methods and paired source-level confidence intervals. Repeated option orders are correlated measurements, not independent samples.

The primary research claim requires measured improvement on unseen sources and option-order robustness. A local development improvement, oracle optimization, calibration, or larger perturbation is insufficient. No claim of superiority to all context-memory methods or state-of-the-art systems follows from this experiment.

## Resource contract

Only root runs a single GPU process. ResourceGuard remains unchanged: 300-second process limit, two CPU threads, one interop thread, two-core affinity, low process priority, CUDA allocator fraction .65, initial RAM >=12GiB, running RAM >=6GiB, RSS <=8GiB, initial GPU <=65C and <=256MiB used with >=6500MiB free, stop >=78C or free GPU <1024MiB. Each process releases its model and allocator before the next. No driver, clock, power, firmware, or threshold changes.
