# Delimiter-token scaling adaptation: frozen development protocol

Written 2026-11-01 before this implementation is evaluated. This is an adaptation of the published Delimiter Token Scaling baseline, not a novelty claim, not the authors' original benchmark setting, and not a claim that visual attention mass increases. No new 32-source confirmation scores are opened by this development experiment.

The author repository is `MYMY-young/DelimScaling`, inspected at commit `a2880738e3ed7284953f3611e7c06a3c0ede9334`. Its Qwen2.5-VL implementation scales visual-delimiter residual hidden states after selected decoder blocks. Its README example for Qwen2.5-VL-3B-Instruct on Mantis uses scale 8 and layers 0,1,2,3. The following layer, scale, model and token choices are explicitly our architecture-based adaptation. Scale .5 is a reverse ablation; the paper's enhancement equation uses a factor greater than one.

## Fixed candidate definition

- Qwen/Qwen3-VL-2B-Instruct revision `89644892e4d85e24eaac8bacfd4f463576704203`, 28 text decoder layers; choose zero-based layer 13.
- HuggingFaceTB/SmolVLM2-500M-Video-Instruct revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`, 32 text decoder layers; choose zero-based layer 15.
- In both cases the layer is `floor((L-1)/2)`, selected from architecture before inspecting this baseline's scores. No layer search, training, checkpoint selection or scale tuning occurs.
- At that decoder block's output, multiply only verified delimiter token residual rows by each fixed scalar `.5`, `2`, or `4`. Keep all other token rows, attention implementations, model weights, input tensors and prompts unchanged. Scale `1` is an exact identity gate.
- Qwen tokens are `<|vision_start|>` = 151652 and `<|vision_end|>` = 151653, exactly bracketing eight processed video time groups: 16 scaled positions.
- Smol token is `<fake_token_around_image>` = 49189, exactly bracketing all sixteen frame groups. Each opening fake token is followed by `<global-img>` = 49152 and 64 `<image>` tokens; the closing fake follows those image tokens: 32 scaled positions. The global-image marker, image placeholders and question text are not scaled. Smol is a token-semantics adaptation, not an author-provided supported-model reproduction.
- Verify the actual tokenizer encoding, pinned configuration, processed visual layout, complete delimiter occurrence set, and token budgets on every case. Qwen retains the 1024 total / 512 visual token budget; Smol retains 2048 total / 1024 visual tokens from the existing cross-model probe.

## Inputs, gates and storage

Use the existing official three-choice question/16-frame case builder and cyclic option rotation semantics. The same source RGB frames and displayed timestamps feed both readers through their own frozen official processors. Static first/last conditions have no gold and are evaluated only at rotation zero. Labels are retained as metadata for offline scoring; the intervention function accepts tokens, positions, layer and scale through label-free model inputs.

Each case runs seven complete uncached forwards: native, scale-one identity with mechanism capture, scales .5/2/4, the fixed update-damping mechanism control below, then restored native. Save all complete-vocabulary logits in NPZ, including identity/zero/restored arrays, input IDs, delimiter positions, option-token IDs and small mechanism tensors. Require native = scale one = restored exactly over the entire vocabulary. Hash all inputs before/after and share the same tensors across conditions. The JSONL has five rows per case: identity, three scales and update damping. Correctness is recomputed offline from saved logits and gold, not used by any intervention.

Use the unchanged 300-second ResourceGuard with two CPU threads, one interop thread, one GPU process, .65 CUDA allocator fraction and existing RAM/temperature thresholds. Archive source files, protocol, manifest hashes, immutable model/processor revision, implementation hash, per-case archive hashes, completion counts and resource summary. Partial or failed matrices are not complete results. The CLI supports validation/test and explicit video IDs so a later frozen confirmation can use the same runner; this protocol authorizes development execution only on already seen sources.

## Mechanism capture and norm interpretation

Without an extra forward, capture the next layer's input RMSNorm output and Q/K/V projection outputs at delimiter positions for scale one, each scale and update damping. Store tensors and per-token/aggregate L2 ratios and relative differences from the exact scale-one reference. Projection captures precede Qwen Q/K head normalization and rotary embedding; do not label them final attention Q/K.

For positive lambda, ignoring any additional affine conventions,

`RMSNorm(lambda h) = gamma h / sqrt(mean(h^2) + epsilon/lambda^2)`.

Therefore positive hidden scaling is approximately canceled by the next RMSNorm; Qwen's Q/K head norms can further remove radial changes. No predicted lambda-fold Q/K/V increase or visual-attention-mass gain is assumed. Later effects can arise because the scaled residual changes the relative contribution of subsequent attention and FFN updates, together with finite precision and nonzero epsilon.

## Fixed mechanism-only update damping

Compare DTS scale 2 with a separate `update_damp0.5` forward. Leave the selected block's hidden output unscaled. In every later block, multiply only delimiter rows of both the attention output and the FFN output by .5 before their original residual additions. Other token rows remain unchanged. This has no learned parameters or searched hyperparameters and is excluded from candidate selection.

In an ideal pre-RMSNorm transformer with zero epsilon and no other downstream residual insertions, the scaled and damped paths satisfy `h_DTS = 2 h_damp` at selected delimiter positions and equal states elsewhere. Normalized attention/FFN inputs are consequently equal, and the residual relation propagates inductively; final normalization removes the remaining selected-token scale. Qwen's DeepStack injections are before the selected middle layer, while the Smol text stack has no such downstream insertion. Nonzero epsilon, BF16 arithmetic and any architecture violation can break exact numerical equality, so this is a checked mechanism prediction rather than a hard runtime equality gate. Store complete-vocabulary and option-logit differences and whether option predictions agree.

## Reporting and decision rule

Report every fixed scale for both models with micro accuracy, source-macro accuracy, three-choice NLL, repairs/regressions, static prediction changes and cyclic-order behavior. Aggregate rotations within each original query and bootstrap sources with all their questions/rotations, never treating repeated presentations as independent sources. Report selection and small-source uncertainty. Label .5 as the reverse ablation and update damping as mechanism-only, without quietly promoting either after looking at results.

Any retained improvement requires fixed choices before unseen-source confirmation. Failure here rejects neither the published method under its original early-layer settings nor all delimiter interventions. It only limits this stated middle-layer adaptation on these readers and development cases.
