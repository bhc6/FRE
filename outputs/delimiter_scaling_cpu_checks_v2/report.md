# Delimiter adaptation CPU review

The implementation and protocol were frozen before any scores from this baseline were inspected. CPU checks passed 78/78. No pretrained checkpoint weights were loaded and no GPU forward was run by this review.

- Verified the actual pinned Qwen and Smol tokenizer/config/processor files using synthetic RGB frames, including exact delimiter IDs and the processed frame-block layout.
- Qwen scales 16 positions: vision-start/end around eight time groups. Smol scales 32 fake-token positions around sixteen image groups; the intervening global-image marker is excluded.
- Missing or extra delimiter occurrences are rejected. Architecture selects zero-based Qwen layer 13 and Smol layer 15 without performance data.
- Tiny models use the real Transformers Qwen3-VL text and Llama decoder classes, SDPA and four layers. The scale-one hook preserves complete-vocabulary output exactly; all hooks are removed and the native output restores exactly.
- On zero-epsilon tiny models, next-layer delimiter RMSNorm and Q/K/V projection outputs remain approximately invariant under positive residual scales .5/2/4. DTS2 agrees with the fixed downstream update-damping control within numerical tolerance.
- Captures occur before Qwen head normalization and rotary embeddings. They are not final attention Q/K or measurements of increased visual attention mass.
- Fixed scaling rows contain no label-dependent computation. Main-run gold labels are retained only as metadata, with correctness deferred to offline analysis.
- The damping condition is mechanism-only and excluded from candidate selection. It adds one forward, yielding seven forwards and five JSONL condition rows per case.

The review does not establish native pretrained-model parity, task performance, or generalization. The runner independently gates complete-vocabulary scale-one identity, restored output, unchanged paired inputs, full output counts, source/protocol hashes and a clean ResourceGuard completion.

Frozen implementation SHA256: `2e51662a8e6c8269538c4c992ff5ec42463e77de04075c6d4c96e61370a0e830`.

Reproduce with `.\.venv-cuda\Scripts\python.exe outputs/delimiter_scaling_cpu_checks_v2/check.py`. Detailed checks, actual token layouts and numerical damping errors are in `checks.json`.
