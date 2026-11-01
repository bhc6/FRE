# Independent SmolVLM2 reader

This reader consumes only the persisted RGB memory frames, their retained timestamps, and the question/options prompt. It does not load the hidden trajectory or receive an answer label. It is an independent reader sensitivity check; model quality on this study must be measured.

## Fixed resources

- Model: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
- Immutable revision: `7b375e1b73b11138ff12fe22c8f2822d8fe03467`
- Official F32 weight file: 2,029,990,624 bytes; SHA-256 `b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3`.
- Parameters: 507,482,304. BF16 parameter storage alone is approximately 1.015 GB; this does not estimate peak runtime memory.
- Transformers: `4.57.1`; extra processor dependency: `num2words==0.5.14`.
- Local snapshot: `.cache/huggingface/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467` under the resolved project directory.

`pip install -e ".[extract,smolvlm]"` declares the adapter dependencies. Use the already provisioned CUDA environment for GPU work. Loading uses local files only, disables remote code, and performs no download.

## Interface and accounting

Use `spectral_memory.smolvlm_backend.load_backend()` with the same reader interface as the Qwen backend: `metadata`, `prepare_text(question)`, `prepare_video(frames, timestamps_s, question, max_pixels)`, and `choose_batch(batch, option_count)`.

Pass `max_pixels=262144` or greater. The official video processor converts each frame to 512 by 512 and emits 64 image tokens per frame. Its spatial transformation performs an aspect-preserving intermediate resize followed by square resizing. No frame is resampled, skipped, duplicated or fetched from an external source by the adapter. Persisted RGB bytes remain the memory budget; reader visual-token counts are reported separately and differ from Qwen's counts.

The prompt uses the official SmolVLM per-frame `MM:SS` timestamps. The upstream formatter truncates fractional seconds, so two nearby frames may have identical textual timestamps. Full source timestamps remain in metadata, and no FPS is inferred. The audited version uses the official fast video processor's BICUBIC fallback for the checkpoint's LANCZOS setting; this is recorded as a processor limitation, not claimed equivalent to the older PIL/LANCZOS path.

Choices are the argmax over unique single-token option letters. For this tokenizer, A/B/C are IDs 49/50/51. The output records option logits/probabilities, the unconstrained top token, prompt/hash, model/processor revision, actual input/visual tokens, forward elapsed time and peak CUDA allocated/reserved memory. Forward timing excludes input preparation and file I/O. There is no reader claim about streaming operation or measured long-term memory utility without the corresponding experiment.

## Verification

`outputs/smolvlm_cpu_verification/verification.json` records the actual processor and mock checks. These checks do not measure video-QA accuracy, runtime VRAM, or GPU throughput. The root experiment runner owns real GPU execution.

## Official sources checked on 2026-11-01

- Model card and frozen files: https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/tree/7b375e1b73b11138ff12fe22c8f2822d8fe03467
- Model metadata and exact public file sizes: https://huggingface.co/api/models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct?blobs=true
- Versioned Transformers documentation: https://huggingface.co/docs/transformers/v4.57.1/model_doc/smolvlm
- Versioned processor implementation: https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/smolvlm/processing_smolvlm.py
- Versioned video processor: https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/smolvlm/video_processing_smolvlm.py

The larger official `SmolVLM2-2.2B-Instruct` checkpoint has 8,987,224,056 bytes of public F32 weights at revision `482adb537c021c86670beed01cd58990d01e72e4`; it was not downloaded. Published model-card benchmarks do not establish performance on this study's retained-frame inputs.
