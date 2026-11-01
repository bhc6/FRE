# Native visual-encoder spectra: fixed development protocol

Written 2026-11-01 before running this probe. The question is whether temporal and spatial structure is visible in unweighted visual-encoder features, distinct from attention-weighted context spectra. This is a descriptive representation diagnostic, not a method or memory-performance claim. The fresh 32-source confirmation set is excluded by the existing old-validation manifest hashes.

## Sources and four controls

Use the first question after sorting `(video_id, query_id)` for each of the same eight old validation sources. Each source has real sixteen frames, sixteen copies of the first frame, sixteen copies of the last frame, and sixteen synthetic uniform gray frames with RGB `(127,127,127)`. Gray uses that source's original first-frame width/height; all four conditions retain the original displayed timestamps and question/options. Only real has gold. Gray/source file provenance and actual RGB-byte hashes are distinguished explicitly.

Both readers use their pinned native processors and complete uncached model forwards:

- Qwen3-VL-2B-Instruct revision `89644892e4d85e24eaac8bacfd4f463576704203`.
- SmolVLM2-500M-Video-Instruct revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.

The all-source run has 32 conditions per model. Explicit source IDs permit resource-bounded shards. Repeated gray conditions with the same processed geometry are not independent visual contents; source repetition is bookkeeping, not additional evidence.

## Exact native stages and grid recovery

All grids are logical `[T,H,W,D]`. No FFT runs across `D`.

### Qwen3-VL, Transformers 4.57.1

The official video processor starts from `[B,T_raw,C,pixel_H,pixel_W]`, groups two frames into each temporal patch, and performs:

`view(B,T,2,C,H/m,m,patch_size,W/m,m,patch_size)`

`permute(0,1,4,7,5,8,3,2,6,9)`

before flattening patch tokens. Therefore the token order is `[T,H/m,W/m,m_h,m_w]`, not ordinary `[T,H,W]`. The native vision positional interpolation and rotary-position construction use the same block-major order; vision blocks preserve it. The final merger groups the contiguous `m*m` inner tokens.

Capture the final merger input before its own LayerNorm. Restore spatial adjacency with:

`flat.view(T,H/m,W/m,m,m,D).transpose(0,1,3,2,4,5).reshape(T,H,W,D)`.

Capture the final merger output and reshape directly to `[T,H/m,W/m,D_text]`. Here `m=2`, patch stride is 16 processed pixels before merging and 32 after merging. Do not hook or aggregate the auxiliary DeepStack mergers. Capture layer-zero language-model input only at the existing visual token positions and require it to equal the final-merger output exactly.

Qwen's `T=8` is eight adjacent-frame tubelets from sixteen supplied frames. This is learned temporal patch mixing and reduced temporal sampling, not a sixteen-frame feature sequence. Its time coordinates are the native adjacent-frame centers. The vision attention cumulative sequence lengths separate temporal groups; identical-frame temporal AC is an empirical control to check, not a required zero baseline.

### SmolVLM2, Transformers 4.57.1

The native vision embedding is a Conv2d patch grid followed by `flatten(2).transpose(1,2)`, so vision sequence positions are row-major. Capture `vision_model.last_hidden_state`, including its final post-LayerNorm. With 512-pixel images and 16-pixel patches this is `[16,32,32,768]`.

The connector's actual pixel shuffle reshapes `[T,H,W,D]` to `[T,H,W/s,D*s]`, transposes coarse width before height, packs `s` neighboring height positions, transposes back, then flattens. It preserves row-major coarse spatial cells; each cell's channel bundle orders local row, local column and feature channel. The factor is `s=4`, giving `[16,8,8,D_text]` after its modality projection. Synthetic checks compare the installed connector's pixel-shuffle result with explicit local-block indexing and its inverse. Capture connector output and layer-zero visual inputs, requiring exact equality.

This extraction never treats a channel bundle introduced by merging as a spatial axis or performs FFT on its arbitrary channel ordering.

## Spectral definitions

For `X[T,H,W,D]`, preserve raw DC and AC rather than subtracting means first.

1. Temporal full-patch spectrum: orthonormal real FFT along `T` for every spatial location/channel; square magnitudes and sum channel power. Save `power[f,h,w]`; double only interior real frequencies, leaving DC and even-length Nyquist undoubled. Summing this array equals `sum X^2`.
2. Spatial full-patch spectrum: full complex orthonormal FFT2 over `H,W` independently for every time/channel. Save channel-summed `power[t,kh,kw]` in unshifted FFT order. Its sum equals `sum X^2`; no real-FFT multiplicity factor applies.
3. Spatial-mean temporal control: first average `H,W` and then FFT along `T`. Save each temporal frequency. This operation can cancel out-of-phase spatial variation and is not interchangeable with summing per-patch power. Also report `H*W*AC(spatial mean)/AC(full patch)` where defined.

Compute in float64 using 32-channel chunks after copying native activations to CPU float32. Record raw energy, per-element energy, DC/AC, AC fraction, each temporal frequency, each spatial frequency in NPZ, per-time spatial DC/AC, and Parseval errors. Require relative Parseval error at most `1e-8`; zero-energy fractions are null rather than invented.

Frequency metadata includes cycles per time group, cycles per sampled clip extent, cycles per spatial cell, and `fftfreq(H)*H` / `fftfreq(W)*W` cycles per full processed grid extent. There is no claim of antialiased physical motion frequency. Spatial dimensions and processed-pixel stride are retained. Pre/post-merge Nyquist-normalized bands are not the same cycles/image bands; primary real/gray/static comparisons are within the same stage/grid. Cross-stage energy must be shown per element and/or as fractions, with dimension and normalization changes disclosed. A lower raw energy or AC after projection is not proof that information was lost.

The temporal `cycles_per_sampled_clip_extent` field counts harmonics over the DFT period of `T` sample intervals; it does not equate that period with the first-to-last displayed timestamp span of `T-1` intervals. Use the saved actual timestamps when relating a sampled frequency to seconds.

## Read-only parity, storage and provenance

For each condition, run native, passive-capture and restored native complete forwards. Require all complete-vocabulary logits to agree exactly. Hooks only copy native tensors to CPU, never replace outputs. Hash all model inputs before/after. Save logits, input IDs, option IDs, full-feature representations, geometry, source/RGB provenance and spectral arrays in per-source compressed NPZ files.

Full-feature storage is lossless float32. If all temporal slices are exactly equal, store one slice with `exact_time_repeat` and the logical `[T,H,W,D]` shape; reconstruct by exact repetition. If layer-zero visual input equals projector output, store `exact_stage_alias` with the same full logical hash. These are exact storage encodings, not feature compression approximations or discarded spatial cells. Dense real features otherwise retain every element. Record logical feature hashes and encoding metadata. Archives for each source must stay below 100 MiB; the full model can be split across source directories/shards. No all-layer features are stored.

Use unchanged ResourceGuard: one GPU process, 300 seconds, two CPU threads/one interop thread, .65 allocator fraction, existing temperature/RAM checks. Snapshot the probe, backend/dependency sources, protocol, installed implementation hashes and immutable model/processor revisions. Require clean resource completion, complete case counts and unchanged source/data hashes. Partial runs are not complete results.

## Interpretation

Real versus static and gray controls can reveal content-associated temporal/spatial structure at visual-encoder stages. Uniform gray can still yield structured spatial spectra through positional embeddings and trained computation; spatial AC is not automatically image texture. Temporal AC at LLM stages may include causal-position/context effects, whereas this probe's encoder stages remove attention-weight envelopes from the measurement.

Frame2Freq's adapter performs temporal spectral processing per spatial location/channel, while its supplementary representation diagnostic uses temporal FFT after spatial aggregation. LFM processes spatial spectra of grouped intermediate visual representations before the connector. Those stages/axes motivate measuring the actual visual grid here; they do not validate an inference from spectral concentration to memory preservation or causal use. The present final-encoder-stage probe is not a reproduction of either trained method.

The primary-source stage and transform-axis audit is recorded separately in `docs/encoder_spectrum_source_alignment_v2.md`.

Report descriptive stage/condition spectra first. Do not force a high-frequency novelty claim, infer failure from low final-layer visual attention mass, or call successful static discrimination a long-term-memory mechanism. Native task relevance and causal benefit require separate experiments.

CLI: `.\.venv-cuda\Scripts\python.exe -m spectral_memory.vision_encoder_probe_v2 --model qwen --questions <old_validation_questions> --frames-manifest <old_validation_frames> --video-ids <optional_source_list> --output-dir <new_directory>`. Use `--model smol` for the second reader.
