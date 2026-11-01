# Visual encoder probe CPU review

57/57 checks passed before pretrained-model evaluation. The review used two CPU threads and one interop thread; no pretrained checkpoint weights or GPU forward were loaded.

- Qwen's actual installed processor was fed patch-coded synthetic RGB videos. Inverting its block-major pre-merger order recovered the intended `[8,4,6]` spatial/time codes exactly; naive flatten-to-raster reshaping was explicitly shown to fail.
- Installed `Qwen3VLVisionPatchMerger` grouping matched explicit spatial blocks on rectangular and square synthetic grids.
- Installed Smol pixel shuffle matched explicit local-row/local-column grouping for factors 2 and 4, and its inverse recovered the original grid exactly.
- Installed Smol vision Conv2d embedding with an indexing kernel and zero positional embedding confirmed row-major patch flattening.
- Both actual pinned processors/configs accepted synthetic gray sixteen-frame inputs and produced the required native time groups, merge factors and visual-token counts.
- Temporal rFFT energy matched an independent full complex FFT for even and odd lengths; spatial and temporal Parseval checks passed. Known-frequency and constant controls passed. Opposite-phase spatial signals retained temporal AC in the full-patch spectrum while canceling in the spatial-mean spectrum.
- Lossless exact-time-repeat storage and the exact projector-to-LLM alias reconstructed every value and logical feature hash.
- Synthetic whole-model hook tests using the native merger/connector classes preserved full-vocabulary outputs exactly, restored native output, recovered the pre-projection grid, verified projector/LLM token mapping and removed all hooks.

The runtime repeats whole-vocabulary native/passive/restored parity and token-layout checks on actual pretrained models. Those native GPU gates, task results and empirical feature spectra are not claimed by this CPU audit.

Frozen implementation SHA256: `05e9f79595895d8a81f82f8f96f5bbd97c183b49c3ce587e0bc4f0b4acb9929a`.

Reproduce using `.\.venv-cuda\Scripts\python.exe outputs/vision_encoder_probe_cpu_checks_v2/check.py`. Detailed per-check outcomes, processor geometry and source/protocol hashes are in `checks.json`.
