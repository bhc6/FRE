# Visual encoder spectra: development audit

This report uses only complete shards from the same eight old validation sources. Source counts below are the evidence units; controls and layer-zero aliases are not extra independent samples.

All captured native/passive/restored vocabulary logits, full logical feature hashes, input-ID hashes, frame/manifest hashes, source snapshots, immutable revisions and recorded resource limits passed audit. Temporal spectra were independently recomputed with a full complex FFT, then paired into positive frequencies; spatial FFT2 arrays were independently recomputed from full features.

Input pixel tensors were hashed during capture but are not stored. This report verifies generated gray RGB bytes, not independently reconstructed postprocessor pixel constancy. Resource monitoring samples every two seconds and can miss short peaks.

Completed shards: 5. Excluded unfinished/failed shards: 0. Audit checks passed: 8723.

| Model/stage | Sources | Real − static temporal AC fraction | Pooled temporal AC fraction difference | Real − gray spatial AC fraction | Gray spatial AC fraction | Temporal AC in spatial-constant subspace |
|---|---:|---:|---:|---:|---:|---:|
| qwen / llm_layer0_visual_input | 8 | 0.287053 | 0.016197 | 0.0962298 | 0.472683 | 0.0245762 |
| qwen / vision_final_postmerger | 8 | 0.287053 | 0.016197 | 0.0962298 | 0.472683 | 0.0245762 |
| qwen / vision_final_premerger | 8 | 0.192872 | 0.000996411 | -0.0160155 | 0.354005 | 0.00356908 |
| smol / llm_layer0_visual_input | 8 | 0.14167 | 0.0101714 | 0.130376 | 0.318202 | 0.0383226 |
| smol / vision_connector_output | 8 | 0.14167 | 0.0101714 | 0.130376 | 0.318202 | 0.0383226 |
| smol / vision_final_postnorm | 8 | 0.329309 | 0.01453 | 0.146527 | 0.610746 | 0.0102489 |

The last column is H×W×AC(spatial mean)/AC(full patch): the proportion of temporal variation energy in the spatial-constant subspace. Zero AC makes it undefined. It measures energy, not retained information. A value below one measures cancellation introduced by spatial averaging; the pooled AC fraction can nevertheless increase because its denominator also changes.

| Model/stage/condition | Sources | Mean raw energy | Mean energy per element | Mean temporal AC fraction | Mean pooled temporal AC fraction | Mean spatial AC fraction |
|---|---:|---:|---:|---:|---:|---:|
| qwen / llm_layer0_visual_input / gray | 8 | 143718 | 0.146198 | 0 | 0 | 0.472683 |
| qwen / llm_layer0_visual_input / real | 8 | 220799 | 0.224608 | 0.287053 | 0.016197 | 0.568913 |
| qwen / llm_layer0_visual_input / static_first | 8 | 299254 | 0.304417 | 0 | 0 | 0.574909 |
| qwen / llm_layer0_visual_input / static_last | 8 | 320674 | 0.326206 | 0 | 0 | 0.593086 |
| qwen / vision_final_postmerger / gray | 8 | 143718 | 0.146198 | 0 | 0 | 0.472683 |
| qwen / vision_final_postmerger / real | 8 | 220799 | 0.224608 | 0.287053 | 0.016197 | 0.568913 |
| qwen / vision_final_postmerger / static_first | 8 | 299254 | 0.304417 | 0 | 0 | 0.574909 |
| qwen / vision_final_postmerger / static_last | 8 | 320674 | 0.326206 | 0 | 0 | 0.593086 |
| qwen / vision_final_premerger / gray | 8 | 6.81565e+09 | 3466.62 | 0 | 0 | 0.354005 |
| qwen / vision_final_premerger / real | 8 | 9.09765e+09 | 4627.3 | 0.192872 | 0.000996411 | 0.33799 |
| qwen / vision_final_premerger / static_first | 8 | 7.65413e+09 | 3893.09 | 0 | 0 | 0.381347 |
| qwen / vision_final_premerger / static_last | 8 | 7.91951e+09 | 4028.07 | 0 | 0 | 0.384928 |
| smol / llm_layer0_visual_input / gray | 8 | 1.6733e+07 | 17.0216 | 0 | 0 | 0.318202 |
| smol / llm_layer0_visual_input / real | 8 | 2.51916e+07 | 25.6263 | 0.14167 | 0.0101714 | 0.448578 |
| smol / llm_layer0_visual_input / static_first | 8 | 2.50651e+07 | 25.4975 | 0 | 0 | 0.439532 |
| smol / llm_layer0_visual_input / static_last | 8 | 2.6107e+07 | 26.5574 | 0 | 0 | 0.448339 |
| smol / vision_connector_output / gray | 8 | 1.6733e+07 | 17.0216 | 0 | 0 | 0.318202 |
| smol / vision_connector_output / real | 8 | 2.51916e+07 | 25.6263 | 0.14167 | 0.0101714 | 0.448578 |
| smol / vision_connector_output / static_first | 8 | 2.50651e+07 | 25.4975 | 0 | 0 | 0.439532 |
| smol / vision_connector_output / static_last | 8 | 2.6107e+07 | 26.5574 | 0 | 0 | 0.448339 |
| smol / vision_final_postnorm / gray | 8 | 8.16129e+06 | 0.648601 | 0 | 0 | 0.610746 |
| smol / vision_final_postnorm / real | 8 | 9.92099e+06 | 0.788449 | 0.329309 | 0.01453 | 0.757273 |
| smol / vision_final_postnorm / static_first | 8 | 9.95474e+06 | 0.791132 | 0 | 0 | 0.753021 |
| smol / vision_final_postnorm / static_last | 8 | 1.02479e+07 | 0.814432 | 0 | 0 | 0.764352 |

Gray feature hashes identify repeated controls; counts below are not additional independent visual contents:

- qwen / llm_layer0_visual_input: 1 unique logical feature hashes across 8 source rows.
- qwen / vision_final_postmerger: 1 unique logical feature hashes across 8 source rows.
- qwen / vision_final_premerger: 1 unique logical feature hashes across 8 source rows.
- smol / llm_layer0_visual_input: 1 unique logical feature hashes across 8 source rows.
- smol / vision_connector_output: 1 unique logical feature hashes across 8 source rows.
- smol / vision_final_postnorm: 1 unique logical feature hashes across 8 source rows.

Real-condition temporal AC distributions are source means of each source's normalized AC energy. Entries correspond to positive DFT harmonics 1…T/2, including Nyquist. Qwen T=8 tubelets and Smol T=16 frames are distinct sampling grids.

| Model/stage | Source count | Fraction of temporal AC at each positive harmonic |
|---|---:|---| 
| qwen / llm_layer0_visual_input | 8 | 0.357951, 0.269227, 0.255427, 0.117395 |
| qwen / vision_final_postmerger | 8 | 0.357951, 0.269227, 0.255427, 0.117395 |
| qwen / vision_final_premerger | 8 | 0.352036, 0.277969, 0.244294, 0.125701 |
| smol / llm_layer0_visual_input | 8 | 0.280581, 0.144559, 0.11845, 0.0997309, 0.103326, 0.102286, 0.102249, 0.0488186 |
| smol / vision_connector_output | 8 | 0.280581, 0.144559, 0.11845, 0.0997309, 0.103326, 0.102286, 0.102249, 0.0488186 |
| smol / vision_final_postnorm | 8 | 0.24831, 0.14395, 0.120994, 0.107545, 0.108434, 0.108415, 0.111069, 0.051282 |

Spatial AC quantiles use radius in the two coordinates (cycles/full-grid-height, cycles/full-grid-width). These are descriptive grid coordinates, not an isotropic physical frequency. The exact two-dimensional distributions remain in spatial_frequencies.csv.

| Model/stage/condition | Grid H×W | Mean within-source AC radius q50 | Mean within-source AC radius q90 |
|---|---|---:|---:|
| qwen / llm_layer0_visual_input / gray | 6×10 | 3 | 5 |
| qwen / llm_layer0_visual_input / real | 6×10 | 2.97855 | 5 |
| qwen / vision_final_postmerger / gray | 6×10 | 3 | 5 |
| qwen / vision_final_postmerger / real | 6×10 | 2.97855 | 5 |
| qwen / vision_final_premerger / gray | 12×20 | 5.65685 | 9.84886 |
| qwen / vision_final_premerger / real | 12×20 | 6.25227 | 9.79614 |
| smol / llm_layer0_visual_input / gray | 8×8 | 2.82843 | 4.47214 |
| smol / llm_layer0_visual_input / real | 8×8 | 2.89277 | 4.24264 |
| smol / vision_connector_output / gray | 8×8 | 2.82843 | 4.47214 |
| smol / vision_connector_output / real | 8×8 | 2.89277 | 4.24264 |
| smol / vision_final_postnorm / gray | 32×32 | 8.60233 | 16.0312 |
| smol / vision_final_postnorm / real | 32×32 | 8.32793 | 16.0574 |

Raw and per-element energies, DC/AC fractions, per-source contrasts, temporal frequency vectors and exact spatial frequency coordinates are retained in summary.json and CSV files. Temporal conditional-AC distributions are undefined when AC is zero. Spatial coordinates are cycles per full grid extent; pre/post Nyquist-normalized bins must not be treated as equal frequencies.

The pure space-to-depth control is invertible and preserves every feature value and total/per-element energy while changing the reported spatial AC fraction. Learned projection additionally changes feature dimension and normalization. Neither lower raw energy nor lower spatial AC establishes loss of information. Gray spatial AC can originate from position-dependent trained computation, and is not automatically visual texture.

These are descriptive development results, not evidence of long-term memory, model forgetting, a causal bottleneck, or out-of-sample benefit. LLM layer-zero visual features are exact aliases of the projection output and do not add an independent stage.
