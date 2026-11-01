# Visual encoder spectra: development audit

This report uses only complete shards from the same eight old validation sources. Source counts below are the evidence units; controls and layer-zero aliases are not extra independent samples.

All captured native/passive/restored vocabulary logits, full logical feature hashes, input-ID hashes, frame/manifest hashes, source snapshots, immutable revisions and recorded resource limits passed audit. Temporal spectra were independently recomputed with a full complex FFT, then paired into positive frequencies; spatial FFT2 arrays were independently recomputed from full features.

Input pixel tensors were hashed during capture but are not stored. This report verifies generated gray RGB bytes, not independently reconstructed postprocessor pixel constancy. Resource monitoring samples every two seconds and can miss short peaks.

Completed shards: 4. Excluded unfinished/failed shards: 1. Audit checks passed: 6554.

| Model/stage | Sources | Real − static temporal AC fraction | Pooled temporal AC fraction difference | Real − gray spatial AC fraction | Gray spatial AC fraction | Temporal AC in spatial-constant subspace |
|---|---:|---:|---:|---:|---:|---:|
| qwen / llm_layer0_visual_input | 8 | 0.287053 | 0.016197 | 0.0962298 | 0.472683 | 0.0245762 |
| qwen / vision_final_postmerger | 8 | 0.287053 | 0.016197 | 0.0962298 | 0.472683 | 0.0245762 |
| qwen / vision_final_premerger | 8 | 0.192872 | 0.000996411 | -0.0160155 | 0.354005 | 0.00356908 |
| smol / llm_layer0_visual_input | 4 | 0.118705 | 0.00785842 | 0.13509 | 0.318202 | 0.0370494 |
| smol / vision_connector_output | 4 | 0.118705 | 0.00785842 | 0.13509 | 0.318202 | 0.0370494 |
| smol / vision_final_postnorm | 4 | 0.286666 | 0.0110329 | 0.151976 | 0.610746 | 0.00950302 |

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
| smol / llm_layer0_visual_input / gray | 4 | 1.6733e+07 | 17.0216 | 0 | 0 | 0.318202 |
| smol / llm_layer0_visual_input / real | 4 | 2.55399e+07 | 25.9806 | 0.118705 | 0.00785842 | 0.453292 |
| smol / llm_layer0_visual_input / static_first | 4 | 2.5824e+07 | 26.2695 | 0 | 0 | 0.44179 |
| smol / llm_layer0_visual_input / static_last | 4 | 2.62659e+07 | 26.7191 | 0 | 0 | 0.460221 |
| smol / vision_connector_output / gray | 4 | 1.6733e+07 | 17.0216 | 0 | 0 | 0.318202 |
| smol / vision_connector_output / real | 4 | 2.55399e+07 | 25.9806 | 0.118705 | 0.00785842 | 0.453292 |
| smol / vision_connector_output / static_first | 4 | 2.5824e+07 | 26.2695 | 0 | 0 | 0.44179 |
| smol / vision_connector_output / static_last | 4 | 2.62659e+07 | 26.7191 | 0 | 0 | 0.460221 |
| smol / vision_final_postnorm / gray | 4 | 8.16129e+06 | 0.648601 | 0 | 0 | 0.610746 |
| smol / vision_final_postnorm / real | 4 | 1.01861e+07 | 0.809515 | 0.286666 | 0.0110329 | 0.762722 |
| smol / vision_final_postnorm / static_first | 4 | 1.02922e+07 | 0.817949 | 0 | 0 | 0.761198 |
| smol / vision_final_postnorm / static_last | 4 | 1.06314e+07 | 0.844907 | 0 | 0 | 0.77162 |

Gray feature hashes identify repeated controls; counts below are not additional independent visual contents:

- qwen / llm_layer0_visual_input: 1 unique logical feature hashes across 8 source rows.
- qwen / vision_final_postmerger: 1 unique logical feature hashes across 8 source rows.
- qwen / vision_final_premerger: 1 unique logical feature hashes across 8 source rows.
- smol / llm_layer0_visual_input: 1 unique logical feature hashes across 4 source rows.
- smol / vision_connector_output: 1 unique logical feature hashes across 4 source rows.
- smol / vision_final_postnorm: 1 unique logical feature hashes across 4 source rows.

Raw and per-element energies, DC/AC fractions, per-source contrasts, temporal frequency vectors and exact spatial frequency coordinates are retained in summary.json and CSV files. Temporal conditional-AC distributions are undefined when AC is zero. Spatial coordinates are cycles per full grid extent; pre/post Nyquist-normalized bins must not be treated as equal frequencies.

The pure space-to-depth control is invertible and preserves every feature value and total/per-element energy while changing the reported spatial AC fraction. Learned projection additionally changes feature dimension and normalization. Neither lower raw energy nor lower spatial AC establishes loss of information. Gray spatial AC can originate from position-dependent trained computation, and is not automatically visual texture.

These are descriptive development results, not evidence of long-term memory, model forgetting, a causal bottleneck, or out-of-sample benefit. LLM layer-zero visual features are exact aliases of the projection output and do not add an independent stage.
