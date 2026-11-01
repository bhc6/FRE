# Spectral Memory Probe

Run type: `synthetic_smoke_test`

This is an offline probe, not an end-to-end memory evaluation.

| Model | Features | Test MSE | Spearman | Group NDCG |
| --- | ---: | ---: | ---: | ---: |
| M1_semantic | 18 | 0.061374 | -0.1368 | 0.7750 |
| M2_semantic_temporal | 114 | 0.056932 | 0.0278 | 0.8013 |
| M3_semantic_spectral | 390 | 0.053829 | 0.0531 | 0.8053 |
| M4_all | 486 | 0.052160 | 0.0692 | 0.8050 |
| M5_ordered_control | 306 | 0.057210 | 0.1362 | 0.8586 |
| M6_complex_control | 330 | 0.068238 | 0.0717 | 0.8852 |

## Limits

- Synthetic runs test plumbing only; their labels are independent of hidden-state features.
- Summed individual utility is a proxy: selected sets require fresh downstream evaluation.
- Ridge probes cannot replace matched nonlinear temporal baselines or causal task interventions.
- Intervals use equal-weight independent split groups, not individual event rows.
- Spectral features include ordered local STFT summaries; they are not power-only features.
