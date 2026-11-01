# Spectral Memory Probe

Run type: `synthetic_smoke_test`

This is an offline probe, not an end-to-end memory evaluation.

| Model | Features | Test MSE | Spearman | Group NDCG |
| --- | ---: | ---: | ---: | ---: |
| M1_semantic | 66 | 0.058421 | -0.1497 | 0.7705 |
| M2_semantic_temporal | 66 | 0.049421 | 0.1385 | 0.8021 |
| M3_semantic_spectral | 66 | 0.050093 | 0.0371 | 0.8468 |
| M4_all | 66 | 0.049901 | 0.1531 | 0.8443 |
| M5_ordered_control | 66 | 0.051056 | 0.0208 | 0.8072 |
| M6_complex_control | 66 | 0.058802 | -0.0985 | 0.8691 |

## Limits

- Synthetic runs test plumbing only; their labels are independent of hidden-state features.
- Summed individual utility is a proxy: selected sets require fresh downstream evaluation.
- Ridge probes cannot replace matched nonlinear temporal baselines or causal task interventions.
- Intervals use equal-weight independent split groups, not individual event rows.
- Spectral features include ordered local STFT summaries; they are not power-only features.
