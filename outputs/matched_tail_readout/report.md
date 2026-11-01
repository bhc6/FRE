# Stored-memory QA study

| Pairs | Method | Subset | Correct | Questions | Video macro | Mean bytes |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 0 | no_memory | all | 3 | 6 | 0.5000 | 0 |
| 0 | no_memory | memory | 1 | 1 | 1.0000 | 0 |
| 0 | no_memory | temporal_state | 2 | 4 | 0.5000 | 0 |
| 8 | first_half | all | 3 | 6 | 0.5000 | 7078860 |
| 8 | first_half | memory | 1 | 1 | 1.0000 | 7078860 |
| 8 | first_half | temporal_state | 2 | 4 | 0.5000 | 7078860 |
| 8 | last_half | all | 3 | 6 | 0.5000 | 7078860 |
| 8 | last_half | memory | 1 | 1 | 1.0000 | 7078860 |
| 8 | last_half | temporal_state | 2 | 4 | 0.5000 | 7078860 |
| 8 | last_pair_reference | all | 4 | 6 | 0.6667 | 885540 |
| 8 | last_pair_reference | memory | 1 | 1 | 1.0000 | 885540 |
| 8 | last_pair_reference | temporal_state | 3 | 4 | 0.7500 | 885540 |
| 8 | pair_uniform | all | 3 | 6 | 0.5000 | 7078860 |
| 8 | pair_uniform | memory | 1 | 1 | 1.0000 | 7078860 |
| 8 | pair_uniform | temporal_state | 2 | 4 | 0.5000 | 7078860 |
| 8 | tail2_old | all | 3 | 6 | 0.5000 | 7078860 |
| 8 | tail2_old | memory | 1 | 1 | 1.0000 | 7078860 |
| 8 | tail2_old | temporal_state | 2 | 4 | 0.5000 | 7078860 |
| 8 | tail2_recent | all | 3 | 6 | 0.5000 | 7078860 |
| 8 | tail2_recent | memory | 1 | 1 | 1.0000 | 7078860 |
| 8 | tail2_recent | temporal_state | 2 | 4 | 0.5000 | 7078860 |
| 8 | tail2_uniform | all | 3 | 6 | 0.5000 | 7078860 |
| 8 | tail2_uniform | memory | 1 | 1 | 1.0000 | 7078860 |
| 8 | tail2_uniform | temporal_state | 2 | 4 | 0.5000 | 7078860 |
| 8 | tail4_old | all | 3 | 6 | 0.5000 | 7078860 |
| 8 | tail4_old | memory | 1 | 1 | 1.0000 | 7078860 |
| 8 | tail4_old | temporal_state | 2 | 4 | 0.5000 | 7078860 |
| 8 | tail4_recent | all | 3 | 6 | 0.5000 | 7078860 |
| 8 | tail4_recent | memory | 1 | 1 | 1.0000 | 7078860 |
| 8 | tail4_recent | temporal_state | 2 | 4 | 0.5000 | 7078860 |
| 8 | tail4_uniform | all | 3 | 6 | 0.5000 | 7078860 |
| 8 | tail4_uniform | memory | 1 | 1 | 1.0000 | 7078860 |
| 8 | tail4_uniform | temporal_state | 2 | 4 | 0.5000 | 7078860 |

## Paired spectral comparisons

| Pairs | Baseline | Subset | Macro difference | 95% video bootstrap |
| ---: | --- | --- | ---: | --- |

## Limitations

- No participant IDs; video bootstrap does not establish cross-person generalization.
- Intervals are unadjusted descriptive comparisons; only the frozen primary contrast is confirmatory.
- Full/last-pair references have different budgets and are reader diagnostics.
- Per-video raw-image bytes are matched across same-budget methods; compute is reported separately.
- Short-video event memory is not lifelong streaming memory.
