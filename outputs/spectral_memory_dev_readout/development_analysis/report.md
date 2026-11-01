# Development-only QA and evidence-coverage report

All 8 original development videos and all 36 questions; exploratory comparisons only. No multiple-testing adjustment or independent-confirmation claim.

Reader: Qwen/Qwen3-VL-2B-Instruct; completed records: 1008; method/budget conditions per question: 28.

## QA: all

| Budget pairs | Method | Actual pairs | Correct / questions | Video macro | Videos | Reference |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 0 | no_memory | 0 | 13/36 | 0.3563 | 8 | False |
| 2 | full_reference | 32 | 25/36 | 0.7250 | 8 | True |
| 2 | kcenter | 2 | 24/36 | 0.7000 | 8 | False |
| 2 | random | 2 | 24/36 | 0.7000 | 8 | False |
| 2 | recent | 2 | 25/36 | 0.6729 | 8 | False |
| 2 | spectral_balanced | 2 | 24/36 | 0.7000 | 8 | False |
| 2 | spectral_power | 2 | 24/36 | 0.7000 | 8 | False |
| 2 | temporal_difference | 2 | 24/36 | 0.7000 | 8 | False |
| 2 | temporal_reconstruction | 2 | 24/36 | 0.7000 | 8 | False |
| 2 | uniform | 2 | 24/36 | 0.7000 | 8 | False |
| 4 | full_reference | 32 | 25/36 | 0.7250 | 8 | True |
| 4 | kcenter | 4 | 26/36 | 0.7458 | 8 | False |
| 4 | random | 4 | 24/36 | 0.7000 | 8 | False |
| 4 | recent | 4 | 22/36 | 0.6062 | 8 | False |
| 4 | spectral_balanced | 4 | 25/36 | 0.7250 | 8 | False |
| 4 | spectral_power | 4 | 25/36 | 0.7250 | 8 | False |
| 4 | temporal_difference | 4 | 23/36 | 0.6750 | 8 | False |
| 4 | temporal_reconstruction | 4 | 25/36 | 0.7250 | 8 | False |
| 4 | uniform | 4 | 24/36 | 0.6625 | 8 | False |
| 8 | full_reference | 32 | 25/36 | 0.7250 | 8 | True |
| 8 | kcenter | 8 | 25/36 | 0.7250 | 8 | False |
| 8 | random | 8 | 25/36 | 0.7208 | 8 | False |
| 8 | recent | 8 | 24/36 | 0.6896 | 8 | False |
| 8 | spectral_balanced | 8 | 23/36 | 0.6750 | 8 | False |
| 8 | spectral_power | 8 | 26/36 | 0.7500 | 8 | False |
| 8 | temporal_difference | 8 | 25/36 | 0.7250 | 8 | False |
| 8 | temporal_reconstruction | 8 | 24/36 | 0.7000 | 8 | False |
| 8 | uniform | 8 | 25/36 | 0.7250 | 8 | False |

## QA: temporal_state

| Budget pairs | Method | Actual pairs | Correct / questions | Video macro | Videos | Reference |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 0 | no_memory | 0 | 5/8 | 0.6667 | 3 | False |
| 2 | full_reference | 32 | 6/8 | 0.8333 | 3 | True |
| 2 | kcenter | 2 | 6/8 | 0.8333 | 3 | False |
| 2 | random | 2 | 6/8 | 0.8333 | 3 | False |
| 2 | recent | 2 | 7/8 | 0.9167 | 3 | False |
| 2 | spectral_balanced | 2 | 6/8 | 0.8333 | 3 | False |
| 2 | spectral_power | 2 | 6/8 | 0.8333 | 3 | False |
| 2 | temporal_difference | 2 | 6/8 | 0.8333 | 3 | False |
| 2 | temporal_reconstruction | 2 | 6/8 | 0.8333 | 3 | False |
| 2 | uniform | 2 | 6/8 | 0.8333 | 3 | False |
| 4 | full_reference | 32 | 6/8 | 0.8333 | 3 | True |
| 4 | kcenter | 4 | 6/8 | 0.8333 | 3 | False |
| 4 | random | 4 | 6/8 | 0.8333 | 3 | False |
| 4 | recent | 4 | 6/8 | 0.8333 | 3 | False |
| 4 | spectral_balanced | 4 | 6/8 | 0.8333 | 3 | False |
| 4 | spectral_power | 4 | 6/8 | 0.8333 | 3 | False |
| 4 | temporal_difference | 4 | 6/8 | 0.8333 | 3 | False |
| 4 | temporal_reconstruction | 4 | 6/8 | 0.8333 | 3 | False |
| 4 | uniform | 4 | 6/8 | 0.8333 | 3 | False |
| 8 | full_reference | 32 | 6/8 | 0.8333 | 3 | True |
| 8 | kcenter | 8 | 6/8 | 0.8333 | 3 | False |
| 8 | random | 8 | 6/8 | 0.8333 | 3 | False |
| 8 | recent | 8 | 7/8 | 0.9167 | 3 | False |
| 8 | spectral_balanced | 8 | 6/8 | 0.8333 | 3 | False |
| 8 | spectral_power | 8 | 6/8 | 0.8333 | 3 | False |
| 8 | temporal_difference | 8 | 6/8 | 0.8333 | 3 | False |
| 8 | temporal_reconstruction | 8 | 6/8 | 0.8333 | 3 | False |
| 8 | uniform | 8 | 6/8 | 0.8333 | 3 | False |

## QA: memory

| Budget pairs | Method | Actual pairs | Correct / questions | Video macro | Videos | Reference |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 0 | no_memory | 0 | 1/1 | 1.0000 | 1 | False |
| 2 | full_reference | 32 | 1/1 | 1.0000 | 1 | True |
| 2 | kcenter | 2 | 1/1 | 1.0000 | 1 | False |
| 2 | random | 2 | 1/1 | 1.0000 | 1 | False |
| 2 | recent | 2 | 1/1 | 1.0000 | 1 | False |
| 2 | spectral_balanced | 2 | 1/1 | 1.0000 | 1 | False |
| 2 | spectral_power | 2 | 1/1 | 1.0000 | 1 | False |
| 2 | temporal_difference | 2 | 1/1 | 1.0000 | 1 | False |
| 2 | temporal_reconstruction | 2 | 1/1 | 1.0000 | 1 | False |
| 2 | uniform | 2 | 1/1 | 1.0000 | 1 | False |
| 4 | full_reference | 32 | 1/1 | 1.0000 | 1 | True |
| 4 | kcenter | 4 | 1/1 | 1.0000 | 1 | False |
| 4 | random | 4 | 1/1 | 1.0000 | 1 | False |
| 4 | recent | 4 | 1/1 | 1.0000 | 1 | False |
| 4 | spectral_balanced | 4 | 1/1 | 1.0000 | 1 | False |
| 4 | spectral_power | 4 | 1/1 | 1.0000 | 1 | False |
| 4 | temporal_difference | 4 | 1/1 | 1.0000 | 1 | False |
| 4 | temporal_reconstruction | 4 | 1/1 | 1.0000 | 1 | False |
| 4 | uniform | 4 | 1/1 | 1.0000 | 1 | False |
| 8 | full_reference | 32 | 1/1 | 1.0000 | 1 | True |
| 8 | kcenter | 8 | 1/1 | 1.0000 | 1 | False |
| 8 | random | 8 | 1/1 | 1.0000 | 1 | False |
| 8 | recent | 8 | 1/1 | 1.0000 | 1 | False |
| 8 | spectral_balanced | 8 | 1/1 | 1.0000 | 1 | False |
| 8 | spectral_power | 8 | 1/1 | 1.0000 | 1 | False |
| 8 | temporal_difference | 8 | 1/1 | 1.0000 | 1 | False |
| 8 | temporal_reconstruction | 8 | 1/1 | 1.0000 | 1 | False |
| 8 | uniform | 8 | 1/1 | 1.0000 | 1 | False |

## Temporal evidence coverage: all

| Pairs | Method | One-frame macro | One-frame micro | Events retained/observable | Two-frame macro | Intact-pair macro |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | kcenter | 0.1421 | 0.1571 | 11/70 | 0.0542 | 0.0586 |
| 2 | random | 0.1421 | 0.1571 | 11/70 | 0.0542 | 0.0586 |
| 2 | recent | 0.0660 | 0.0714 | 5/70 | 0.0669 | 0.0725 |
| 2 | spectral_balanced | 0.1421 | 0.1571 | 11/70 | 0.0542 | 0.0586 |
| 2 | spectral_power | 0.1421 | 0.1571 | 11/70 | 0.0542 | 0.0586 |
| 2 | temporal_difference | 0.1421 | 0.1571 | 11/70 | 0.0542 | 0.0586 |
| 2 | temporal_reconstruction | 0.1421 | 0.1571 | 11/70 | 0.0542 | 0.0586 |
| 2 | uniform | 0.1421 | 0.1571 | 11/70 | 0.0542 | 0.0586 |
| 4 | kcenter | 0.4281 | 0.4286 | 30/70 | 0.3001 | 0.3166 |
| 4 | random | 0.4263 | 0.4286 | 30/70 | 0.2976 | 0.3084 |
| 4 | recent | 0.2330 | 0.2286 | 16/70 | 0.2110 | 0.2218 |
| 4 | spectral_balanced | 0.3031 | 0.3143 | 22/70 | 0.2325 | 0.2478 |
| 4 | spectral_power | 0.3096 | 0.3143 | 22/70 | 0.2453 | 0.2513 |
| 4 | temporal_difference | 0.3295 | 0.3286 | 23/70 | 0.2018 | 0.2168 |
| 4 | temporal_reconstruction | 0.4355 | 0.4286 | 30/70 | 0.3388 | 0.3582 |
| 4 | uniform | 0.4390 | 0.4429 | 31/70 | 0.3333 | 0.3527 |
| 8 | kcenter | 0.7381 | 0.7000 | 49/70 | 0.6087 | 0.6344 |
| 8 | random | 0.6885 | 0.6571 | 46/70 | 0.5473 | 0.5451 |
| 8 | recent | 0.3148 | 0.3286 | 23/70 | 0.2966 | 0.2994 |
| 8 | spectral_balanced | 0.7096 | 0.7000 | 49/70 | 0.5783 | 0.6127 |
| 8 | spectral_power | 0.4943 | 0.5143 | 36/70 | 0.4265 | 0.4191 |
| 8 | temporal_difference | 0.5354 | 0.5286 | 37/70 | 0.3563 | 0.3810 |
| 8 | temporal_reconstruction | 0.7638 | 0.7571 | 53/70 | 0.6325 | 0.6669 |
| 8 | uniform | 0.8044 | 0.7857 | 55/70 | 0.6679 | 0.6941 |

Official instances: 71; zero-observation instances: 1. Two-frame and intact-pair values are temporal coverage, not visual readability or action-recognition accuracy.

## Temporal evidence coverage: short

| Pairs | Method | One-frame macro | One-frame micro | Events retained/observable | Two-frame macro | Intact-pair macro |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | kcenter | 0.2381 | 0.2632 | 5/19 | 0.1667 | 0.2381 |
| 2 | random | 0.2381 | 0.2632 | 5/19 | 0.1667 | 0.2381 |
| 2 | recent | 0.1190 | 0.1053 | 2/19 | 0.1190 | 0.1905 |
| 2 | spectral_balanced | 0.2381 | 0.2632 | 5/19 | 0.1667 | 0.2381 |
| 2 | spectral_power | 0.2381 | 0.2632 | 5/19 | 0.1667 | 0.2381 |
| 2 | temporal_difference | 0.2381 | 0.2632 | 5/19 | 0.1667 | 0.2381 |
| 2 | temporal_reconstruction | 0.2381 | 0.2632 | 5/19 | 0.1667 | 0.2381 |
| 2 | uniform | 0.2381 | 0.2632 | 5/19 | 0.1667 | 0.2381 |
| 4 | kcenter | 0.2738 | 0.3158 | 6/19 | 0.1667 | 0.2381 |
| 4 | random | 0.3571 | 0.3684 | 7/19 | 0.1667 | 0.2381 |
| 4 | recent | 0.1667 | 0.1579 | 3/19 | 0.1905 | 0.3333 |
| 4 | spectral_balanced | 0.2381 | 0.2632 | 5/19 | 0.1667 | 0.2381 |
| 4 | spectral_power | 0.3929 | 0.4211 | 8/19 | 0.3452 | 0.5000 |
| 4 | temporal_difference | 0.2738 | 0.3158 | 6/19 | 0.2024 | 0.2857 |
| 4 | temporal_reconstruction | 0.2381 | 0.2632 | 5/19 | 0.1667 | 0.2381 |
| 4 | uniform | 0.2381 | 0.2632 | 5/19 | 0.1667 | 0.2381 |
| 8 | kcenter | 0.5238 | 0.4737 | 9/19 | 0.3095 | 0.3810 |
| 8 | random | 0.5595 | 0.5263 | 10/19 | 0.2024 | 0.2857 |
| 8 | recent | 0.2857 | 0.2632 | 5/19 | 0.3333 | 0.4048 |
| 8 | spectral_balanced | 0.6190 | 0.5789 | 11/19 | 0.3095 | 0.4048 |
| 8 | spectral_power | 0.5119 | 0.5263 | 10/19 | 0.4524 | 0.5476 |
| 8 | temporal_difference | 0.4524 | 0.4211 | 8/19 | 0.2024 | 0.2857 |
| 8 | temporal_reconstruction | 0.5833 | 0.5263 | 10/19 | 0.2381 | 0.3333 |
| 8 | uniform | 0.4524 | 0.4211 | 8/19 | 0.3571 | 0.4286 |

Official instances: 20; zero-observation instances: 1. Two-frame and intact-pair values are temporal coverage, not visual readability or action-recognition accuracy.

## Same-budget paired QA contrasts

| Pairs | Method minus baseline | Subset | Video macro difference | 95% video-bootstrap interval |
| ---: | --- | --- | ---: | --- |
| 2 | kcenter minus uniform | all | 0.0000 | [0.0000, 0.0000] |
| 2 | kcenter minus uniform | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 2 | kcenter minus uniform | memory | 0.0000 | [0.0000, 0.0000] |
| 2 | random minus kcenter | all | 0.0000 | [0.0000, 0.0000] |
| 2 | random minus uniform | all | 0.0000 | [0.0000, 0.0000] |
| 2 | random minus kcenter | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 2 | random minus uniform | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 2 | random minus kcenter | memory | 0.0000 | [0.0000, 0.0000] |
| 2 | random minus uniform | memory | 0.0000 | [0.0000, 0.0000] |
| 2 | recent minus kcenter | all | -0.0271 | [-0.2125, 0.1604] |
| 2 | recent minus uniform | all | -0.0271 | [-0.2125, 0.1604] |
| 2 | recent minus kcenter | temporal_state | 0.0833 | [0.0000, 0.2500] |
| 2 | recent minus uniform | temporal_state | 0.0833 | [0.0000, 0.2500] |
| 2 | recent minus kcenter | memory | 0.0000 | [0.0000, 0.0000] |
| 2 | recent minus uniform | memory | 0.0000 | [0.0000, 0.0000] |
| 2 | spectral_balanced minus kcenter | all | 0.0000 | [0.0000, 0.0000] |
| 2 | spectral_balanced minus uniform | all | 0.0000 | [0.0000, 0.0000] |
| 2 | spectral_balanced minus kcenter | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 2 | spectral_balanced minus uniform | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 2 | spectral_balanced minus kcenter | memory | 0.0000 | [0.0000, 0.0000] |
| 2 | spectral_balanced minus uniform | memory | 0.0000 | [0.0000, 0.0000] |
| 2 | spectral_power minus kcenter | all | 0.0000 | [0.0000, 0.0000] |
| 2 | spectral_power minus uniform | all | 0.0000 | [0.0000, 0.0000] |
| 2 | spectral_power minus kcenter | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 2 | spectral_power minus uniform | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 2 | spectral_power minus kcenter | memory | 0.0000 | [0.0000, 0.0000] |
| 2 | spectral_power minus uniform | memory | 0.0000 | [0.0000, 0.0000] |
| 2 | temporal_difference minus kcenter | all | 0.0000 | [0.0000, 0.0000] |
| 2 | temporal_difference minus uniform | all | 0.0000 | [0.0000, 0.0000] |
| 2 | temporal_difference minus kcenter | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 2 | temporal_difference minus uniform | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 2 | temporal_difference minus kcenter | memory | 0.0000 | [0.0000, 0.0000] |
| 2 | temporal_difference minus uniform | memory | 0.0000 | [0.0000, 0.0000] |
| 2 | temporal_reconstruction minus kcenter | all | 0.0000 | [0.0000, 0.0000] |
| 2 | temporal_reconstruction minus uniform | all | 0.0000 | [0.0000, 0.0000] |
| 2 | temporal_reconstruction minus kcenter | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 2 | temporal_reconstruction minus uniform | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 2 | temporal_reconstruction minus kcenter | memory | 0.0000 | [0.0000, 0.0000] |
| 2 | temporal_reconstruction minus uniform | memory | 0.0000 | [0.0000, 0.0000] |
| 2 | uniform minus kcenter | all | 0.0000 | [0.0000, 0.0000] |
| 2 | uniform minus kcenter | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 2 | uniform minus kcenter | memory | 0.0000 | [0.0000, 0.0000] |
| 4 | kcenter minus uniform | all | 0.0833 | [-0.0333, 0.2292] |
| 4 | kcenter minus uniform | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 4 | kcenter minus uniform | memory | 0.0000 | [0.0000, 0.0000] |
| 4 | random minus kcenter | all | -0.0458 | [-0.1083, 0.0000] |
| 4 | random minus uniform | all | 0.0375 | [-0.1250, 0.2125] |
| 4 | random minus kcenter | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 4 | random minus uniform | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 4 | random minus kcenter | memory | 0.0000 | [0.0000, 0.0000] |
| 4 | random minus uniform | memory | 0.0000 | [0.0000, 0.0000] |
| 4 | recent minus kcenter | all | -0.1396 | [-0.2750, -0.0021] |
| 4 | recent minus uniform | all | -0.0562 | [-0.2062, 0.1000] |
| 4 | recent minus kcenter | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 4 | recent minus uniform | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 4 | recent minus kcenter | memory | 0.0000 | [0.0000, 0.0000] |
| 4 | recent minus uniform | memory | 0.0000 | [0.0000, 0.0000] |
| 4 | spectral_balanced minus kcenter | all | -0.0208 | [-0.0958, 0.0542] |
| 4 | spectral_balanced minus uniform | all | 0.0625 | [-0.1000, 0.2375] |
| 4 | spectral_balanced minus kcenter | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 4 | spectral_balanced minus uniform | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 4 | spectral_balanced minus kcenter | memory | 0.0000 | [0.0000, 0.0000] |
| 4 | spectral_balanced minus uniform | memory | 0.0000 | [0.0000, 0.0000] |
| 4 | spectral_power minus kcenter | all | -0.0208 | [-0.0958, 0.0542] |
| 4 | spectral_power minus uniform | all | 0.0625 | [-0.1000, 0.2375] |
| 4 | spectral_power minus kcenter | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 4 | spectral_power minus uniform | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 4 | spectral_power minus kcenter | memory | 0.0000 | [0.0000, 0.0000] |
| 4 | spectral_power minus uniform | memory | 0.0000 | [0.0000, 0.0000] |
| 4 | temporal_difference minus kcenter | all | -0.0708 | [-0.1375, -0.0208] |
| 4 | temporal_difference minus uniform | all | 0.0125 | [-0.1500, 0.1875] |
| 4 | temporal_difference minus kcenter | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 4 | temporal_difference minus uniform | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 4 | temporal_difference minus kcenter | memory | 0.0000 | [0.0000, 0.0000] |
| 4 | temporal_difference minus uniform | memory | 0.0000 | [0.0000, 0.0000] |
| 4 | temporal_reconstruction minus kcenter | all | -0.0208 | [-0.0625, 0.0000] |
| 4 | temporal_reconstruction minus uniform | all | 0.0625 | [-0.0500, 0.2125] |
| 4 | temporal_reconstruction minus kcenter | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 4 | temporal_reconstruction minus uniform | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 4 | temporal_reconstruction minus kcenter | memory | 0.0000 | [0.0000, 0.0000] |
| 4 | temporal_reconstruction minus uniform | memory | 0.0000 | [0.0000, 0.0000] |
| 4 | uniform minus kcenter | all | -0.0833 | [-0.2292, 0.0333] |
| 4 | uniform minus kcenter | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 4 | uniform minus kcenter | memory | 0.0000 | [0.0000, 0.0000] |
| 8 | kcenter minus uniform | all | 0.0000 | [0.0000, 0.0000] |
| 8 | kcenter minus uniform | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 8 | kcenter minus uniform | memory | 0.0000 | [0.0000, 0.0000] |
| 8 | random minus kcenter | all | -0.0042 | [-0.0750, 0.0625] |
| 8 | random minus uniform | all | -0.0042 | [-0.0750, 0.0625] |
| 8 | random minus kcenter | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 8 | random minus uniform | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 8 | random minus kcenter | memory | 0.0000 | [0.0000, 0.0000] |
| 8 | random minus uniform | memory | 0.0000 | [0.0000, 0.0000] |
| 8 | recent minus kcenter | all | -0.0354 | [-0.1437, 0.0708] |
| 8 | recent minus uniform | all | -0.0354 | [-0.1437, 0.0708] |
| 8 | recent minus kcenter | temporal_state | 0.0833 | [0.0000, 0.2500] |
| 8 | recent minus uniform | temporal_state | 0.0833 | [0.0000, 0.2500] |
| 8 | recent minus kcenter | memory | 0.0000 | [0.0000, 0.0000] |
| 8 | recent minus uniform | memory | 0.0000 | [0.0000, 0.0000] |
| 8 | spectral_balanced minus kcenter | all | -0.0500 | [-0.1250, 0.0000] |
| 8 | spectral_balanced minus uniform | all | -0.0500 | [-0.1250, 0.0000] |
| 8 | spectral_balanced minus kcenter | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 8 | spectral_balanced minus uniform | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 8 | spectral_balanced minus kcenter | memory | 0.0000 | [0.0000, 0.0000] |
| 8 | spectral_balanced minus uniform | memory | 0.0000 | [0.0000, 0.0000] |
| 8 | spectral_power minus kcenter | all | 0.0250 | [-0.0750, 0.1500] |
| 8 | spectral_power minus uniform | all | 0.0250 | [-0.0750, 0.1500] |
| 8 | spectral_power minus kcenter | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 8 | spectral_power minus uniform | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 8 | spectral_power minus kcenter | memory | 0.0000 | [0.0000, 0.0000] |
| 8 | spectral_power minus uniform | memory | 0.0000 | [0.0000, 0.0000] |
| 8 | temporal_difference minus kcenter | all | 0.0000 | [0.0000, 0.0000] |
| 8 | temporal_difference minus uniform | all | 0.0000 | [0.0000, 0.0000] |
| 8 | temporal_difference minus kcenter | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 8 | temporal_difference minus uniform | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 8 | temporal_difference minus kcenter | memory | 0.0000 | [0.0000, 0.0000] |
| 8 | temporal_difference minus uniform | memory | 0.0000 | [0.0000, 0.0000] |
| 8 | temporal_reconstruction minus kcenter | all | -0.0250 | [-0.0750, 0.0000] |
| 8 | temporal_reconstruction minus uniform | all | -0.0250 | [-0.0750, 0.0000] |
| 8 | temporal_reconstruction minus kcenter | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 8 | temporal_reconstruction minus uniform | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 8 | temporal_reconstruction minus kcenter | memory | 0.0000 | [0.0000, 0.0000] |
| 8 | temporal_reconstruction minus uniform | memory | 0.0000 | [0.0000, 0.0000] |
| 8 | uniform minus kcenter | all | 0.0000 | [0.0000, 0.0000] |
| 8 | uniform minus kcenter | temporal_state | 0.0000 | [0.0000, 0.0000] |
| 8 | uniform minus kcenter | memory | 0.0000 | [0.0000, 0.0000] |

## Same-budget one-frame coverage contrasts

| Pairs | Method minus baseline | Action subset | Video macro difference | 95% video-bootstrap interval |
| ---: | --- | --- | ---: | --- |
| 2 | kcenter minus uniform | all | 0.0000 | [0.0000, 0.0000] |
| 2 | kcenter minus uniform | short | 0.0000 | [0.0000, 0.0000] |
| 2 | random minus kcenter | all | 0.0000 | [0.0000, 0.0000] |
| 2 | random minus uniform | all | 0.0000 | [0.0000, 0.0000] |
| 2 | random minus kcenter | short | 0.0000 | [0.0000, 0.0000] |
| 2 | random minus uniform | short | 0.0000 | [0.0000, 0.0000] |
| 2 | recent minus kcenter | all | -0.0761 | [-0.1200, -0.0322] |
| 2 | recent minus uniform | all | -0.0761 | [-0.1200, -0.0322] |
| 2 | recent minus kcenter | short | -0.1190 | [-0.2262, -0.0357] |
| 2 | recent minus uniform | short | -0.1190 | [-0.2262, -0.0357] |
| 2 | spectral_balanced minus kcenter | all | 0.0000 | [0.0000, 0.0000] |
| 2 | spectral_balanced minus uniform | all | 0.0000 | [0.0000, 0.0000] |
| 2 | spectral_balanced minus kcenter | short | 0.0000 | [0.0000, 0.0000] |
| 2 | spectral_balanced minus uniform | short | 0.0000 | [0.0000, 0.0000] |
| 2 | spectral_power minus kcenter | all | 0.0000 | [0.0000, 0.0000] |
| 2 | spectral_power minus uniform | all | 0.0000 | [0.0000, 0.0000] |
| 2 | spectral_power minus kcenter | short | 0.0000 | [0.0000, 0.0000] |
| 2 | spectral_power minus uniform | short | 0.0000 | [0.0000, 0.0000] |
| 2 | temporal_difference minus kcenter | all | 0.0000 | [0.0000, 0.0000] |
| 2 | temporal_difference minus uniform | all | 0.0000 | [0.0000, 0.0000] |
| 2 | temporal_difference minus kcenter | short | 0.0000 | [0.0000, 0.0000] |
| 2 | temporal_difference minus uniform | short | 0.0000 | [0.0000, 0.0000] |
| 2 | temporal_reconstruction minus kcenter | all | 0.0000 | [0.0000, 0.0000] |
| 2 | temporal_reconstruction minus uniform | all | 0.0000 | [0.0000, 0.0000] |
| 2 | temporal_reconstruction minus kcenter | short | 0.0000 | [0.0000, 0.0000] |
| 2 | temporal_reconstruction minus uniform | short | 0.0000 | [0.0000, 0.0000] |
| 2 | uniform minus kcenter | all | 0.0000 | [0.0000, 0.0000] |
| 2 | uniform minus kcenter | short | 0.0000 | [0.0000, 0.0000] |
| 4 | kcenter minus uniform | all | -0.0109 | [-0.0843, 0.0565] |
| 4 | kcenter minus uniform | short | 0.0357 | [0.0000, 0.1071] |
| 4 | random minus kcenter | all | -0.0018 | [-0.1107, 0.1036] |
| 4 | random minus uniform | all | -0.0126 | [-0.0692, 0.0469] |
| 4 | random minus kcenter | short | 0.0833 | [-0.0714, 0.2619] |
| 4 | random minus uniform | short | 0.1190 | [0.0000, 0.2619] |
| 4 | recent minus kcenter | all | -0.1950 | [-0.2857, -0.1040] |
| 4 | recent minus uniform | all | -0.2059 | [-0.2551, -0.1389] |
| 4 | recent minus kcenter | short | -0.1071 | [-0.2500, 0.0000] |
| 4 | recent minus uniform | short | -0.0714 | [-0.1429, 0.0000] |
| 4 | spectral_balanced minus kcenter | all | -0.1250 | [-0.2058, -0.0347] |
| 4 | spectral_balanced minus uniform | all | -0.1358 | [-0.1827, -0.0858] |
| 4 | spectral_balanced minus kcenter | short | -0.0357 | [-0.1071, 0.0000] |
| 4 | spectral_balanced minus uniform | short | 0.0000 | [0.0000, 0.0000] |
| 4 | spectral_power minus kcenter | all | -0.1185 | [-0.1679, -0.0448] |
| 4 | spectral_power minus uniform | all | -0.1293 | [-0.1673, -0.0842] |
| 4 | spectral_power minus kcenter | short | 0.1190 | [0.0000, 0.2619] |
| 4 | spectral_power minus uniform | short | 0.1548 | [0.0357, 0.2976] |
| 4 | temporal_difference minus kcenter | all | -0.0986 | [-0.1598, -0.0237] |
| 4 | temporal_difference minus uniform | all | -0.1094 | [-0.1744, -0.0451] |
| 4 | temporal_difference minus kcenter | short | 0.0000 | [0.0000, 0.0000] |
| 4 | temporal_difference minus uniform | short | 0.0357 | [0.0000, 0.1071] |
| 4 | temporal_reconstruction minus kcenter | all | 0.0074 | [-0.0630, 0.0792] |
| 4 | temporal_reconstruction minus uniform | all | -0.0035 | [-0.0521, 0.0521] |
| 4 | temporal_reconstruction minus kcenter | short | -0.0357 | [-0.1071, 0.0000] |
| 4 | temporal_reconstruction minus uniform | short | 0.0000 | [0.0000, 0.0000] |
| 4 | uniform minus kcenter | all | 0.0109 | [-0.0565, 0.0843] |
| 4 | uniform minus kcenter | short | -0.0357 | [-0.1071, 0.0000] |
| 8 | kcenter minus uniform | all | -0.0663 | [-0.1613, 0.0390] |
| 8 | kcenter minus uniform | short | 0.0714 | [-0.0714, 0.2500] |
| 8 | random minus kcenter | all | -0.0496 | [-0.1661, 0.0660] |
| 8 | random minus uniform | all | -0.1159 | [-0.2306, 0.0051] |
| 8 | random minus kcenter | short | 0.0357 | [0.0000, 0.1071] |
| 8 | random minus uniform | short | 0.1071 | [0.0000, 0.2500] |
| 8 | recent minus kcenter | all | -0.4233 | [-0.5618, -0.2850] |
| 8 | recent minus uniform | all | -0.4896 | [-0.5991, -0.3839] |
| 8 | recent minus kcenter | short | -0.2381 | [-0.6071, 0.1429] |
| 8 | recent minus uniform | short | -0.1667 | [-0.5238, 0.1905] |
| 8 | spectral_balanced minus kcenter | all | -0.0285 | [-0.1973, 0.1169] |
| 8 | spectral_balanced minus uniform | all | -0.0949 | [-0.1937, -0.0063] |
| 8 | spectral_balanced minus kcenter | short | 0.0952 | [-0.1786, 0.3690] |
| 8 | spectral_balanced minus uniform | short | 0.1667 | [0.0000, 0.3810] |
| 8 | spectral_power minus kcenter | all | -0.2438 | [-0.4260, -0.0758] |
| 8 | spectral_power minus uniform | all | -0.3101 | [-0.4875, -0.1533] |
| 8 | spectral_power minus kcenter | short | -0.0119 | [-0.4643, 0.4524] |
| 8 | spectral_power minus uniform | short | 0.0595 | [-0.3690, 0.4881] |
| 8 | temporal_difference minus kcenter | all | -0.2026 | [-0.3271, -0.0877] |
| 8 | temporal_difference minus uniform | all | -0.2690 | [-0.3951, -0.1506] |
| 8 | temporal_difference minus kcenter | short | -0.0714 | [-0.2143, 0.0000] |
| 8 | temporal_difference minus uniform | short | 0.0000 | [-0.1071, 0.1071] |
| 8 | temporal_reconstruction minus kcenter | all | 0.0257 | [-0.1332, 0.1656] |
| 8 | temporal_reconstruction minus uniform | all | -0.0406 | [-0.1131, 0.0417] |
| 8 | temporal_reconstruction minus kcenter | short | 0.0595 | [-0.1071, 0.2857] |
| 8 | temporal_reconstruction minus uniform | short | 0.1310 | [-0.0714, 0.3571] |
| 8 | uniform minus kcenter | all | 0.0663 | [-0.0390, 0.1613] |
| 8 | uniform minus kcenter | short | -0.0714 | [-0.2500, 0.0714] |

## Limits

- All analyses are development exploration with no multiple-testing correction or independent-confirmation claim.
- Paired bootstrap uses videos, 10000 resamples and seed 17; participant identity is unknown.
- Overlapping official action labels count as separate instances, not independent temporal repetitions.
- Temporal frame coverage does not establish readability, action recognition or QA benefit.
- No-memory and full/last references are summarized, but different actual budgets are excluded from matched-budget comparisons.
