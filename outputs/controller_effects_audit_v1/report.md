# Independent complete-test intervention effects audit

All results below are independently recomputed from full-vocabulary NPZ archives. All four test shards and all 106 × 13 rows are present. No test subset, seed, or method was selected by performance.

Primary spectral − time video-macro difference, averaged across three fixed seeds: 0.0000 percentage points. Corresponding micro difference: 0.0000 percentage points.

## Real questions

| Method | Seed | Correct / 74 | Video macro | Full logits changed | Options changed | Repairs | Regressions | Gold-margin mean change | Median change | Max head ratio | Varying gain coordinates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | single | 32/74 | 40.521% | 0/74 | 0/74 | 0 | 0 | 0 | 0 | 0 | n/a |
| time_mlp | 20261101 | 31/74 | 39.271% | 74/74 | 10/74 | 0 | 1 | -0.00844595 | 0 | 0.0190005 | 128 |
| time_mlp | 20261102 | 31/74 | 39.271% | 74/74 | 11/74 | 0 | 1 | -0.00844595 | 0 | 0.0172142 | 128 |
| time_mlp | 20261103 | 31/74 | 39.271% | 74/74 | 9/74 | 0 | 1 | -0.00337838 | 0 | 0.016038 | 128 |
| spectral_mlp | 20261101 | 31/74 | 39.271% | 74/74 | 10/74 | 0 | 1 | -0.00337838 | 0 | 0.0199433 | 128 |
| spectral_mlp | 20261102 | 31/74 | 39.271% | 74/74 | 10/74 | 0 | 1 | -0.00506757 | 0 | 0.0173997 | 128 |
| spectral_mlp | 20261103 | 31/74 | 39.271% | 74/74 | 12/74 | 0 | 1 | -0.00337838 | 0 | 0.0187738 | 128 |
| spectral_phaseblind | 20261101 | 31/74 | 39.271% | 74/74 | 11/74 | 0 | 1 | 0 | 0 | 0.0196333 | 128 |
| spectral_phaseblind | 20261102 | 31/74 | 39.271% | 74/74 | 12/74 | 0 | 1 | -0.00675676 | 0 | 0.0174701 | 128 |
| spectral_phaseblind | 20261103 | 32/74 | 40.521% | 74/74 | 13/74 | 0 | 0 | -0.00675676 | 0 | 0.0188117 | 128 |
| constant_spectral | 20261101 | 32/74 | 40.521% | 74/74 | 3/74 | 0 | 0 | 0 | 0 | 0.00278988 | 0 |
| constant_spectral | 20261102 | 31/74 | 39.271% | 74/74 | 4/74 | 0 | 1 | -0.00168919 | 0 | 0.00249386 | 0 |
| constant_spectral | 20261103 | 31/74 | 39.271% | 74/74 | 3/74 | 0 | 1 | -0.00337838 | 0 | 0.00265034 | 0 |

## All no-gold static conditions

| Method | Seed | Condition | Cases | KL mean | Prediction flips | Full logits changed | Options changed | Max head ratio |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | single | static_first | 16 | 0 | 0/16 | 0/16 | 0/16 | 0 |
| identity | single | static_last | 16 | 0 | 0/16 | 0/16 | 0/16 | 0 |
| identity | single | static_combined | 32 | 0 | 0/32 | 0/32 | 0/32 | 0 |
| time_mlp | 20261101 | static_first | 16 | 0.000266666 | 0/16 | 16/16 | 4/16 | 0.0140562 |
| time_mlp | 20261101 | static_last | 16 | 0.000209145 | 0/16 | 16/16 | 6/16 | 0.0123857 |
| time_mlp | 20261101 | static_combined | 32 | 0.000237905 | 0/32 | 32/32 | 10/32 | 0.0140562 |
| time_mlp | 20261102 | static_first | 16 | 0.000193016 | 0/16 | 16/16 | 4/16 | 0.0109663 |
| time_mlp | 20261102 | static_last | 16 | 0.000212817 | 0/16 | 16/16 | 6/16 | 0.011221 |
| time_mlp | 20261102 | static_combined | 32 | 0.000202916 | 0/32 | 32/32 | 10/32 | 0.011221 |
| time_mlp | 20261103 | static_first | 16 | 0.000125059 | 0/16 | 16/16 | 3/16 | 0.0110579 |
| time_mlp | 20261103 | static_last | 16 | 0.00013523 | 0/16 | 16/16 | 3/16 | 0.0103218 |
| time_mlp | 20261103 | static_combined | 32 | 0.000130144 | 0/32 | 32/32 | 6/32 | 0.0110579 |
| spectral_mlp | 20261101 | static_first | 16 | 0.000271322 | 0/16 | 16/16 | 4/16 | 0.0156729 |
| spectral_mlp | 20261101 | static_last | 16 | 0.00019703 | 0/16 | 16/16 | 8/16 | 0.0133212 |
| spectral_mlp | 20261101 | static_combined | 32 | 0.000234176 | 0/32 | 32/32 | 12/32 | 0.0156729 |
| spectral_mlp | 20261102 | static_first | 16 | 0.000271902 | 0/16 | 16/16 | 4/16 | 0.012207 |
| spectral_mlp | 20261102 | static_last | 16 | 0.000212766 | 0/16 | 16/16 | 5/16 | 0.011323 |
| spectral_mlp | 20261102 | static_combined | 32 | 0.000242334 | 0/32 | 32/32 | 9/32 | 0.012207 |
| spectral_mlp | 20261103 | static_first | 16 | 0.000204446 | 0/16 | 16/16 | 4/16 | 0.0124012 |
| spectral_mlp | 20261103 | static_last | 16 | 9.77803e-05 | 0/16 | 16/16 | 6/16 | 0.0114627 |
| spectral_mlp | 20261103 | static_combined | 32 | 0.000151113 | 0/32 | 32/32 | 10/32 | 0.0124012 |
| spectral_phaseblind | 20261101 | static_first | 16 | 0.000271322 | 0/16 | 16/16 | 4/16 | 0.0154941 |
| spectral_phaseblind | 20261101 | static_last | 16 | 0.000196722 | 0/16 | 16/16 | 6/16 | 0.0132513 |
| spectral_phaseblind | 20261101 | static_combined | 32 | 0.000234022 | 0/32 | 32/32 | 10/32 | 0.0154941 |
| spectral_phaseblind | 20261102 | static_first | 16 | 0.000272403 | 0/16 | 16/16 | 5/16 | 0.0122364 |
| spectral_phaseblind | 20261102 | static_last | 16 | 0.000213812 | 0/16 | 16/16 | 6/16 | 0.011335 |
| spectral_phaseblind | 20261102 | static_combined | 32 | 0.000243107 | 0/32 | 32/32 | 11/32 | 0.0122364 |
| spectral_phaseblind | 20261103 | static_first | 16 | 0.000204446 | 0/16 | 16/16 | 4/16 | 0.0124199 |
| spectral_phaseblind | 20261103 | static_last | 16 | 9.77803e-05 | 0/16 | 16/16 | 6/16 | 0.0114646 |
| spectral_phaseblind | 20261103 | static_combined | 32 | 0.000151113 | 0/32 | 32/32 | 10/32 | 0.0124199 |
| constant_spectral | 20261101 | static_first | 16 | 0.000203946 | 0/16 | 16/16 | 3/16 | 0.00228678 |
| constant_spectral | 20261101 | static_last | 16 | 9.68297e-05 | 0/16 | 16/16 | 2/16 | 0.00189125 |
| constant_spectral | 20261101 | static_combined | 32 | 0.000150388 | 0/32 | 32/32 | 5/32 | 0.00228678 |
| constant_spectral | 20261102 | static_first | 16 | 0.000174945 | 0/16 | 15/16 | 3/16 | 0.00204503 |
| constant_spectral | 20261102 | static_last | 16 | 9.68297e-05 | 0/16 | 16/16 | 2/16 | 0.00168945 |
| constant_spectral | 20261102 | static_combined | 32 | 0.000135887 | 0/32 | 31/32 | 5/32 | 0.00204503 |
| constant_spectral | 20261103 | static_first | 16 | 0.000203946 | 0/16 | 16/16 | 3/16 | 0.00216888 |
| constant_spectral | 20261103 | static_last | 16 | 9.68297e-05 | 0/16 | 16/16 | 2/16 | 0.00179582 |
| constant_spectral | 20261103 | static_combined | 32 | 0.000150388 | 0/32 | 32/32 | 5/32 | 0.00216888 |

Static conditions have no semantic gold and no accuracy. Gold-margin mean/median changes are question-weighted; the machine table also supplies the video-macro mean change. Gain variation alone does not establish useful learned structure. Head ratios are the recorded FP32 controller outputs before BF16 projection conversion.

The audit also verifies raw archived input-ID bytes, dtype, and shape against the paired input-tensor hashes, and checks that all four resource summaries completed without resource-stop markers. Full case effects, all method/seed/condition summaries, source hashes, and original archive hashes are retained.
