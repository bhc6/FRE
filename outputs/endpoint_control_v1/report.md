# Constructed endpoint controls

All video QA: 20/32; static controls: 8/8; mixed: 12/24.
Static readability gate passed: True.
Interpretation: interference_or_endpoint_readout_candidate_only. No long-term forgetting claim.

All 32 video conditions are reported. These are repeated controls on one source example.

| Sequence | Endpoint | Option order | Gold word | Predicted word | Correct | Gold margin |
| --- | --- | --- | --- | --- | --- | --- |
| A16 | first | men_enm | men | men | 1 | 7.875000 |
| A16 | first | enm_men | men | men | 1 | 4.875000 |
| A16 | last | men_enm | men | men | 1 | 7.375000 |
| A16 | last | enm_men | men | men | 1 | 5.250000 |
| B16 | first | men_enm | enm | enm | 1 | 3.375000 |
| B16 | first | enm_men | enm | enm | 1 | 3.625000 |
| B16 | last | men_enm | enm | enm | 1 | 2.375000 |
| B16 | last | enm_men | enm | enm | 1 | 2.875000 |
| A8B8 | first | men_enm | men | men | 1 | 4.000000 |
| A8B8 | first | enm_men | men | men | 1 | 0.125000 |
| A8B8 | last | men_enm | enm | men | 0 | -4.125000 |
| A8B8 | last | enm_men | enm | men | 0 | -1.125000 |
| B8A8 | first | men_enm | enm | men | 0 | -0.875000 |
| B8A8 | first | enm_men | enm | enm | 1 | 1.750000 |
| B8A8 | last | men_enm | men | men | 1 | 2.625000 |
| B8A8 | last | enm_men | men | enm | 0 | 0.000000 |
| A2B14 | first | men_enm | men | men | 1 | 1.625000 |
| A2B14 | first | enm_men | men | enm | 0 | -1.375000 |
| A2B14 | last | men_enm | enm | men | 0 | -2.750000 |
| A2B14 | last | enm_men | enm | enm | 1 | 0.250000 |
| B14A2 | first | men_enm | enm | enm | 1 | 2.375000 |
| B14A2 | first | enm_men | enm | enm | 1 | 3.250000 |
| B14A2 | last | men_enm | men | enm | 0 | -0.750000 |
| B14A2 | last | enm_men | men | enm | 0 | -1.875000 |
| B2A14 | first | men_enm | enm | men | 0 | -4.375000 |
| B2A14 | first | enm_men | enm | men | 0 | -0.375000 |
| B2A14 | last | men_enm | men | men | 1 | 5.375000 |
| B2A14 | last | enm_men | men | men | 1 | 2.125000 |
| A14B2 | first | men_enm | men | men | 1 | 6.125000 |
| A14B2 | first | enm_men | men | men | 1 | 3.125000 |
| A14B2 | last | men_enm | enm | men | 0 | -5.875000 |
| A14B2 | last | enm_men | enm | men | 0 | -3.500000 |

No-video prompt priors have no ground truth or accuracy.

| Endpoint | Option order | Predicted word | Option A logit | Option B logit |
| --- | --- | --- | --- | --- |
| first | men_enm | enm | 40.000000 | 42.500000 |
| first | enm_men | men | 36.000000 | 47.250000 |
| last | men_enm | enm | 39.000000 | 43.000000 |
| last | enm_men | men | 35.750000 | 44.750000 |
