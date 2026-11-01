# Independent endpoint-control audit

All 493 independent CPU checks passed; source artifacts were not modified.

- Recomputed directly from 36 NPZ files: visual 20/32, static 8/8, mixed 12/24.
- Static readability gate: True; interpretation: interference_or_endpoint_readout_candidate_only.
- Predicted-word agreement across option orders: 12/16 endpoint pairs.
- Mixed first/last question pairs with identical predicted word: 12/12; this does not imply identical logits.
- Exact option-logit ties: 1. NumPy argmax selects option A on a tie; the choice is not positive evidence.
- All four no-video priors have null gold/correctness/margin; no no-video accuracy is calculated.
- All 32 visual cases share the checked input budget: 618 total tokens, 480 visual tokens, grid [8, 12, 20].
- Protocol/source/snapshot/artifact/NPZ/full-logit/token hashes and recorded scores match.
- Reviewed source images independently: first image men; final image enm.

## Recomputed condition scores

| Sequence | Endpoint | Option order | Gold | Prediction | Correct | Margin |
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

## Mixed first/last pairs

Gold margin compares against different gold words at the two endpoints. The men-minus-enm columns keep the semantic direction fixed.

| Sequence | Option order | First prediction | Last prediction | First gold margin | Last gold margin | First men-enm | Last men-enm | Last-first men-enm | Tie |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A8B8 | men_enm | men | men | 4.000000 | -4.125000 | 4.000000 | 4.125000 | 0.125000 | none |
| A8B8 | enm_men | men | men | 0.125000 | -1.125000 | 0.125000 | 1.125000 | 1.000000 | none |
| B8A8 | men_enm | men | men | -0.875000 | 2.625000 | 0.875000 | 2.625000 | 1.750000 | none |
| B8A8 | enm_men | enm | enm | 1.750000 | 0.000000 | -1.750000 | 0.000000 | 1.750000 | last |
| A2B14 | men_enm | men | men | 1.625000 | -2.750000 | 1.625000 | 2.750000 | 1.125000 | none |
| A2B14 | enm_men | enm | enm | -1.375000 | 0.250000 | -1.375000 | -0.250000 | 1.125000 | none |
| B14A2 | men_enm | enm | enm | 2.375000 | -0.750000 | -2.375000 | -0.750000 | 1.625000 | none |
| B14A2 | enm_men | enm | enm | 3.250000 | -1.875000 | -3.250000 | -1.875000 | 1.375000 | none |
| B2A14 | men_enm | men | men | -4.375000 | 5.375000 | 4.375000 | 5.375000 | 1.000000 | none |
| B2A14 | enm_men | men | men | -0.375000 | 2.125000 | 0.375000 | 2.125000 | 1.750000 | none |
| A14B2 | men_enm | men | men | 6.125000 | -5.875000 | 6.125000 | 5.875000 | -0.250000 | none |
| A14B2 | enm_men | men | men | 3.125000 | -3.500000 | 3.125000 | 3.500000 | 0.375000 | none |

## Text-only priors

| Endpoint | Option order | Prediction | Option A logit | Option B logit | P(men), two-option normalized |
| --- | --- | --- | --- | --- | --- |
| first | men_enm | enm | 40.000000 | 42.500000 | 0.075858 |
| first | enm_men | men | 36.000000 | 47.250000 | 0.999987 |
| last | men_enm | enm | 39.000000 | 43.000000 | 0.017986 |
| last | enm_men | men | 35.750000 | 44.750000 | 0.999877 |

This audit establishes stored-result integrity and agreement with the fixed design. It does not establish long-term forgetting or a general mechanism.
Processed visual tensors are not re-created here; their recorded hashes are checked for consistency across identical sequence inputs.
