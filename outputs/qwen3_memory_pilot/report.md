# Small-model MCQA pilot

Status: `complete`

Descriptive diagnostics only; each question's conditions are paired observations.

| Condition | Questions | Correct | Accuracy | Parse failures |
| --- | ---: | ---: | ---: | ---: |
| no_memory | 1 | 1 | 1.000 | 0 |
| first_half | 1 | 1 | 1.000 | 0 |
| last_half | 1 | 1 | 1.000 | 0 |
| uniform_full | 1 | 1 | 1.000 | 0 |

## Model and outputs

Model: `Qwen/Qwen3-VL-2B-Instruct`; revision: `89644892e4d85e24eaac8bacfd4f463576704203`.

Complete prompts, payload hashes and generation metadata are retained in report.json and records.jsonl.

| Query | Condition | Raw answer | Parse | Reward | Input/output tokens | Seconds | Peak bytes |
| --- | --- | --- | --- | ---: | --- | ---: | ---: |
| perception_test:video_1580:mc:1 | no_memory | "C" | valid | 1 | 95/2 | 0.6658266999984335 | 4283119104 |
| perception_test:video_1580:mc:1 | first_half | "C" | valid | 1 | 642/2 | 0.3314453000020876 | 4456241664 |
| perception_test:video_1580:mc:1 | last_half | "C" | valid | 1 | 647/2 | 0.23339060000216705 | 4455686144 |
| perception_test:video_1580:mc:1 | uniform_full | "C" | valid | 1 | 645/2 | 0.2382767999988573 | 4455401984 |

## Limitations

- Descriptive small-model mechanism pilot; repeated conditions are not independent samples.
- Equal frame count does not prove equal visual token count or information coverage.
- No-memory has zero visual budget; differences are coverage diagnostics, not fair method comparisons.
- Official QA answers score real generations; they are never injected as memory-utility labels.
- Utility is measured against empty memory for one frozen readout, not lifelong storage quality.
- No probe is trained and no hidden-state path is fabricated by this command.
- Payload descriptors bind frame bytes but do not measure total retained-memory storage cost.
