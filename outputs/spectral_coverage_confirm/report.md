# Action-time evidence coverage

Declared: 2026-11-01T09:49:42Z; primary budget: 4 pairs; short duration: <= 2.0 seconds.

Videos: 40; official action instances: 246; videos with empty official action lists: 8.

Primary outcome: video-macro probability of retaining at least one source frame, conditional on a short event being observable in the original 64 frames.

| Method | Macro hit rate | Eligible videos | Retained / observable events |
| --- | ---: | ---: | ---: |
| kcenter | 0.3244 | 30 | 31 / 144 |
| random | 0.2328 | 30 | 26 / 144 |
| recent | 0.1079 | 30 | 13 / 144 |
| spectral_balanced | 0.2834 | 30 | 28 / 144 |
| spectral_power | 0.2629 | 30 | 27 / 144 |
| temporal_centered | 0.2288 | 30 | 26 / 144 |
| temporal_difference | 0.2299 | 30 | 23 / 144 |
| temporal_reconstruction | 0.2981 | 30 | 28 / 144 |
| uniform | 0.2967 | 30 | 25 / 144 |

## Predeclared main contrasts

- spectral_balanced minus kcenter: -0.0410; 95% paired-video percentile interval [-0.1446, 0.0539].
- spectral_balanced minus temporal_centered: 0.0545; 95% paired-video percentile interval [-0.0180, 0.1488].

## Observation limitations

- all: 246 actions, 4 with no observed source frame, 242 observable with at least one frame.
- short: 148 actions, 4 with no observed source frame, 144 observable with at least one frame.

- Temporal overlap is a coverage proxy, not evidence of visibility, recognition or QA utility.
- Zero-observation events are reported separately and cannot be recovered by selecting observed frames.
- Video bootstrap does not establish participant independence; intervals are unadjusted.
- Official empty action lists are reported without treating their videos as action-free.
