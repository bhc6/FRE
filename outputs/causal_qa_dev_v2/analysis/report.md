# V2 固定输入视觉分支干预：CPU审计

36题×15条件=540条完整核验；128帧输入哈希、官方题目、选项、金标、margin与冻结源码/协议一致。
3个结构/identity负对照共108次全词表逐元素比较全部相同。

预声明主要条件 middle13_high_up：全部 25/36（baseline 25/36）；时序 6/8（baseline 6/8）。
开发门槛：**未通过**。不以其它最好条件替换主要候选；margin改变不等于解决问答错误。

## 全条件结果

| 条件 | 全部正确 | 视频宏平均 | 时序正确 | 时序宏平均 | 时序净纠正 | 平均margin差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 25/36 | 0.7250 | 6/8 | 0.8333 | +0 | +0.0000 |
| late27_zero | 25/36 | 0.7250 | 6/8 | 0.8333 | +0 | +0.0000 |
| late27_temporal_up | 25/36 | 0.7250 | 6/8 | 0.8333 | +0 | +0.0000 |
| middle13_identity | 25/36 | 0.7250 | 6/8 | 0.8333 | +0 | +0.0000 |
| middle13_zero | 26/36 | 0.7500 | 6/8 | 0.8333 | +0 | +0.0000 |
| middle13_dc_down | 25/36 | 0.7250 | 6/8 | 0.8333 | +0 | -0.0104 |
| middle13_temporal_down | 25/36 | 0.7250 | 6/8 | 0.8333 | +0 | +0.0382 |
| middle13_temporal_up | 25/36 | 0.7250 | 6/8 | 0.8333 | +0 | +0.0035 |
| middle13_high_down | 25/36 | 0.7250 | 6/8 | 0.8333 | +0 | +0.0139 |
| middle13_high_up | 25/36 | 0.7250 | 6/8 | 0.8333 | +0 | -0.0104 |
| middle13_low_up | 25/36 | 0.7250 | 6/8 | 0.8333 | +0 | +0.0104 |
| middle13_temporal_matched_up | 25/36 | 0.7250 | 6/8 | 0.8333 | +0 | -0.0347 |
| middle13_phase_control_up | 25/36 | 0.7250 | 6/8 | 0.8333 | +0 | +0.0000 |
| middle13_attention_matched_up | 25/36 | 0.7250 | 6/8 | 0.8333 | +0 | +0.0035 |
| early0_high_up | 25/36 | 0.7250 | 6/8 | 0.8333 | +0 | +0.0486 |

## 预声明开发门槛

| 门槛 | 结果 |
| --- | --- |
| all_structural_controls_exact | 通过 |
| temporal_net_corrections_at_least_two | 未通过 |
| temporal_positive_net_videos_at_least_two | 未通过 |
| all_questions_no_net_degradation | 通过 |
| primary_beats_norm_matched_temporal_accuracy | 未通过 |

两视频门槛按两个视频各自时序题净纠正为正解释，JSON保留全部纠正/损伤视频。

## 主要条件对等扰动范数控制

| 控制 | 全部净纠正 | 时序净纠正 | 时序宏平均差 | 时序margin差 | 实际范数比中位数[范围] |
| --- | ---: | ---: | ---: | ---: | --- |
| middle13_temporal_matched_up | +0 | +0 | +0.0000 | +0.0156 | 0.999980 [0.999935, 1.000074] |
| middle13_phase_control_up | +0 | +0 | +0.0000 | -0.0156 | 0.999977 [0.999865, 1.000004] |
| middle13_attention_matched_up | +0 | +0 | +0.0000 | -0.0781 | 0.999851 [0.999742, 0.999952] |

范数比=控制实际BF16扰动/主要条件实际扰动。请求范数相等已核验，实际范数不逐bit相等。循环位移控制保留高频子空间，控制时间方向，不是独立随机频带。

## 全部预测改变

| 条件 | query_id | 原→新答案 | 金标 | 正确性 | margin变化 |
| --- | --- | --- | --- | --- | ---: |
| middle13_zero | perception_test:video_3501:mc:3 | A→C | C | 0→1 | +0.625 |

## 独立证据链缺项

- Only 8 reused development videos; no independent confirmation or cross-person inference.
- No condition tests bounded stored-memory selection/compression or question-grounded evidence removal.
- Action-coverage overlap does not provide a causal QA evidence label.
- Pixel-order edits and legally relabeled counterfactual QA are not part of this run.
- BF16 actual perturbation norms differ despite matched requested norms; ratios are reported.
- Nonstructural full-vocabulary arrays were not saved; their hashes cannot be independently recomputed.
- No repeated baseline estimates runtime margin variability; exact controls verify implementation, not robust task benefit.
- No second model; best nonprimary conditions remain exploratory.
- Temporal filtering uses the complete observed prefix, not an online causal memory update.

仅为开发探索，不作显著性、泛化、长期记忆或频谱记忆成功推断；v1负结果不变。

## 来源

- Run: D:\FRE\outputs\causal_qa_dev_v2
- config_sha256: efbad34be0aed278fd9b59bd939a7006f93f428690f0831c0eacf30151d3d397
- records_sha256: bb288f7755ee3c86f19fef781df0a45da9836f4074da8373ab14f528952a2c92
- questions_sha256: 24d9ae4b93a8bae34fc7f586047862f96a4bb593a630ab655be1ff0c6b4815eb
- protocol_sha256: 48a58eccae0dd3b5e19f63962813cbb7bad8484aaf6f3e7956ede18f2cb637b8
- source_sha256: c0b0afbdce8d1a95e393e5684d9922ea08a05f0edf17499c78debe42648d0dd5
- analysis_source_sha256: d8084748cbcbebaa6d30c937b71285bafb775c122bbe313712df9a8f44727db0
