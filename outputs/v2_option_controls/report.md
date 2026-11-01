# V2字母logit校准控制（仅CPU）

此对照只校准已有三个选项分数，不修改模型，不构成记忆方法。只用142条旧训练题拟合两个自由参数；validation和旧test均作为已看开发诊断。没有读取新32源的模型分数。

固定全批Adam 200步，lr=0.03，betas=(0.9,0.999)，eps=1e-8；目标为训练平均NLL + 0.01×三个bias的平方和，bias之和为0。初始全0，不挑checkpoint。

最终A/B/C bias：`[-0.2562399472650754, 0.24295402350768303, 0.013285923757392348]`。

| split | 方法 | 正确/题数 | micro % | source macro % | NLL | source macro NLL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | native | 84/142 | 59.15 | 60.29 | 1.162837 | 1.076487 |
| train | train_only_letter_calibration | 88/142 | 61.97 | 62.82 | 1.151403 | 1.064575 |
| validation | native | 17/40 | 42.50 | 43.48 | 1.221479 | 1.218923 |
| validation | train_only_letter_calibration | 17/40 | 42.50 | 43.18 | 1.272293 | 1.268073 |
| old_test_seen | native | 32/74 | 43.24 | 40.52 | 1.214509 | 1.224203 |
| old_test_seen | train_only_letter_calibration | 31/74 | 41.89 | 44.06 | 1.248846 | 1.241399 |

| split | 修复 | 回退 | 预测改变 | source macro差（百分点） |
| --- | ---: | ---: | ---: | ---: |
| train | 5 | 1 | 9 | 2.53 |
| validation | 1 | 1 | 2 | -0.30 |
| old_test_seen | 2 | 3 | 6 | 3.54 |

选项位置偏差仍需ABC/BCA/CAB循环换位、按答案内容还原后检验。此处分数校准不能取代真实内部方法，也不能把旧test重新称为未见确认集。
