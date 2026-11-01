# 小模型频谱记忆：2026-11-01 实验结果与决策

## 当前结论

主确认实验没有支持“频谱记忆解决了短暂证据丢失或状态问答错误”。在冻结的 40 个新视频上，主预算为 8 帧，Qwen3-VL-2B 的时序/状态问答视频宏平均正确率：频谱方法 46.72%，k-center 54.80%；差值 −8.08 个百分点，配对视频 bootstrap 95% 区间 [−19.70, +2.02]。区间跨零，不能断言主指标上存在确定劣势，更不能声称优于基线。

这里完成的是短视频事件结束后的有限外部记忆实验。尚未建立长期流式记忆有效性，也没有达到论文所需的积极方法证据。代码可运行、数学目标明确以及问题确实存在，都不能替代方法解决问题的实测结果。

## 1. 一句话研究问题

**在小型 VLM 的固定记忆预算下，保留隐藏轨迹的多尺度时间变化，能否减少短暂事件证据遗失，并提高事后问答正确率？**

明确问题是：原始观察中已经存在的短动作证据，在压缩为少量记忆帧后可能消失。原 8 个开发视频中可以定位均匀采样漏掉拍手事件的案例；新 40 视频的动作标注则提供批量覆盖度量。这不是本项目首次发现的现象：ICCV 2025 LSDBench/RHS 已讨论低密度漏掉短动作及高密度冗余，CVPR 2025 AKS 也研究固定 token 预算的自适应采样。正式来源及作者实现状态见 [文献矩阵](related_work_matrix.md)。

“多帧上下文导致末态读出失败”是另一个观察，但固定尾部的等预算实验没有证实其原因就是重复旧状态。不能以这一未确认机制论证频谱方法。

## 2. 已实现方法与对照

spectral_memory/memory_selection.py 实现查询不可见的帧对选择。观察 64 帧，以相邻两帧对应一个隐藏时间点；选出的真实 RGB 帧对按时间顺序持久化。目标为三个频带的相对复频谱重建误差均值，排除 DC、保留相位，使用固定能量下限。从首尾帧对开始，贪心加入锚点。

区别来自频带权重和相位处理。傅里叶变换没有增加信息；该目标等价于依赖输入轨迹的有色时域二次损失。[方法说明](spectral_memory_method.md) 给出推导、Gram 等价约化、边界条件与局限。

同预算对照包括 uniform、recent、random(seed17)、temporal_difference、kcenter、原时域重建、去均值时域重建、去相位的幅值重建。普通方法使用相同原始观察、画布、记忆格式和读出模型。除 recent 外都保留首尾，因此 4 帧预算下多数方法相同，这一退化预算不构成频谱证据。

没有复现第三方 AKS、RHS、InfiniPot-V、FlashCache、FreqKV 的完整算法。“作者 README 可访问”不等于本地基线已跑通。当前 k-center 等是明确描述的内部对照。

## 3. 数据与预声明

| 项目 | 实际设置 |
| --- | --- |
| 开发 | 8 视频、36 QA |
| 独立确认 | 官方 Perception Test validation：40 视频、182 QA |
| 分层选择 | 有 memory 问题的视频 20 个，无此类问题的视频 20 个 |
| 主要题型 | 预声明 temporal_state：77 QA、33 视频 |
| 官方 memory 次要题型 | 32 QA、20 视频 |
| 方法 | 9 个同预算选择方法；全 64 帧和无视频参考 |
| 主预算 | 4 帧对 = 8 帧，NPZ 3,539,820 bytes |
| 次要预算 | 4 帧 1,770,300 bytes；16 帧 7,078,860 bytes |
| 主模型 | Qwen3-VL-2B-Instruct，固定 revision，BF16 |
| 主观察层 | 层 0；没有按确认成绩挑层 |
| 解码 | 合法单 token 选项 logits argmax |
| 主比较 | spectral_balanced − kcenter |
| 机制比较 | spectral_balanced − temporal_centered |
| 通过要求 | 主差值 ≥3pp 且 95% CI 下界 >0；机制差值 CI 下界也 >0 |

协议于 **2026-11-01 09:52:07 UTC** 冻结，先于确认 QA 输出。协议、输入与源码 SHA 保存在 outputs/spectral_memory_protocol_v1/。主模型记录 5,642 条，去除同题同载荷缓存复用后有 3,863 次真实前向。不能把复用行或同一视频的多个问题当成独立重复。

这是 memory 富集子集，不代表完整 benchmark 分布。参与者身份未知，视频级 bootstrap 不保证跨人独立性；未排除预训练污染。这 40 个视频一经查看结果，不得再作为改进方法的独立确认集。

## 4. 主模型结果

均为视频宏平均百分比；普通选择方法均保存 8 帧。

| 方法 | 全部 182 QA | 时序/状态 77 QA（主） | 官方 memory 32 QA |
| --- | ---: | ---: | ---: |
| 无视频参考 | 37.90 | 34.85 | 35.00 |
| 全 64 帧参考 | 57.20 | 45.45 | 40.00 |
| uniform | 52.79 | 50.00 | 45.00 |
| recent | 54.47 | 47.73 | 40.00 |
| random | 51.85 | 53.54 | 43.33 |
| temporal_difference | 52.20 | 44.95 | 41.67 |
| kcenter | 59.70 | 54.80 | 41.67 |
| temporal_reconstruction | 48.39 | 47.73 | 40.00 |
| temporal_centered | 49.93 | 48.99 | 43.33 |
| spectral_power | 52.53 | 45.45 | 45.00 |
| spectral_balanced | 52.63 | 46.72 | 43.33 |

全帧参考改变了记忆量和视觉 token 数，仅用于观察读出表现；它不是理论上限。完整证据并不保证小模型答对。

| 主预算差值 | 宏平均差（pp） | 95% 配对视频区间（pp） | 解释 |
| --- | ---: | --- | --- |
| 频谱 − k-center，时序/状态 | −8.08 | [−19.70, +2.02] | 主门槛未通过 |
| 频谱 − centered-time，时序/状态 | −2.27 | [−10.61, +4.55] | 独立频带价值未获支持 |
| 频谱 − k-center，全部问题 | −7.07 | [−14.88, −0.19] | 次要、未作多重校正 |
| 频谱 − centered-time，全部问题 | +2.71 | [−2.33, +7.79] | 正点估计不构成有效性证据 |

主要题型中，相对 k-center，频谱纠正 2 题、损失 4 题，其余相同；视频宏平均差较题目微平均更大，因为各视频题数不同。不能把 37/77 与 39/77 的微平均差混写成 −8.08pp。

官方 memory 的频谱 − k-center 为 +1.67pp，区间 [0.00, +5.00]，仅一题变化，不能替代失败的主比较。16 帧次要预算上，全部问题的频谱 − k-center 为 −1.29pp，区间 [−5.25, +2.50]，也没有稳定正收益。

## 5. 动作证据是否保住

完整官方验证标注提供 246 个动作，分布于 32 个视频；另 8 视频的动作列表为空，不能理解为没有发生动作。短动作预定义为 ≤2 秒，共 148 个，其中 144 个在最初 64 帧观察里至少命中一帧。观察已漏掉的 4 个动作不能由后续选择恢复。

主要覆盖指标是可观察短动作上，按视频平均的“至少留下一帧”比例。使用解码源 PTS；隐藏时间戳是目标网格中心，不能替代源帧时间。

| 8 帧方法 | 短动作视频宏覆盖率 |
| --- | ---: |
| kcenter | 32.44% |
| temporal_reconstruction | 29.81% |
| uniform | 29.67% |
| spectral_balanced | 28.34% |
| spectral_power | 26.29% |
| random | 23.28% |
| temporal_difference | 22.99% |
| temporal_centered | 22.88% |
| recent | 10.79% |

频谱 − k-center：**−4.10pp，95% CI [−14.46, +5.39]**；频谱 − centered-time：**+5.45pp，95% CI [−1.80, +14.88]**。两者均跨零。事件微平均上频谱实际保住 28/144 个可观察短动作，不能把 28.34% 的视频宏平均误写成事件微平均。

命中区间不保证动作完整可见或可被识别。动作与 QA 没有完整官方一对一映射，不能声称建立了逐题因果链。结果及独立复核见 outputs/spectral_coverage_confirm/；全部 60 个方法×预算×题型组从官方微秒区间和源 PTS 重算，最大数值差 1.11e-16。

## 6. 第二读者与进一步探索

500M 独立家族迁移评估因系统重启相关中断已停止：计划 2,002 条记录，当前仅有 1,633 条，尚未完成，不能给出完整汇总或迁移结论。该检查使用相同冻结 8 帧记忆和 182 道问题，观察隐藏表示仍来自 Qwen3-VL。原始记录保持不变；故障事实、资源计数与未确定原因见 [重启事件报告](resource_incident_20261101.md)。

开发集上另做约束交换及全枚举：固定首尾与四个锚点时，只有 435 个内部组合，可区分贪心不足与目标不对齐。它不改变冻结确认成绩。见 [后续诊断](next_iteration_diagnostics.md) 和 [后续文献定位](next_iteration_literature.md)。

## 7. 成本、完整性与复现

Qwen 主预算有 240 个视觉 token；全帧参考为 1,920。频谱和 k-center 请求延迟中位数都约 0.139 秒，没有额外读出加速。全帧参考约 1.434 秒，但读取了更多帧，不能归功于频谱目标。频谱组峰值 CUDA allocated 4,326,707,200 bytes；全帧参考 5,280,961,536 bytes，均不是整机或驱动显存占用。

40 视频三层隐藏提取合计约 44.61 秒。含共享 Gram 和完整诊断的每视频 CPU 选择均值：频谱 8.06ms、k-center 4.84ms。均匀/近期/随机无需隐藏提取及这些诊断，不能强加费用制造效率优势。三层合计也不是优化后的单层最低费用。

NPZ 包含所选 RGB、时间戳与索引，不包含外部目录、模型权重、完整观察工作区和原视频。写入仍使用完整轨迹，没有固定工作内存的持续流更新。

重跑使用新输出目录。已有固定环境为 .venv-cuda，绘图依赖声明为 .[plots]：

~~~powershell
.\.venv-cuda\Scripts\python.exe -m spectral_memory.memory_study build --frames-manifest data/perception_confirm_frames64/frames.manifest.jsonl --hidden-manifest outputs/qwen3_confirm_hidden/manifest.layer_0.jsonl --methods uniform,recent,random,temporal_difference,kcenter,temporal_reconstruction,temporal_centered,spectral_balanced,spectral_power,full_reference --budgets 2,4,8 --output-dir outputs/confirmation_rebuild
.\.venv-cuda\Scripts\python.exe -m spectral_memory.memory_study evaluate --memory-manifest outputs/confirmation_rebuild/memories.jsonl --questions data/perception_confirm/questions.jsonl --output-dir outputs/confirmation_rerun
.\.venv-cuda\Scripts\python.exe -m spectral_memory.audit_study --study outputs/spectral_memory_confirm_readout --memories outputs/spectral_memory_confirm/memories.jsonl --questions data/perception_confirm/questions.jsonl --freeze outputs/spectral_memory_protocol_v1/freeze.json --output outputs/spectral_memory_confirm_readout/integrity_audit.json
~~~

integrity_audit.json 已核对 5,642 条完整配对、NPZ 字节与内容哈希、选项 argmax、官方答案重评分、缓存一致性、源码快照及冻结输入。审计通过表示记录可信，不表示方法有效。

## 8. 投稿判断

小模型可以是顶会研究对象；当前瓶颈是贡献证据。“频域＋记忆”已有 CVPR 2026 FlashCache、ICLR 2026 FreqKV 等直接先例，使用 FFT 或固定预算本身不足以声称新颖。

可以声称：建立了可审计的固定字节多模态记忆实验，定位了短动作证据丢失，检验了相位保留的多频带相对重构方法。**不能声称：该方法解决了观察到的问题、优于强基线，或已具备顶会录用条件。**

下一次正式确认必须先有开发集的可恢复性与机制证据，再冻结新方法并使用未见视频；不能在这 40 视频上调层、调频带或挑题型后将其包装为独立验证。
