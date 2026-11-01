# 当前学习式读出控制器：已核先验工作的直接对照

日期：2026-11-01。本文只整理项目已有且已核实的论文材料，不新增检索、不扩大文献范围、不声称已穷尽相关工作，也未读取本轮训练或测试预测。正式身份与阅读深度沿用 `related_work_source_audit.json`、`internal_probe_novelty_audit.md`、`frame2freq_axis_audit.md` 和 `mechanism_related_work.md`。

## 1. 与已有矩阵的关系

`related_work_matrix.md` 的22篇主表及独立6篇FFN分区是先前“外部真实帧证据与频域记忆压缩”问题的矩阵，合计28篇近期正式工作；其计数与编号保持不变。`internal_probe_novelty_audit.md` 的9篇内部机制清单与其部分重叠，不能把两份数字简单相加；其中8篇主会、1篇ACL Findings。

当前方案已转为**监督学习的、query条件化的末层内部读出残差**：固定2B基座、实际attention/value交叉时间谱、每头有界context修正、完整原生forward评估。因而，旧矩阵中“无任务标签写入”“不训练adapter”“未来问题未知的外部存储”不再是当前方案的区别点。下面按新对象重排最接近先验，不改写旧路线的历史定义。

## 2. 最接近工作的具体重合与差异

| 已核工作与正式身份 | 已经覆盖的内容 | 本轮方案的具体差异 | 不能据此宣称什么 |
| --- | --- | --- | --- |
| **Frame2Freq: Spectral Adapters for Fine-Grained Video Understanding，CVPR 2026主会** [1] | 冻结视觉骨干、训练时域/频域adapter与分类头；沿内部时间embedding做FFT；补充A.1/Algorithm 1还以动作标签计算频带between/within方差判别性。 | 当前在decoder末层答案query的attention context处，以实际attention与V的交叉谱构造有界残差；监督为原官方三选QA，且显式对照等参数时域controller。对象、插入位置、读出形式和目标不同。 | 不能说“首次在内部隐状态做时间频谱诊断”或“首次训练视频谱适配器”。不能再把Frame2Freq概括为只有前向adapter。任务换成QA和模型变小并不自动成为新机制。 |
| **Probing Rotary Position Embeddings through Frequency Entropy，ICLR 2026** [2] | 对RoPE二维Q/K子空间范数的token轨迹做DFT/STFT/谱熵诊断，再缩放选中Q/K分量，已有谱诊断指导内部激活干预。 | 当前沿8个视觉时间group分析attention/value交叉谱，由训练得到的query条件化MLP输出gain，修改最终context而非直接修改Q/K二维幅值。 | 不能说“首次根据频谱干预内部激活”。既有论文准确率与baseline统计上不可区分，不能借它证明本轮会改善QA；标准RoPE旋转保范数，缩放也不等于减慢相位。 |
| **Map the Flow: Revealing Hidden Pathways of Information in VideoLLMs，ICLR 2026** [3] | attention knockout与logit lens研究跨帧、视频到问题、问题到末token的传播；正文主分析筛基线答对样本，用9层窗口阻断路径。 | 当前不按baseline对错筛选74道新test题，冻结末层位置，学习小的读出修正并做全样本paired comparison；不是只对成功行为做删除归因。 | 改善答案不自动定位了原错误的唯一路径；读出修正也可能补偿多个上游原因。不能说首次定位时间信息到末端问题读出。 |
| **Temporal Reasoning Transfer from Text to Video（T3），ICLR 2025** [4] | 正式附录已用视觉入口及decoder末层probe展示“内部时序信息可读，而原QA表现更低”；文本时序训练可转移到VideoQA。 | 当前不是另训练一个外部probe报告可读性，而是在原生QA计算图内插入受限残差，保持原输出头，并在独立源上比较时域/谱结构。 | “模型内部有信息但不会答”不是首次观察；probe较强和干预收益都不足以证明特定频带是原模型实际使用的必要证据。 |
| **Language-guided Frequency Modulation for Large Vision-Language Models，CVPR 2026主会** [5] | 语言引导下调制视觉频率，关注空间全局结构与局部细节，再交给语言模型。已有语言条件化的VLM频域调制。 | 当前关注视频时间group上的attention/value交叉项与最后query context，而非视觉空间频率调制。 | 不可泛称首次“语言/问题条件化频域VLM模块”。本项目对LFM证据深度为正式摘要，未核清全部训练配置，不补写其无训练、精确模块位置或等参数细节。 |
| **Causality Matters: How Temporal Information Emerges in Video Language Models，AAAI 2026** [6] | 帧间因果交互、位置/顺序对照、逐层时序信息与attention knockout。 | 当前是固定最后层上的学习式局部补偿比较，包含有界残差、真实QA与静态输入分布保持。 | 不能把因果attention形成时间信息或“位置会混入时序轨迹”写成新发现；局部收益不等于解释了全网络的时序机制。 |
| **What Do VLMs NOTICE? A Mechanistic Interpretability Pipeline for Gaussian-Noise-free Text-Image Corruption and Evaluation，NAACL 2025主会** [7] | 语义图文替换与activation patching；显示corruption选择会改变层位解释。 | 当前静态first/last只做输入构造上的baseline蒸馏与保持诊断，gold为空；真实QA和原有端点构造金标分开。 | 重复帧不是通用无混杂反事实；保持该构造上的输出不保证真实静态视觉语义无损，也不能沿用原视频金标。 |

以上是明确的机制和实验对象差异，不是首创性证明。Frame2Freq是最接近的**学习式时间谱模块**先例；Frequency Entropy是最接近的**内部谱诊断→激活干预**先例；Map the Flow和T3界定了**内部信息仍在但读出/传播失败**这一问题已被研究。

## 3. 保留为边界的工作

FreqKV（ICLR 2026）和FlashCache（CVPR 2026）已覆盖频域KV压缩、低通/残差与预算分配。当前不压缩KV、不建立持久化外部记忆、不承诺更长上下文，它们是原研究定位的边界而不是当前等计算量controller的现成复现基线。不可用“未修改KV”来证明新颖性，也不可把当前短视频QA改善叫长期记忆改善。

`Improving Causal Interventions in Amnesic Probing with Mean Projection or LEACE` 是 **ACL 2025 Findings** [8]，不能计作NAACL/ACL主会。其信息/选择性控制提醒：一般损害或修复不自动定位目标信息，需匹配非目标变化。当前时域、constant与conditioner phase消融提供有限对照，但不能等价宣称已验证所有因果选择性。

## 4. 当前能检验的主张

可直接表述为：**在冻结小型VLM的最后读出位置，比较等参数时域与交叉谱条件化残差对短视频三选QA的作用，并报告其相位条件器消融、常量增益消融及静态构造分布保持。**

若谱版本优于identity但未稳定优于等参数time版本，证据支持这一局部学习式补偿可能有用，不支持频谱结构有额外收益。若优势只出现在一个seed、已看源或事后子集，也不能作独立确认。

即便谱版本有优势，time逐位置basis与cross-spectrum对应循环lag结构，不是同一特征的纯正交换基；等参数并不等函数类。`spectral_phaseblind`仅消融条件器的聚合cross-phase，实际delta basis和query仍保留相关信息，不能据该消融声称整个网络的相位必要性。Parseval读出恒等式与功率不足反例是标准代数应用，不应作为新定理。

本轮新test仅16源、74题、14.362–34.910秒，需保留小样本和分层选源限制。尚未进行第三方论文原方法复现；不得把作者报告的成绩与本地controller成绩拼成公平横向表，也不以当前矩阵宣称SOTA或顶会录用级创新已获证明。

## 5. 已核的一手出处

以下URL均复用已有审计记录；本轮没有重新联网核查或扩展条目。

1. Frame2Freq：[CVF正式页](https://openaccess.thecvf.com/content/CVPR2026/html/Ponbagavathi_Frame2Freq_Spectral_Adapters_for_Fine-Grained_Video_Understanding_CVPR_2026_paper.html)；[正式补充材料](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Ponbagavathi_Frame2Freq_Spectral_Adapters_CVPR_2026_supplemental.pdf)。补充A.1/Algorithm 1的逐项核查和文件SHA见 `frame2freq_axis_audit.md`。
2. Frequency Entropy：[ICLR正式页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0aee38a6fe9fffc8b658cfb1d872c1d5-Abstract-Conference.html)。正式PDF方法第3、5节已在 `mechanism_related_work.md` 核查。
3. Map the Flow：[ICLR正式页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/537b3416a0c8035794a12b3fa596f941-Abstract-Conference.html)。正式PDF第2–3节证据见 `internal_probe_novelty_audit.md`。
4. T3：[ICLR正式页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b7eecb72574b043ad0c69ea296212450-Abstract-Conference.html)。正文和最终版附录A.4.2/Table 14已核；不以较早预印本替代最终附录。
5. LFM：[CVF正式页](https://openaccess.thecvf.com/content/CVPR2026/html/Ouyang_Language-guided_Frequency_Modulation_for_Large_Vision-Language_Models_CVPR_2026_paper.html)。本项目证据为正式摘要，阅读深度见原主矩阵R20。
6. Causality Matters：[AAAI正式PDF](https://ojs.aaai.org/index.php/AAAI/article/download/37856/41818)，DOI `10.1609/aaai.v40i11.37856`。
7. NOTICE：[NAACL主会正式页](https://aclanthology.org/2025.naacl-long.571/)。正式摘要及PDF干预对照已在内部审计核查。
8. MP/LEACE：[ACL Findings正式页](https://aclanthology.org/2025.findings-acl.674/)。只作选择性控制的方法学边界。
