# 内部只读探针与时间频谱：失败定位的新颖性审计

核查日期：2026-11-01。研究对象是“读取模型内部激活→分析沿真实视觉时间轴的频谱/相位→定位时序证据传播或读出失败”，不是预先设计adapter，也不是继续优化外部选帧器。本轮只检索论文并审阅正文；没有运行GPU、训练模型、读取已看过40个确认视频的结果或修改旧协议。

**审计结论：已有正式工作直接研究视觉时序信息可读但模型答错、VideoLLM内部时序传播路径，以及激活的频谱诊断。尚不能将“内部只读probe+FFT”本身作为充分创新，也不能因已有邻近工作就判定这条具体研究路线没有空间。** 值得检验的增量是：特定频率/相位结构能否比同容量非频谱诊断更可靠地预测并定位失败，而且这种定位通过有选择性的干预得到支持。这个增量目前是问题，不是结果。

## 1. 范围与术语

当前核心清单共9篇：8篇2025–2026正式主会论文，1篇明确标为ACL 2025 Findings。优先复用已有审阅；新增深读T3、Map the Flow、NOTICE及amnesic probing论文，另在Frame2Freq补充材料审计后将其内部频谱诊断纳入核心对照。会议身份以正式论文集为准，网页HTTP 200本身不构成身份验证。没有把搜索关键词未命中当成不存在一致工作的证明。

须区分三个操作：

- **只读hook**：记录前向中的张量，不改模型输出。它是采集工具，不自动提供关于内部能力的结论。
- **统计/读出probe**：在采集张量上计算谱、训练分类器或使用logit lens。它测试指定读出方式下的可用信息；较强probe也可能自己完成时序计算。
- **因果干预**：改变内部表示或路径并测下游行为。它检验特定替换的作用；不满足选择性控制时，性能下降可能来自一般破坏。

同时区分**任务标签**（顺序、方向、事件状态）与**失败标签**（固定模型在该输入上是否答对）。能解码任务标签不等于能定位错误来源；能预测答错也可能只是识别输入难度。

## 2. 九篇核心工作的精确对照

| ID / 正式身份 | 对象与轴 | 标签或读出目标 | 干预与实际解释 | 与当前问题的关系 |
| --- | --- | --- | --- | --- |
| **P01 T3，ICLR 2025** | 视觉入口及decoder末层hidden；有序token特征，无Fourier分析 | 顺序、属性变化、时序指代、时序定位；LSTM及末时间步线性probe | 文本时序训练改善VideoQA；probe本身不是层位干预 | **直接重合于失败问题**：内部可读但原模型答错已有证据；未给出时间谱定位 |
| **P02 Map the Flow，ICLR 2026** | VideoLLM逐层视觉/文本hidden、attention边与logit lens | 时序VideoQA、时间关键词、答案概率；正文分析筛选基线答对样本 | 阻断跨帧、视频到问题、问题到末token路径；检验保留关键路径 | **直接重合于传播定位**；成功路径分析不能直接代替失败机制 |
| **P03 Causality Matters，AAAI 2026** | 帧间因果交互、显式位置与按层信息传播 | 视频时间任务与输出行为 | 位置/帧顺序对照及attention knockout | **直接相邻于时序机制**；“因果attention形成时间信息”已有先例 |
| **P04 Frequency Entropy，ICLR 2026** | RoPE二维Q/K子空间的范数轨迹，沿token序列分析频谱 | 谱熵、上下文表示和语言模型性能 | 根据熵调整维度幅值；准确率与基线统计上不可区分 | **最接近内部谱诊断**，但不是视觉物理时间轴或时序证据标签 |
| **P05 NerVE，ICLR 2026** | FFN内部表示的特征值/协方差谱 | 谱熵、有效维度、分布变化及泛化关联 | 比较非线性、归一化和优化器等设计 | **谱几何邻近工作**；特征值谱不是时间Fourier谱 |
| **P06 Video-LevelGauge，ICLR 2026** | 同一probe片段在不同上下文、位置中的行为 | 受控片段问答与位置/邻居偏差 | 调整上下文和插入位置 | **混杂控制先例**；这里probe主要指输入片段，不是hidden分类器 |
| **P07 NOTICE，NAACL 2025主会** | BLIP/LLaVA内部attention/MLP，静态图文任务 | 语义图像对及文本替换下的模型行为 | 语义corruption与activation patching；对比高斯噪声 | **干预可靠性邻近工作**；相关于选择clean/corrupt条件，不是视频谱诊断 |
| **P08 MP/LEACE审计，ACL 2025 Findings** | BERT表示中的语言属性子空间 | 依存关系、细/粗POS及语言模型表现 | 属性移除、同维随机投影、显式标签恢复 | **probe选择性方法学**；证明大幅下降不自动等于目标信息的因果作用 |
| **P09 Frame2Freq，CVPR 2026主会** | 视觉骨干adapter及其时间embedding；沿时间FFT，另有原视频3D谱图 | 动作分类；补充A.1对时间embedding功率做类别间/内方差判别性分析 | 学习式时间/频率adapter；内部谱的标签判别性用于分析 | **直接相邻于内部谱诊断**；不能仅概括为adapter，尚不同于decoder支路的QA失败定位与因果修复 |

P09的正式身份、补充材料Algorithm 1、作者代码FFT轴及布局风险详见`frame2freq_axis_audit.md`。该补充修正了此前将其仅归为前向adapter的过窄描述；不据此推断当前路线不存在创新空间。

### P01：T3是目前必须正面比较的直接先例

[正式页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b7eecb72574b043ad0c69ea296212450-Abstract-Conference.html)；完整标题为**Temporal Reasoning Transfer from Text to Video**。本轮读取正式PDF第2节、附录A.3/A.4.2和Table 14。预印本网页与最终正式版内容不同，末层probe结论依据正式版。

正文使用受控顺序、亮度/形状变化、指代与定位任务，训练视觉入口的LSTM读出；另以帧描述测试LLM的文本时序能力。正式附录进一步从LongVA-7B末层取hidden：全特征训练LSTM，最后时间步训练线性probe。Table 14平均准确率分别为视觉入口LSTM **94.0**、末层LSTM **86.8**、末时间步线性probe **78.4**、原LongVA-7B QA **65.5**。这是论文结果，不是本项目复现。

由此，不能把“时序信息在内部仍能被读取，但原模型没有正确回答”写成首次观察。另一方面，全序列LSTM接收有序特征，可能自身完成时序运算；它不单独证明每个输入token已编码关系，也不确定原模型的哪条路径导致失败。最后时间步线性结果缩小了这个解释空间，但仍不定位某个FFN或频带。

### P02：Map the Flow已经把路径定位推进到问题token

[正式页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/537b3416a0c8035794a12b3fa596f941-Abstract-Conference.html)；完整标题为**Map the Flow: Revealing Hidden Pathways of Information in VideoLLMs**。正式PDF第2–3节已读。

作者在TVBench的五类任务上结合attention knockout与logit lens，分析跨帧交互、时间关键词整合和向末token传递；正文主模型为LLaVA-NeXT-7B经视频指令训练的版本。knockout用9层窗口，以免单层删除被残差路径绕过。logit lens对视觉hidden作词表投影，统计时间/空间关键词；这里的词频不是Fourier频率。

其正文明确筛选**基线答对**的样本，再测原答案概率变化。这适合分析已成功行为使用的路径，不能据此证明答错样本在相同位置失去证据。只筛答对也不排除猜中或捷径。当前若研究失败，必须显式包含错误与正确条件，并匹配内容、位置和任务，不能照搬该筛选后仍声称解释失败。

### P03–P06：已核材料的复用与边界

- **P03 Causality Matters: How Temporal Information Emerges in Video Language Models**：[AAAI正式PDF](https://ojs.aaai.org/index.php/AAAI/article/download/37856/41818)，DOI `10.1609/aaai.v40i11.37856`。沿用`mechanism_related_work.md`的M01方法核查。帧间→query→输出路径及位置对照已有先例；本文未重新下载该PDF。
- **P04 Probing Rotary Position Embeddings through Frequency Entropy**：[ICLR正式页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0aee38a6fe9fffc8b658cfb1d872c1d5-Abstract-Conference.html)。沿用M05与`internal_spectral_forward.md`。二维范数的token轨迹含上游内容与计算影响；RoPE旋转本身保范数。论文缩放不等于减慢相位，其准确率无显著差异不能写成已证实的准确率提升。
- **P05 NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks**：[ICLR正式页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2764bbdbd9f36d9f1ec63f4ad22b24ff-Abstract-Conference.html)。当前证据范围为正式摘要：谱熵、participation ratio、早期特征值富集与Jensen–Shannon变化追踪FFN表示几何。没有据此补写视频时间标签或具体谱因果干预。
- **P06 Video-LevelGauge: Investigating Contextual Positional Bias in Video Language Models.**：[ICLR正式页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4c2a220d89b4f88222222b46d9de1495-Abstract-Conference.html)。沿用M02的正式PDF第3节核查。相同事件移位、改变邻居或背景是控制方案先例，不证明hidden可被线性分成“内容+固定位置项”。

### P07：NOTICE说明corruption会改变定位结果

[NAACL正式页](https://aclanthology.org/2025.naacl-long.571/)；完整标题为**What Do VLMs NOTICE? A Mechanistic Interpretability Pipeline for Gaussian-Noise-free Text-Image Corruption and Evaluation**。本轮读取正式摘要及PDF方法/对照部分。

它采用语义图像对与对称文本替换来构造clean/corrupt条件，研究静态图文任务中的内部路径。数据包括SVO-Probes、MIT-States和表情识别；语义对力求仅改变一个因素，但自然背景仍可能不同。正文比较语义corruption与高斯噪声时得到不同的MLP层位定位，说明corruption选择本身影响机制结论。

可迁移原则是使用尽可能匹配的事件对和多种corruption核查；不能声称“语义对”已解决所有混杂。重复静帧、倒放、移除事件、加噪会改变不同因素，不能互换。所有替换需重新核对题意及标签，不能沿用变换前答案。

### P08：MP/LEACE需要标为Findings，且不提供通用因果保证

[正式页](https://aclanthology.org/2025.findings-acl.674/)；完整标题为**Improving Causal Interventions in Amnesic Probing with Mean Projection or LEACE**。本轮读取正文的information control、selectivity control与限制部分。

作者对依存关系、细/粗POS表示比较INLP、MP和LEACE。目标属性移除须与移除同维随机方向比较；另向修改后的表示提供显式金标签，检查能否恢复语言任务。部分INLP条件中随机投影的损害更大，因而仅凭原干预损害不能认定目标属性在起作用。MP/LEACE在所测控制中更具选择性。

这支持本项目增加同维、同范数非目标干预以及移除后的重新probe。论文的金标签恢复是解释性控制，不能放入部署QA并算作模型能力改善。MP/LEACE只控制特定可读性及投影性质，不保证消除所有非线性信息，也不保证跨视频时序属性仍有同样选择性。

## 3. 数学上哪些对比可能没有意义

以下是本项目的代数边界，不作为新论文定理或文献实验结果。

### 3.1 完整FFT不会增加可用信息

令时间轨迹展平为实向量`x`，`Q`是等价于完整Fourier表示的实正交基，`z=Qx`。任一线性得分满足：

    wᵀx + b = (Qw)ᵀz + b
    ||Qw||₂ = ||w||₂.

因此在完整表示、同训练样本、等容量线性分类器、同L2正则及达到相同最优解等条件下，时域与频域的线性probe最优目标相同。数值、优化或特征标准化差异可能破坏实际等价，但不能被描述为FFT创造了新证据。复数DFT或单边rFFT须正确处理实虚部、冗余和权重，不能因表示维数不同偷偷改变probe容量。

有意义的待检验差异是频带限制/结构化正则的样本效率、跨长度与采样条件迁移、失败可解释性或干预定位能力。相同FIR的时域与频域实现数学等价，不能当成不同机制。

### 3.2 功率谱可能丢掉答案所需相位

实序列逆序保留对应Fourier幅值，但可反转事件顺序。故功率谱无法普遍识别所有方向/顺序任务。需要比较保留相位的完整表示、幅值/功率、带相位关系或交叉谱；若只分析能量比例，结论应限于能量统计。

能量的层间变化还可能由DC抵消、范数重标定、空间池化和DeepStack注入造成。先检查真实支路恒等关系及绝对AC/DC，不能把比例上升称作信息新生。静态反事实差只是指定参考下的差，不能未经识别假设称为“纯运动分量”。

## 4. 一个能被否定的增量验证方案

本节是设计建议，不代表已开始GPU实验或形成新的冻结确认集。优先验证是否值得继续，而非预设必须得到谱方法。

1. **固定对象**：用只读hook分开attention增量、FFN前残差、归一化输入、FFN增量和输出；初层另记DeepStack。还原`[T,P,D]`，不把时间戳文本、空间patch编号或通道编号当物理时间。Q/K要记录归一化及RoPE前后边界。
2. **定义任务与反事实**：预声明方向、状态或顺序标签，使用内容匹配且答案确实变化的事件对；同时保留静态能力和内容识别控制。问题、选项位置及提示保持可比。不把采样漏掉关键帧直接称为内部遗忘。
3. **先做只读层位诊断**：比较视觉入口、若干中层、问题/读出位置。区分“事件标签可读性”与“模型错误预测”。更强probe需同容量非谱对照、随机标签控制及按源视频/主体分组的独立测试；全序列probe自己推理的可能性必须保留。
4. **检验谱的增量**：与完整时域、差分、范数/DC、简单位置/时长特征及同参数probe比较；完整正交FFT作为等价性检查。只有在限制频带、相位结构或跨条件泛化上出现可重复差异，才继续提出谱特异假设。
5. **事先预测再干预**：仅用开发划分的谱诊断预选层位/频带，在新的匹配事件对上做抑制和救援；对照同维、同范数非目标方向、identity/sham、最后层视觉FFN结构负对照。检验任务选择性及静态副作用。出现QA变化只能先说明该替换有作用。
6. **根据结果收窄或停止**：若谱不优于时域/位置基线，结论可为“该参数化未提供额外诊断价值”；若hidden仍可读而问题位置失败，优先研究路由；若入口已不可读，回到输入/视觉编码问题；若干预影响广泛任务，撤回特定时序机制解释。

只读阶段的成功不自动授权“长期记忆改善”表述。长期保持还需要明确历史存在KV、递归状态还是外部上下文，并独立控制证据年龄、干扰和输入预算。共享模块跨harness是后续验证，不是本轮谱诊断的默认结论。

## 5. 新颖性声明应如何写

可以作为研究问题：**在受控时序任务中，内部时间谱/相位是否揭示超过既有可读性与attention路径诊断的失败因素，并能预测有选择性的因果干预结果？**

目前不能作为已证事实：首次发现内部时序证据未被使用；首次定位VideoLLM时间路径；首次对内部激活做谱诊断；只读probe发现的频带就是记忆存储；所有外部记忆框架都将改善。

判断贡献不能只看是否存在标题完全相同的论文。即使没有检索到相同的完整组合，也仍需证明这一组合解决了此前方法不能解释的问题。相反，邻近adapter工作不等于已完成“时序失败的谱诊断”，应比较对象、轴、监督、干预和可证伪的预测，而非凭关键词否定。

### 正式清单之外的近期提醒

[Using OCR Heads to Verbalize Image Semantics](https://arxiv.org/abs/2609.18823)，提交于2026-09-16，本轮仅核实arXiv身份与摘要，**不计入9篇核心清单**。作者报告通过OCR相关头构建verbalization lens，可在包括Qwen3-VL在内模型的早层读出语义。它不是视频时间谱工作，但提示读出坐标系可能影响“早层没有语义”的判断。未经进一步正式身份与方法核查，不把其结论扩展为当前时序任务的证据。

### 核查限制

这是针对性查重，不是完整系统综述；没有进行引文网络的穷尽搜索，没有复现任何第三方方法。P03/P04/P06复用已有正文核查，P05为摘要层面；P01/P02/P07/P08新增读取了相关正式正文。来源标题和会场经直接HTTP读取核实；不依赖不可核验的搜索摘要编号。文献主矩阵计数、旧实验协议及已看确认结果均未修改。
