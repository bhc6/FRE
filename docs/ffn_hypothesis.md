# FFN假说：它是否改变了视频时间证据的可读性？

核查日期：2026-11-01。本文的文献与源码核查未加载模型、未读取确认QA成绩，未修改冻结实验。用户随后明确授权使用GPU，现已另行完成**原8个开发视频、每次单视频16帧的限额FFN机制探针**；结果见第9节。未恢复Smol全量QA。电脑出现多次重启后，降低输入规模不能被当作硬件稳定性保证。

## 1. 直觉值得检验，但先把“记忆”拆开

**FFN可能影响动态视觉信息的表达和读出，但目前没有证据说明它把新看到的视频写进了长期参数记忆。** 在冻结Qwen3-VL推理中，FFN权重不会随每条视频更新；历史信息由输入、激活、KV或显式外部记忆承载。FFN可以对这些上下文相关激活做非线性变换，放大、压低或重编码某些方向，也可能增加语言先验相对视觉证据的影响。

需要分清三件事：

| 概念 | 实际对象 | 不能直接推出的结论 |
| --- | --- | --- |
| 参数知识 | 训练后权重中的统计关联、事实与语言先验 | 新视频在冻结推理后被写入权重。 |
| 上下文工作记忆 | 当前序列的激活与KV，以及attention可访问的信息 | FFN独立于attention形成了跨帧持久状态。 |
| 外部视频记忆 | 被选择并存下来的帧、时间戳、特征或其他证据 | 某层谱特征更明显，就必然改善后续QA。 |

Qwen3-VL包含视觉编码器的MLP和语言decoder的MLP。本文先限定为**语言decoder的FFN/MLP**，不能把其结论推广到所有视觉MLP。现有缓存中的`layer0`是decoder输入，尚未通过语言decoder的FFN；它已经包含视觉编码器和投影的处理。因此，若只分析`layer0`，最多研究视觉前端表示，不能归因为语言FFN。

## 2. 六篇已独立确认的2025–2026正式论文

只使用官方论文集确认会场；作者arXiv录用声明不代替正式身份。本轮没有核到足以支持“FFN天然保存动态视频长期记忆”的直接论文。以下工作提供邻近证据和可复用的分析方法，不是本假说的有效性证明。

| ID / 正式论文 | 已核机制和阅读深度 | 对当前FFN假说的意义 | 不能移植的主张 |
| --- | --- | --- | --- |
| **F01 Skip-Vision: Efficient and Scalable Acceleration of Vision-Language Models via Adaptive Token Skipping**，**ICCV 2025**, pp.21384–21397，[CVF正式页](https://openaccess.thecvf.com/content/ICCV2025/html/Zeng_Skip-Vision_Efficient_and_Scalable_Acceleration_of_Vision-Language_Models_via_Adaptive_ICCV_2025_paper.html) | 正式摘要和PDF第3–5页已读：比较视觉/文字token的FFN更新，保留重要视觉token、跳过部分冗余token的FFN，增加summary tokens，并在训练后删除相应KV。 | 最直接的视觉FFN先例。应测`FFN增量/残差`与token类别，并考虑少数重要token被空间均值掩盖。 | 它含训练和适配summary token；不是证明可在冻结Qwen3-VL中无损跳过FFN。平均更新小不等于对短动作或QA不重要。 |
| **F02 Too Late to Recall: Explaining the Two-Hop Problem in Multimodal Knowledge Retrieval**，**NeurIPS 2025**，[正式页](https://papers.neurips.cc/paper_files/paper/2025/hash/4526cfacdbca6b6e184568dac91bf070-Abstract-Conference.html) | 正式摘要和PDF第3–5页已读：区分视觉实体形成与关联事实召回；比较同层MLP/attention的attribution，并进一步通过activation patching检查事实召回路径。论文测试包含Qwen2.5-VL。 | 支持把“视觉证据形成太晚”和“知识调用失败”作为不同解释；要求同层attention对照及干预，不能只看FFN相关性热图。 | 对象是图像实体的静态事实召回，不是视频顺序、短暂动作或Qwen3-VL-2B的长期记忆。梯度归因是近似，仍需具体干预验证。 |
| **F03 ParamMute: Suppressing Knowledge-Critical FFNs for Faithful Retrieval-Augmented Generation**，**NeurIPS 2025**，[正式页](https://papers.neurips.cc/paper_files/paper/2025/hash/90d1e515d74686ad148b469a3cecbbaa-Abstract-Conference.html) | 正式摘要和PDF第3–4页已读：在文字RAG里按不忠实回答的激活差选择FFN，缩放其激活；完整方法还包含知识适应与偏好优化。 | 为“语言参数先验可能压过当前视觉证据”提供可检验的邻近假设。可借鉴单层残差缩放的局部干预形式。 | 不是无训练的视频记忆方法。其层选择用带标签数据，不能用确认集答案挑出最有利层；本项目简单缩放不能冒充ParamMute复现。 |
| **F04 Distributional Associations vs In-Context Reasoning: A Study of Feed-forward and Attention Layers**，**ICLR 2025**，[正式页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/533a89d7c2980218d82dbe3ea85f77ae-Abstract-Conference.html) | 正式摘要已读：在受控合成任务上分析FFN倾向学习分布关联、attention倾向处理上下文关系，并在Pythia中作消融。 | 为“参数关联”与“输入内时序推理”提供清楚区分，说明attention是必要机制对照。 | 这是特定任务和模型中的分工倾向，不是所有Transformer的硬性分工定理；不能推出FFN不处理时间信息。 |
| **F05 MLP Memory: A Retriever-Pretrained Memory for Large Language Models**，**ICLR 2026**，[正式页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d72ae75abaa70a3b19c5d4f436c680d1-Abstract-Conference.html) | 正式摘要已读：额外MLP预训练模仿kNN检索器，以概率插值接入decoder。 | 说明“MLP可作参数化记忆”需要交代训练对象、写入过程和集成方式。 | 它是专门训练的附加模块；不能据此认定现成FFN会在冻结前向中写入新视频。没有本项目所需的动态视频验证。 |
| **F06 NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks**，**ICLR 2026**，[正式页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2764bbdbd9f36d9f1ec63f4ad22b24ff-Abstract-Conference.html) | 正式摘要已读：用谱熵、有效维度及特征值分布变化分析FFN非线性和优化器对隐空间方差的影响。 | 支持同时观察FFN前后表征几何，避免只看整体范数。 | **协方差/特征值谱不是视频时间DFT频谱**。其“方差再注入”不能转述为“恢复高频短动作”，相关泛化现象也不是记忆因果证据。 |

F01–F03读取了选定方法章节，其余三篇当前仅是官方摘要级核查；未声称完成全文证明审查。六项均未在本项目复现，文中不引用论文成绩作为本地效果。

## 3. Qwen3-VL里应该测哪里

实现基准为项目固定的`transformers==4.57.1`，并核对[该版本官方源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)。`Qwen3VLTextMLP`使用gate/up投影、非线性乘积和down投影；它逐token运算，不在自身内部显式跨时间token混合。其输入已包含attention带来的上下文，所以输出仍可能携带时间信息。

对decoder第`ell`层，把五个量分开记录：

\[
h_\ell=\text{block input},\qquad
a_\ell=\operatorname{Attention}_\ell(\operatorname{RMSNorm}(h_\ell)),
\]
\[
y_\ell=h_\ell+a_\ell,\qquad
u_\ell=\operatorname{RMSNorm}(y_\ell),\qquad
m_\ell=\operatorname{MLP}_\ell(u_\ell),
\]
\[
v_\ell=y_\ell+m_\ell.
\]

其中`m`才是FFN增量，`y→v`才是完整的FFN残差更新。`u→m`是归一化输入到MLP输出的映射，不能与`y→v`混为一谈。

| Hook位置 | 应捕获的量 | 用途 |
| --- | --- | --- |
| `language_model.layers[ell]` pre-hook | `h` | 当前decoder层输入。 |
| 同层`self_attn` output hook | `a`，从返回tuple取第一个张量 | 与FFN比较的attention增量；不要求输出完整attention矩阵。 |
| 同层`post_attention_layernorm` pre-hook | `y` | 加完attention、尚未归一化的FFN残差输入。 |
| 同层`mlp` pre-hook | `u` | 真正送入MLP的归一化激活。 |
| 同层`mlp` output hook | `m` | FFN自身的输出。 |
| decoder block output hook | `v` | 加完FFN、尚未加入外部DeepStack视觉残差的状态。 |

**DeepStack是关键混杂项。** 该版本在最初若干decoder层结束后，另外向视觉位置添加DeepStack特征。现有后续层pre-hook看到的状态可能包含这次添加，所以相邻层差分不是纯FFN增量，甚至不是仅attention加FFN。必须在block输出和DeepStack之后的边界上分别识别；不能把DeepStack带来的视觉变化记到FFN头上。最终RMSNorm也要单列。

项目现有缓存仅含选定decoder阶段的空间均值，不含上述支路输出。`layer0/14/28`不能事后代数分离FFN与attention。更不能对已池化向量重新跑MLP，声称还原原token的均值输出，因为一般有

\[
\operatorname{MLP}(\operatorname{mean}_p u_p)
\ne \operatorname{mean}_p\operatorname{MLP}(u_p).
\]

## 4. 三个可被否定的假说

### H1：FFN改变时间信号的可分性

在同一层、相同视觉token位置和相同池化下，`y→v`使方向、顺序或短暂状态的可分性稳定改变，且这种改变不能仅用整体尺度、DC偏移、DeepStack或同层attention解释。

如果只看到频带能量变化，或变化在归一化后消失，或只能在单个开发视频上挑出，应否定当前证据足以支持H1。非线性变换会产生谐波/频率混合；高频能量增长本身不说明时间证据增加。单纯DFT或功率谱也不能区分所有反转/位移，因此不能把“高频多”用作唯一判据。

### H2：某层FFN对具体时间判断有可测的因果影响

预先固定的一层与视觉位置上，温和缩放或替换FFN增量，对一组受控时间任务的答案分数产生一致、方向符合预期的变化；同层attention、同范数扰动及静态问题对照帮助排除普遍性能损伤。

若所有题都一并变差、变化随任意相同幅度扰动出现，或只改变格式而非答案证据，就不能声称FFN专门承载时间记忆。没有明显效应仅能否定这组层位、干预及任务的具体假说，不能证明全模型所有FFN都无关。

### H3：FFN诊断可改善有限存储的选帧

只有在H1/H2的证据足够且独立复核后，才考虑用FFN增量或前后变化构造query-free写入分数。在同4个pair、实际存储预算、reader及强时域/覆盖基线下验证；不能直接把QA干预效果等同选帧收益。

如果后续reader参数也同时改变，必须另开实验因素，不能把更换reader后的成绩归因给写入方法。当前尚未达到H3的有效性结论。

## 5. 低资源、逐步停止的检验设计

### 阶段A：现在可以完成的CPU工作

核对源码、层编号、token分组与DeepStack边界，建立输出字段和分析公式，使用已存在缓存检查整层表示的现象。现有缓存只能帮助挑选后续问题，**不能生成FFN专属结果**。GPU探针遵循用户已授权的单独限额，不并发加载模型或连带恢复全QA；本文本身提供的是文献与设计。

### 阶段B：机器稳定后才考虑的单次前向诊断

先固定一个开发视频和相同16帧输入，使用少量预先声明的decoder层。当前实现入口固定0起始索引`0,13,27`，在同一个前向中测早、中、末层，并检查层号范围；不扫全28层、不做全题生成、不回传梯度。16帧对应8个时间格，是支路测量的输入预算，**不同于记忆选择实验最终4个pair/8帧的存储预算**。初次只为确认hook及代数关系，不能凭单例声称有效；未经额外验证，也不能把16帧观测外推到64帧候选轨迹。

使用单进程、batch=1、`inference_mode`、`use_cache=False`、`output_hidden_states=False`、`output_attentions=False`，直接调用基座获得状态，避免全词表logits与生成循环。在hook中先选视觉位置，再按已有时间格做空间聚合；只保留指定少量层的CPU聚合数组，并及时移除hook。所有条件使用相同输入token上限、dtype、图像尺寸与时间戳。若必须降低分辨率，先确认关键对象和动作仍可见；低分辨率造成的信息缺失是单独混杂项。

一个时间格除均值向量外，可保留支路token范数的均值与最大值，以及空间方差标量。这样能检查均值抵消，不需要保存整段、全层、全部中间维度激活。首轮不采集巨大`intermediate_size`的gate/up张量、不训练SAE、不做全神经元梯度归因。

验证数值关系`y≈h+a`及`v≈y+m`，使用符合当前dtype的误差范围；如果不满足，先修hook边界，不进入科学解释。这些是采集正确性的检查，不是方法效果测试。

### 阶段C：先比较观测，再做极少量前向干预

在扩展到更多开发样例前，先确定固定层、频带、指标与题型，不根据确认QA调参。观察至少包含：

- FFN残差比`||m||/(||y||+eps)`与attention残差比`||a||/(||h||+eps)`；视觉位置与问题/末token分别统计，不能把不同位置混成一条曲线。
- 对`h,a,y,m,v`沿同一时间格去均值后做正确归一化DFT，同时报告绝对能量及总能量归一化占比，排除纯尺度变化。
- 比较`y→v`的频带失真、特征距离与简单受控标签可分性；仅有8个开发视频时避免高维训练探针，多视频后仍需按源视频分组留出。
- 用原样、调换两个时间片段内容且保持时间槽位/长度不变的受控样例比较；需要人工确认新的顺序标签与视觉可见性。frame-pair内部结构、时间戳及静态对象应尽量一致。

随后才测试同层、同位置的轻量干预：`v=y+alpha*m`，例如预先固定`alpha=1`为原样、`alpha=0.8`为温和缩放。一次只改一个层位和一种token范围。相同`alpha`作用于attention并不意味着相同扰动强度，必须同时报告实际残差变化范数；若加范数匹配对照，其规则在看结果前固定。

不以“删除所有FFN后模型崩了”作为时间记忆证据。强消融会同时破坏语言、视觉与一般推理。温和缩放无效时不应立即在大量层和系数上搜索到显著结果。

### 阶段D：小规模activation patching检验

如果阶段C出现一致现象，再构造配对的clean/corrupted输入。优先使用保持长度、位置、图像尺寸的受控顺序变化，避免任意高斯噪声造成不可解释的分布外损伤；变换后哪一个答案正确必须清楚。

先取得clean的单层FFN增量，再在corrupted输入同层同位置注入，观察是否朝clean答案改变。对照至少包含同层attention输出patch、identity/sham patch，以及预先定义的同范数非相关扰动。clean→corrupted激活替换本身仍可能产生分布外状态，因此“恢复”只能支持特定层位对该受控行为的因果贡献，不能证明它存储了完整事件或长期记忆。

低资源读出可以固定答案格式、评分候选答案的条件对数似然或logit margin，避免长文本生成；单token字母评分前必须核实tokenization。若候选不是单token，需按相同规则累计对应token的log-prob，不能比较不等价的分数。保留原样准确率和原始margin，只有clean/corrupted差足够明确时才计算归一化恢复比例，避免小分母夸大效果。

这些QA样例属于机制开发，不接触已冻结确认结果。后续若作为新方法证据，需要另行冻结层位、干预、预算和独立确认集。

## 6. 应输出的数据与停止条件

每条记录至少包括：模型revision、transformers版本、视频与输入帧哈希、源帧PTS、时间槽位、prompt哈希、层索引及其语义、hook位置、token掩码、dtype、`use_cache`、DeepStack边界、每个支路的聚合规则。干预记录另含`alpha`、patch来源、实际扰动范数、答案格式与评分方法。记录峰值资源和耗时，但不把未运行的资源估算写成测量值。

先停止并修正分析的情况：hook代数不符、只存池化输入却试图重建真实FFN输出、混入DeepStack、顺序变换改变了题目语义却未更新标签、分辨率降低后动作已不可见。科学上停止有效性主张的情况：只见频谱变化没有行为证据；FFN效应被同层attention或同范数扰动完整解释；只在挑选后的个例成立；一般能力损害大于目标改善。

当前合理结论是：**FFN是值得精确分解和局部干预的候选环节，暂不能称作视频长期记忆的根因或解决方案。** 现有正式工作最支持的是“参数先验、视觉表示形成与读出路径需要分开测”，而不是从FFN名称或频谱图直接推断记忆能力。

## 7. 本轮完成与未完成

- 已核六篇正式会议页面；F01–F03另读选定方法章节；同时从ICLR/NeurIPS官方总目录核对相关条目。
- 已核项目的decoder阶段采集语义，以及固定transformers版本的MLP、残差与DeepStack源码顺序。
- 本轮文献/源码核查未导入或执行模型，未运行论文实现，未修改冻结实验，未读取确认QA成绩；用户授权的单视频16帧GPU探针单独记录，不把计划写成成功结果。
- 论文有效性、Qwen3-VL上的因果效果、资源稳定性与选帧收益仍待分别验证；本文件是可执行前的研究定义。

## 8. 独立工程入口与产物

入口为`python -m spectral_memory.probe_ffn`，资源监测实现位于`spectral_memory/resource_guard.py`。该入口只读原开发视频manifest，使用固定中性指令`Describe the visible events.`，默认按视频ID字典序取一个视频；不加载问题或答案，不做生成。它不是冻结QA运行器，也不改变其选择器和协议。

当前实现抓取`decoder_input / attention_delta / pre_ffn / ffn_input / ffn_delta / post_ffn`，按时间格保存空间均值、空间均方和最大token RMS。真实视觉token在hook内复制到CPU；`pre_ffn = decoder_input + attention_delta`及`post_ffn = pre_ffn + ffn_delta`在空间池化前按实际dtype验证。`post_ffn`来自decoder block返回值，位于外部DeepStack添加之前；另存浮点加法的舍入残差，避免把数值舍入误读成FFN机制。

产物约定：`probe.json`记录状态、模型/输入/源码哈希和身份检查；`branches.npz`记录聚合分支；`frames.manifest.jsonl`保存采样信息；`resources.jsonl`及资源汇总记录可取得的遥测。资源监控中止时须一并检查`resource_stop.json`与进程退出码，不能仅凭某个结果文件存在就判定完成。

本节记录源码入口与产物定义，实测结果另列于第9节。监测为轮询，可能漏掉短时资源峰值，也不能保证系统不会重启。实际参数和停止策略以被记录哈希对应的源码及`resource_limits.json`为准。

## 9. 已完成的 GPU 开发诊断

固定 Qwen3-VL-2B 在原8视频上完成16帧、decoder 0/13/27层的采集；全部48个原dtype残差恒等检查通过，144条分支时间轨迹已保存。独立汇总直接从NPZ重算时间方差和DC，核对视频/帧字节、版本、提示、时间网格和源码快照；完整逐视频值、范围、资源记录和图见 [实测报告](../outputs/ffn_development_report/report.md)。

末层FFN的时间能量占比之比中位数为 **9.4422**，但绝对时间能量之比只有 **1.0832**，DC能量之比为 **0.1101**。8/8视频均出现占比升高伴随DC下降；`video_2564`的绝对时间能量甚至下降约3.07%，占比仍增为8.43倍。因此不能把归一化占比的提升直接称为有用动态信息增强。DC只是数学上的时间均值成分，尚不能语义化为背景或无用噪声。

这识别了一个明确的**指标解释混杂**，没有证明FFN记忆方法有效。原v1 `spectral_balanced`已先去均值，该观察不能解释v1失败或证明修复。探针没有测QA、因果patching、选帧收益或长期记忆；每视频只有8个时间格，频率分辨率也有限。

本轮最大CUDA allocated为4,363,261,440 bytes，reserved为4,410,310,656 bytes；八次运行均正常结束，本机启动时间未改变，结束后GPU已空闲。资源上限、两秒采样与故障边界见独立报告。这些事实不代表先前三次蓝屏的根因已经修复。
