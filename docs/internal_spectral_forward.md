# 内部谱干预：正式先例、真实算子与跨记忆框架验证

核查日期：2026-11-01。本文配合`internal_memory_protocol_v3.md`，聚焦模型内部前向传播的时序证据，不继续把外部选帧器作为创新核心。本轮只读取正式论文、作者代码与现有协议，没有加载模型、运行GPU、训练模块或读取确认QA结果，也未改变文献主表计数。下述方法均未在本项目复现。

用户已明确首要路线是在FFN/attention内部挂只读probe，记录输入相关激活并读出时间频谱。诊断probe与把频谱adapter插入前向是两个不同操作；下面的adapter文献是可能干预方案的相关工作，不能仅凭其存在认定已覆盖本项目尚待完成的机制诊断。反过来，内部挂probe本身也不证明新颖性，需比较所识别的问题、谱统计、因果证据与干预结果。最终干预形式由诊断决定，不预先锁定adapter。

结论：内部时间谱adapter、视觉空间谱调制、KV频域压缩和attention前的谱子空间编辑都已有正式先例。待检验的空间是：找到任务相关的内部证据传播失败，证明频率/相位诊断比普通时域诊断多提供什么，再用同一个小模块在未参与调参的memory harness上验证改善。目前不能把这条证据链写成已成立的结果。本轮未核到下列论文对“同一模块跨多种外部memory harness”的直接验证。

## 1. 五种不同的“谱”

| 对象 | 变换轴 | 主要解释边界 |
| --- | --- | --- |
| 视觉时间Fourier谱 | `H[time, patch, channel]`的时间轴 | 变化幅值和相位不自动等于任务证据 |
| 视觉空间Fourier谱 | 单帧二维patch格 | 空间结构和纹理不能直接解释跨帧顺序 |
| K/V序列DCT | 文本或多模态token序号 | cycles/token不是视频Hz |
| 协方差SVD谱 | 通道统计矩阵的奇异值和子空间 | 语义相关方向不是沿物理时间的快慢变化 |
| RoPE频率 | Q/K的二维位置旋转子空间 | 位置旋转速率不同于hidden时间Fourier谱 |

这些机制可组合，但不能因都叫spectral就混为同一种证据。某一轴的低频有效，不证明另一轴的低频就是长期记忆。

## 2. 四篇最直接的正式先例

### I01：Frame2Freq，CVPR 2026

正式标题：[Frame2Freq: Spectral Adapters for Fine-Grained Video Understanding](https://openaccess.thecvf.com/content/CVPR2026/html/Ponbagavathi_Frame2Freq_Spectral_Adapters_for_Fine-Grained_Video_Understanding_CVPR_2026_paper.html)，pp.24073–24083。正式PDF方法及[作者实现](https://github.com/th-nesh/Frame2Freq/blob/main/models_adapter.py)已读；与主矩阵R02为同一篇。

作者在冻结的图像视觉Transformer内部加入可训练adapter。ST版本沿时间STFT后变换、逆变换；MS版本结合多窗口FFT和时间卷积，再投影回骨干维度。训练adapter及分类头，目标是细粒度视频动作识别，不能直接转述为decoder长期记忆问答。

本轮源码核查确认：

- `TemporalFreqAdapter`去掉CLS并降维，频率支路形成`[B,chunks,window,H,W,C]`，在`dim=2`做`rfft`。各窗口逆变换回相同时间长度后再平均；不是先硬对齐不同尺寸的频谱。
- 复数实部/虚部展开给实值depthwise `Conv3d`，再组合并`irfft`；不能自动称为有复数乘法约束的复值卷积。
- 支持attention前或MLP前插入。默认B/16配置在attention残差后、原MLP前，将更新加回视觉patch。后续attention提供跨token传播通路。
- MS路径只接受8、16、32帧；STFT用Hann窗。它不是适用于任意时间戳、空间格和混合文本序列的现成模块。

**复现前需解决的源码问题**：读取版本的MS路径对`view_as_real`结果直接`view`成通道优先布局，未显式把末尾通道/实虚轴换到前面；STFT路径也有需核对的轴重排。另外`stft(center=False)`与未指定`center`的`istft`不一致。需用带坐标标签的张量和恒等往返检查轴语义、边界重建；形状一致不足以证明语义正确。这只是静态审阅，未运行作者代码或判定论文结果失效。读取文件SHA256为`63bb1052f16fee7ffc681b1a39164a19ef8f3888db8f7718878a67297c286f5b`；链接为可变`main`分支。

**碰撞结论**：“冻结骨干+学习时间谱模式+逆变换回注hidden”已有先例。新研究必须靠内部失败定位、因果约束和跨harness读出证据建立区别，而非只将分类换成QA。

### I02：LFM，CVPR 2026

正式标题：[Language-guided Frequency Modulation for Large Vision-Language Models](https://openaccess.thecvf.com/content/CVPR2026/html/Ouyang_Language-guided_Frequency_Modulation_for_Large_Vision-Language_Models_CVPR_2026_paper.html)，pp.39175–39185。正式PDF第3节和第4.1节已读；与R20相同。

LFM读取视觉编码器attention后、FFN后的各层表示，把相邻层分组；组内attention后特征经FFT，用语言表示调制，再逆变换，与FFN后表示融合。各组拼接、MLP投影后送入LLM。理论针对二维空间Fourier，不是内部时间谱记忆。

训练不是零成本插入：CLIP版本预训练冻结视觉、语言编码器与LLM，只更新LFM的MLP；非CLIP版本预训练还解冻语言编码器。instruction tuning阶段正文明确优化全部参数。跨视觉骨干、LVLM和benchmark的结果，不等于同一权重跨memory harness迁移。

本轮未取得可验证作者代码；压平写法`FFT(P²×D)`的实际二维实现、复数cosine和调制系数参数化仍不擅自补充。借鉴其式子时须明确本项目自己的算子定义。

**碰撞结论**：内部多层视觉谱、语言条件调制和前向接入已有先例。查询未知写入不能使用未来问题；读出时条件化若要求重新prefill，须计入计算。

### I03：FreqKV，ICLR 2026

正式标题：[FreqKV: Key-Value Compression in Frequency Domain for Context Window Extension](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ac4c13a5c6ec1bba68c9ce1d908321d2-Abstract-Conference.html)。正式PDF第3节已读；与R03相同。

沿KV的token序列轴DCT，保留低频，按更短长度IDCT并重标幅值，生成较短合成KV；后续query直接attention这些KV。缓存满后反复压缩，sink token单独保留。压缩算子无参数，但长上下文实验还使用训练，不能把全部结果称为冻结模型零训练增益。

这确实改变内部前向和后续可读历史，但token时间不等于视觉物理时间；文本低频保留效果不证明短视频事件细节可以丢弃。它改变缓存长度，应与等输入、等缓存长度的残差adapter分开比较。

**碰撞结论**：内部KV谱、迭代频域压缩和长上下文读出均已有先例。需要区别“保持有限缓存容量”和“同一输入下纠正读出失败”。

### I04：SEKA / AdaSEKA，ICLR 2026

正式标题：[Spectral Attention Steering for Prompt Highlighting](https://proceedings.iclr.cc/paper_files/paper/2026/hash/32ddf8ee55540539eb46083cd72361e8-Abstract-Conference.html)。已核正式总目录及正式PDF第3节。

SEKA从中性、相关和不相关对比prompt采集key，对交叉协方差做SVD，构造按层/头的投影子空间；attention前编辑指定token的key。AdaSEKA用query方向组合专家子空间。作者称training-free，指不需梯度微调模型，不等于无需离线样本、子空间拟合或超参。

其谱是通道统计矩阵的奇异值谱，不是时间DFT。key编辑等价于向attention logits加入特定相关性项，不需保存完整attention矩阵。原任务是prompt highlighting，不能作为视频记忆改善证据。

**碰撞结论**：从内部激活拟合子空间、再编辑K影响读取已有直接先例。新方法应证明时序结构或约束带来的贡献，不能仅更换spectral一词所指的轴。

## 3. 两个相邻机制的效应边界

[Causality Matters: How Temporal Information Emerges in Video Language Models，AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/download/37856/41818)已通过按层attention knockout追踪帧间、帧到query、query到输出路径。因此“早层形成时序、中层汇入问题”不能不加限定地当作新发现。当前模型的具体失败与可修复位置仍需实验；详见`mechanism_related_work.md`的M01。

[Probing Rotary Position Embeddings through Frequency Entropy，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0aee38a6fe9fffc8b658cfb1d872c1d5-Abstract-Conference.html)已有内部子空间诊断与缩放干预。正式摘要明确：衰减极端熵维度后，下游准确率与基线统计上不可区分，困惑度有温和改善。不能写成已证明准确率或视频长期记忆提升。二维RoPE旋转保范数；缩幅也不等于降低旋转相位速度。

## 4. 当前Qwen3-VL如何取内部状态

依据固定版本[Transformers v4.57.1源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)及本地已核计算图，区分：

    a = Attention(RMSNorm(h))
    y = h + a
    u = RMSNorm(y)
    m = MLP(u)
    v = y + m

| 目标量 | 实际取点 | 主要误区 |
| --- | --- | --- |
| attention增量a | attention输出tuple第0项 | 不是attention权重或整层输出 |
| FFN前残差y | post_attention_layernorm输入 | 不等于归一化后的FFN输入 |
| FFN输入u、增量m | mlp输入/输出hook | MLP(mean(u))不等于mean(MLP(u)) |
| 层输出v | decoder层返回值 | 初始若干层另有外层DeepStack注入 |
| Q/K/V | projection、Q/K norm、RoPE前后分别记录 | projection输出未必是attention实际输入 |
| 问题/答案位置 | 对应残差、query及最终读出 | 视觉状态可读不保证问题token已读取 |

先用模态mask与格点元数据还原`[B,T,P,D]`。时间戳文本不可混入时间FFT；一个视觉时间格是temporal patch，不是任意单帧，16帧探针只有8格。空间均值只作廉价筛查，无法据此重建局部patch变化；固定空间patch也不等于跟踪同一物体。

最后decoder层历史视觉位置的FFN输出没有后续跨token通路影响标准末端文本答案，适合作为结构负对照。中层有通路但不证明是瓶颈。残差adapter、K编辑和V编辑的生效位置不同，不能混为同一干预。

## 5. 从学习谱模式到因果效应

候选算子以`internal_memory_protocol_v3.md`中的因果低秩FIR残差为准，不另立冲突的训练方案。FIR与其频率响应是同一线性操作，“用了FFT”不是独立贡献。

1. **诊断**：预定义频率、相位或交叉谱指标能否在新视频预测读出失败，并超过同容量时域特征？必须匹配划分、probe容量和调参次数。
2. **干预**：超过同范数随机方向、标量门控、普通通道adapter和等参数时域模块，才能排除通用扰动/增加容量的解释。数学等价的时域与频域实现不能当作两种机制。
3. **迁移**：训练一次的参数在未见视频和未参与开发的harness中是否有效？模块位于内部不自动保证外部框架无关。

不要把最大化高频能量当任务目标。使用训练划分中的任务损失和预声明保持/扰动约束；teacher、事件标注及patching监督均须隔离确认集。输出投影零初始化、输入投影非零初始化可提供恒等起点，全部因子同时置零可能使梯度无法启动。复数滤波须保持实值逆变换条件，并区分幅值和相位操作。

整段Fourier滤波重写过去token可能引入未来依赖，原decoder的causal mask不会自动修复。在线方法必须明确可见前缀、边界填充、缓存更新及模块状态；离线块级实验可单列，不能同时宣称原生在线。

## 6. 同一模块跨harness的交叉实验

| harness类别 | reader输入 | 视觉模块适用性与限制 |
| --- | --- | --- |
| 均匀网格/原始短片 | RGB或原始视觉token | 基础条件；固定时间分辨率及位置 |
| 固定预算真实帧选择 | 稀疏RGB、原始时间戳 | 可运行但间隔不均匀；不能把cycles/sample当相同Hz |
| 检索真实片段 | 多段视觉证据 | 核对片段边界、检索顺序与原始时间顺序 |
| 历史KV回载 | 已计算K/V | 原hidden hook未必重跑；检查缓存版本与重新编码成本 |
| 纯文本摘要 | 没有视觉时间格 | 当前视觉模块不适用，需要另设文本路径 |

第一阶段先固定证据：每种harness生成一次payload，保存内容哈希、时间戳、token位置和预算；基线与同一theta的reader读取同一payload。它隔离内部读出收益。第二阶段若模块同时作用writer，则另行重建记忆，计入写入、存储、检索和reader全部成本，不能将输入变化混入第一阶段的机制结论。

共享theta仅在训练划分拟合，至少留出一类harness不参加训练和超参选择。每类比较baseline、shared module与等参数非频谱控制，报告任务分差、视频级配对区间、样本数、额外计算和失败题型，再列宏平均及最差组。每个harness单独重训或按测试答案搜频带，不是共享模块迁移。

“每个已测试harness均改善”需要预声明逐组判定并处理多重比较；平均提高不能替代。有限集合的正结果也不保证任意框架改善：删除个例证据、已经满分的基线、跳过干预点的缓存路径都是明确边界。

## 7. 当前结论与下一步门槛

可主张的是研究设计及既有先例核查；不能主张首次内部谱学习、首次前向谱编辑、首次KV谱长上下文，或内部模块天然跨harness有效。内部激活若没有持久化状态，只能影响当前可见证据的传播与读出。

下一步先确定时序证据在哪个位置不可读或未被读取，再选择FFN残差、attention路由还是KV状态作为训练入口。没有这一定位，直接学频带增益容易重复已有adapter且无法解释收益来源。

本轮新增正式核心项为I04；I01–I03为已有主矩阵的深读，未改变22篇计数。未将正式身份尚未核实的SEA、FAST或泛用Fourier网络候选加入核心清单。
