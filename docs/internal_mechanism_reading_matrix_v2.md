# 内部机制阅读矩阵 v2：13 篇已核实主会论文

本轮核查截止 2026-11-01。13 篇为互不重复的 2025/2026 ICLR、NeurIPS、CVPR 主会论文；每篇已核对正式 proceedings/CVF 接收页面，并读取方法、公式/张量定义、实验设置、结果和相关附录。完整 PDF 缓存在本地；这不表示逐字阅读参考文献、checklist或全部案例页。

证据范围：论文报告的数值不等于本地复现结果。本文件没有读取本地模型 predictions/scores，没有访问 GPU，没有下载新模型权重，也没有在新159题上试参或打分。不可把机制诊断、外部probe、oracle patching、训练后增益和训练自由原生QA改善混为一类。

参数口径：除明确列出的Hub tensor metadata外，B标签均保留为作者名义型号；unknown表示没有核实全模型唯一参数总数。MoE active参数、trainable参数、language-backbone参数和vision+projector+decoder总参数分开。未报告/未核实的GPU-hours不以GPU型号推算。

同源与版本：RAR只计正式标题 Reversing Visual Attention Re-Sinking 一篇；预印本、最终论文、补充材料和作者代码都不是额外论文。Frame2Freq已有内部时序频谱诊断，不得把它描述为完全不研究hidden表示。

## 1. 总览

| ID | 正式会议 | 机制/干预 | 训练与评价口径 | 对本项目的角色 |
|---|---|---|---|---|
| [P01](#p01) | ICLR 2025 | 冻结模型的 attention 概率重分配 | 无权重训练；任务阈值 | 直接attention基线 |
| [P02](#p02) | ICLR 2026 | sink 功能分解与训练时 head 稀疏化 | 670k SFT；另有无训练诊断 | sink功能反例与训练型对照 |
| [P03](#p03) | CVPR 2026 | 按层和 head 对 attention 输出乘增益 | 无权重训练；需校准 | head-output gain基线 |
| [P04](#p04) | ICLR 2026 | 真实分隔 token 的 residual hidden 缩放 | 无权重训练；test调参限制 | 优先residual/token基线 |
| [P05](#p05) | CVPR 2025 | attention head 定位与删除诊断 | 无权重训练；删头诊断 | 必要性诊断 |
| [P06](#p06) | ICLR 2026 | token轴 RoPE频率诊断与幅度调制 | 主干预无训练；附录预训练 | token频率工具，非视频结论 |
| [P07](#p07) | ICLR 2026 | attention 边敲除与层间信息路由 | 模型先训练；mask边分析 | 路由因果诊断 |
| [P08](#p08) | ICLR 2025 | 用合成文字QA训练来迁移时序推理 | 200k文字QA+原instruction微调 | 可读出/可使用差距 |
| [P09](#p09) | CVPR 2026 | 真实时序FFT/STFT adapter与中间表征频谱分析 | adapter+分类head训练 | 最直接时序频谱前作 |
| [P10](#p10) | CVPR 2026 | 语言引导空间频率调制与全参数训练 | 全参数instruction微调 | 空间频率强对照 |
| [P11](#p11) | NeurIPS 2025 | FFN实体信息时机、梯度归因和activation patching | oracle donor activation patching | FFN时机与patching方法 |
| [P12](#p12) | ICLR 2026 | FFN非线性前后协方差特征谱诊断 | 小LLM从头训练并诊断 | FFN协方差谱工具 |
| [P13](#p13) | CVPR 2025 | 少量head的外部grounding读出 | 训练集选头；外部grounding读出 | 小模型视觉路径诊断 |

## 2. 逐篇方法、预算与证据

<a id="p01"></a>

### P01. See What You Are Told: Visual Attention Sink in Large Multimodal Models

**会议与读取范围：** ICLR 2025；§4–5; Eq.4; Tables 1,4; Appendix D.2。来源：[正式接收页面](https://proceedings.iclr.cc/paper_files/paper/2025/hash/da8a39bc39ae1c89dd6ebb1e3bcbb3f3-Abstract-Conference.html)；[论文PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/da8a39bc39ae1c89dd6ebb1e3bcbb3f3-Paper-Conference.pdf)。本地：`outputs/literature_internal_mechanism_v2/01_var.pdf` / `01_var.txt`。

**实际张量/算子。** 对每个文本 query、head 的完整 attention 行 A[i,:] 操作。先用非 sink 视觉质量/全部视觉质量选择 image-centric heads；再从 sink 集 S 收回 Ω_i=p Σ_{j∈S}a_ij，将质量按原比例加到非 sink 视觉集 V。sink 项乘(1−p)，其他非视觉项不变。保持全行概率和为1，不改 V 或 residual。

```text
Ω_i=p∑_{j∈S}a_ij; a′_ij=(1−p)a_ij (j∈S); a′_ij=a_ij+Ω_i a_ij/∑_{k∈V}a_ik (j∈V).
```

**模型与参数。** LLaVA-1.5、VILA、Qwen2-VL 等六种 LMM；主要 7B/13B 名义规格。 论文型号常指 decoder 容量；未独立核计 vision/projector 后的唯一参数总数。未见直接2B结果。

**训练数据与计算。** 不训练模型权重；sink 阈值τ=20、p=.6；任务相关ρ=.8/.5/.9；不干预最后一层。 Appendix D.2 报告单张 NVIDIA A6000 48GB；未据此推算 GPU-hours。

**可核查结果。** GQA：LLaVA-1.5-7B 62.0→63.5；Qwen2-VL-7B 64.5→64.7。 Table4 不做 image-centric head 筛选而对所有 heads 重分配，POPE/MME/MM-Vet 三项均退化到0.0。

**限制与反证。** sink 判定依赖架构中的异常 hidden 维度；不能把所有 sink 或所有 heads 视为有害。选择和校准影响收益。

**与本项目的差异。** 直接可比的冻结内部干预基线；需要完整 key 归一化和 identity 检查。我们的跨模型干预应报告 head选择、全行质量、失败样本与负效应。

<a id="p02"></a>

### P02. RAR: Reversing Visual Attention Re-Sinking for Unlocking Potential in Multimodal Large Language Models

**会议与读取范围：** ICLR 2026；§4–6; §6.1; results tables; appendix。来源：[正式接收页面](https://proceedings.iclr.cc/paper_files/paper/2026/hash/42a1f9ef76af656f3fe33580ab794351-Abstract-Conference.html)；[论文PDF](https://eprints.gla.ac.uk/382078/1/800_RAR_Reversing_Visual_Atten.pdf)。本地：`outputs/literature_internal_mechanism_v2/02_rar.pdf` / `02_rar.txt`。

**实际张量/算子。** 用最大视觉 attention 的两成分 GMM 谷值区分 vision/sink heads；对非视觉 keys 子矩阵逐行归一化，再跨 query 平均得到 p_j，计算熵并再次用 GMM 分出高熵 global sink 与低熵 specific sink。保留 vision、global sink 和共享 head0，对晚层其余头稀疏化。

```text
p_j=mean_i normalize_keys(A[i,I])_j; H=−∑_j p_j log p_j.
```

**模型与参数。** Qwen2.5-VL-3B/7B/32B，InternVL2-2B，LLaVA-1.5-7B。Qwen3B从层20、Intern2B从层15、LLaVA7B从层20干预。 2B/3B 等是作者型号；本阅读未独立计数全模型参数，不能当成与本地2.13B精确等量。

**训练数据与计算。** 主要结果使用670k SFT 样本，来源 RefCOCO、Dcube、Visual Genome、GQA、OCRVQA、TextVQA、CLEVER。参数无新增不等于无训练。 已查实验设置未确认 GPU型号/总 GPU-hours，记作未核实，不填估值。

**可核查结果。** GQA Qwen3B：baseline60.4、SFT62.0、RAR64.2；Intern2B：55.6、56.9、59.0。 §4.1 另有最后五层重分配的训练自由诊断，平均+0.74pp，不能代替主方法训练设置。

**限制与反证。** 高熵 sink 可能承担全局整合；去除所有 sink 会破坏功能。GMM/阈值/层边界需要模型适配。

**与本项目的差异。** 重要机制证据及训练预算明确的强对照，不能放进纯训练自由比较列并引用主表增益。标题必须用本条正式标题；不将其他标题或版本重复计数。

<a id="p03"></a>

### P03. Reallocating Attention Across Layers to Reduce Multimodal Hallucination

**会议与读取范围：** CVPR 2026；§3.1–3.2; final Eq.6–10; Table1; experimental settings。来源：[正式接收页面](https://openaccess.thecvf.com/content/CVPR2026/html/Lu_Reallocating_Attention_Across_Layers_to_Reduce_Multimodal_Hallucination_CVPR_2026_paper.html)；[论文PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Lu_Reallocating_Attention_Across_Layers_to_Reduce_Multimodal_Hallucination_CVPR_2026_paper.pdf)。本地：`outputs/literature_internal_mechanism_v2/03_layer_reallocation.pdf` / `03_layer_reallocation.txt`。

**实际张量/算子。** 按视觉 attention 质量比、浅/深层边界和阈值挑选 perception/reasoning heads。把整个 head context A_hV_h 乘 g_h≥1 后拼接，经 W_O 输出；attention 概率 A 和 V投影本身不变。不是只增加 visual-value 分量。

```text
Y′=Concat_h(g_h A_h V_h)W_O; unselected heads: g_h=1.
```

**模型与参数。** Kimi-VL-A3B-Thinking、OceanR1-7B、R1-Onevision-7B。 Kimi A3B 是激活参数约3B，绝非全模型总参数3B；另外两个7B标签也未独立核计视觉总量。

**训练数据与计算。** 无需权重训练，但需校准层边界、视觉质量阈值和增益。Ocean示例 g_reason1.30/τ_reason.01，g_perception1.16/τ_perception.22。 Ocean用A40，其他模型用A800。未核实总GPU-hours；不把早期预印本的50样本说法写成最终版事实。

**可核查结果。** Table1 MathVista：Kimi63.48→69.78；Ocean54.58→59.32。

**限制与反证。** Eq7 用单个 query i*，附近文字又说 query平均，移植需明确定义。不能把概念性论证视作误差必然下降的定理。

**与本项目的差异。** 可在冻结模型中实现的 head-output gain；与残差/FFN干预区分。需独立验证集选参，等强度随机头、反向gain和identity作对照。

<a id="p04"></a>

### P04. Enhancing Multi-Image Understanding through Delimiter Token Scaling

**会议与读取范围：** ICLR 2026；§3–5; Eq2; Table1; §7; Appendix A.3 and Tables A2/A5; official author code。来源：[正式接收页面](https://proceedings.iclr.cc/paper_files/paper/2026/hash/aebf6284fe85a8f44b4785d41bc8249a-Abstract-Conference.html)；[论文PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/aebf6284fe85a8f44b4785d41bc8249a-Paper-Conference.pdf)。本地：`outputs/literature_internal_mechanism_v2/04_delimiter.pdf` / `04_delimiter.txt`。

**实际张量/算子。** 论文 Eq2 对真实 image delimiter tokens 的 hidden乘λ>1。作者 Qwen2.5/InternVL3 代码在 decoder block输出后缩放 residual stream；Qwen token为vision_start/end，Intern为<img>/</img>。代码只在开始/结束token数分别>1时执行；Intern还要求序列长度>1。

```text
h_t^(l)←λ h_t^(l), t∈D. Public Qwen2.5-VL-3B Mantis command: λ=8, zero-based layers 0,1,2,3.
```

**模型与参数。** Qwen2.5-VL3B/7B/32B，InternVL3-1B/2B/8B/14B，LLaVA-OneVision0.5B/7B；更大模型另做实验。 额外Hub metadata核查：InternVL3-1B 938,193,024总tensor参数，2B 2,088,957,440；LLaVA-OV0.5B 893,675,552。来源为指定Hub revision的声明，非本地权重审计或推理验证。

**训练数据与计算。** 无需权重训练。Appendix A.3 多图用10% test作validation；多文档/多表直接在test调参。层在同模型各benchmark固定，λ可调。 作者按模型用A5000/A6000/H200。未下载新权重，也未把本地8GB能否运行当作已证实。

**可核查结果。** Qwen3B Mantis59.91→63.13、Muir37.31→42.42。 InternVL3-2B Muir27.69→27.65；InternVL3-1B QBench2 50.80→50.20；并非每格提升。

**限制与反证。** §7明确缺少帧delimiter的视频不适用。公开README只给Qwen3B Mantis λ8/层0–3，未找到InternVL3-1B最优层/λ；CLI默认禁用。pre-RMSNorm使正标量近似被归一化抵消，不能自动推出Q/K幅值放大。

**与本项目的差异。** 最优先训练自由基线，但需固定真实token、层、prefill行为。Smol/Qwen3或中层/.5-2-4属移植与消融，不能称原版复现。理论解释转向相对residual更新并用实际QKV测量检验。

<a id="p05"></a>

### P05. Unveiling Visual Perception in Language Models: An Attention Head Analysis Approach

**会议与读取范围：** CVPR 2025；attention scoring method; §5; main results; supplement。来源：[正式接收页面](https://openaccess.thecvf.com/content/CVPR2025/html/Bi_Unveiling_Visual_Perception_in_Language_Models_An_Attention_Head_Analysis_CVPR_2025_paper.html)；[论文PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Bi_Unveiling_Visual_Perception_in_Language_Models_An_Attention_Head_Analysis_CVPR_2025_paper.pdf)。本地：`outputs/literature_internal_mechanism_v2/05_visual_heads.pdf` / `05_visual_heads.txt`。

**实际张量/算子。** 从最后输入位置/首个输出query读完整attention；组合视觉区域质量、集中度及层权重来识别视觉头。集中度由token分布熵定义，不能用只对视觉子集重新归一化的分数替代原口径。

```text
Concentration C=1−H(α)/log₂(N+ε); score additionally combines regional attention mass and layer weighting.
```

**模型与参数。** 九个LLaVA/VIP变体，包括Phi3-3.8B及7B/8B/13B骨干。 型号为名义语言骨干尺寸；没有独立全模型唯一参数统计。

**训练数据与计算。** 诊断/删除heads，无新的模型权重训练；PointQA约5700样本，控制prompt，并扩展MMBench/POPE/SEED/QBench。 阅读的main/supp未确认GPU型号及GPU-hours。

**可核查结果。** 早层top20视觉头删除比随机删除破坏更大；晚层可删约100heads而影响较小。 同家族头模式相关性>0.8，跨家族<0.1，不能移植固定head编号。

**限制与反证。** 关键head删除导致损失只说明模型对该路径敏感，不能推出增强该head必然改善正确率。

**与本项目的差异。** 用于设计选头统计和必要性消融；不是已有原生QA增益基线。跨模型对齐应按功能/相对深度而非同一headID。

<a id="p06"></a>

### P06. Probing Rotary Position Embeddings through Frequency Entropy

**会议与读取范围：** ICLR 2026；frequency-entropy definitions; intervention; downstream table; Appendix G。来源：[正式接收页面](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0aee38a6fe9fffc8b658cfb1d872c1d5-Abstract-Conference.html)；[论文PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/0aee38a6fe9fffc8b658cfb1d872c1d5-Paper-Conference.pdf)。本地：`outputs/literature_internal_mechanism_v2/06_frequency_entropy.pdf` / `06_frequency_entropy.txt`。

**实际张量/算子。** 沿token位置记录RoPE二维pair范数序列。STFT窗1024/hop512、长度4096产生7窗；平均power归一化后计算频率熵。SequenceFE采用去DC的正频DFT。低熵pair的Q/K旋转分量乘α。

```text
q*_(m,j)=α_(l,h,j) R_(m,j)q_(m,j), similarly K, for pairs selected by entropy threshold.
```

**模型与参数。** Llama4-Scout17B-16E、Llama3-8B、Qwen3-8B、Gemma2-9B；另有自训练小模型。 Scout17B是激活规模，不是总模型参数；其他标签也不等于本地精确总量。

**训练数据与计算。** 主Wikitext103干预不finetune、扫描阈值。Appendix G另训练16层/d1024/8head模型约286k更新/205epochs，不能说全篇都无训练。 未确认GPU-hours；主推理和附录预训练预算必须分列。

**可核查结果。** 主语言建模PPL可改善；下游准确率无显著增益例：Qwen3 Winogrande58.94→58.92，其余两项不变。

**限制与反证。** 幅度乘α不改变RoPE角频率或相位速度；paper叙述不能替代数学。token轴不等于视频时间轴，也不支持长期多模态记忆结论。

**与本项目的差异。** 给出Q/K pair统计工具；本地必须保留真实时序轴并区分token频谱、空间FFT和协方差特征谱。原生QA提升需单独证明。

<a id="p07"></a>

### P07. Map the Flow: Revealing Hidden Pathways of Information in VideoLLMs

**会议与读取范围：** ICLR 2026；causal knockout method; main pathway experiments; Table3; Appendix G.7; training settings。来源：[正式接收页面](https://proceedings.iclr.cc/paper_files/paper/2026/hash/537b3416a0c8035794a12b3fa596f941-Abstract-Conference.html)；[论文PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/537b3416a0c8035794a12b3fa596f941-Paper-Conference.pdf)。本地：`outputs/literature_internal_mechanism_v2/07_map_flow.pdf` / `07_map_flow.txt`。

**实际张量/算子。** 在softmax之前对指定query-key边设−∞，以9层滑窗追踪路径。早中层视觉跨帧，中层视频→问题，晚层问题→最后token；这些是经干预的功能路径而非attention图的直接语义解释。

```text
Attention(Q,K,V)=softmax(QKᵀ/√d+M)V; selected M_ij=−∞.
```

**模型与参数。** LLaVA-NeXT7B/13B、MiniInternVL4B、VideoLLaMA3-7B。 7B/4B等为论文名义规格，未独立核计总参数。

**训练数据与计算。** 其视频模型训练设置为VideoChat2-IT874k、3epochs；冻结vision，训练projector+LLM。因果分析主要筛选baseline答对样本。 A6000；未核实完整GPU-hours。

**可核查结果。** Table3 LLaVA7B保留42% attention边：TVBench51.5→51.2、TOMATO30.2→29.2。 VideoLLaMA3保留58%边：TVBench55.2→57.2。

**限制与反证。** 正确样本筛选影响外推；保留边比例不等于实际kernel加速，也非所有模型/数据都涨分。Appendix G.7局部概率上升不是完整新测试集的修复结果。

**与本项目的差异。** 直接指导内部路由干预和路径层定位；我们的验证应包含全部预冻结样本、native答案、latency/VRAM实测和失败率。

<a id="p08"></a>

### P08. Temporal Reasoning Transfer from Text to Video

**会议与读取范围：** ICLR 2025；representation probes; text training methods; main task ablations; training appendix。来源：[正式接收页面](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b7eecb72574b043ad0c69ea296212450-Abstract-Conference.html)；[论文PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/b7eecb72574b043ad0c69ea296212450-Paper-Conference.pdf)。本地：`outputs/literature_internal_mechanism_v2/08_t3.pdf` / `08_t3.txt`。

**实际张量/算子。** 比较视觉编码器和decoder最终特征中可被probe读出的事件信息，再用按顺序的文字caption合成QA训练原VLM。任务涵盖distractor、顺序、属性、指代与grounding。外部LSTM读出并不说明原模型实际使用该信息。

```text
Autoregressive supervised QA loss on caption sequences; no temporal FFT operator.
```

**模型与参数。** LongVA7B、VILA8B用于probe；LongVA7B和Qwen2-VL72B用于迁移实验。 7B/8B/72B为名义模型尺寸，没有本文独立总参数审计。

**训练数据与计算。** 任务探索22k；综合200k文字QA，与原instruction按1:2混合，一epoch。不是adapter-only；学习率vision2e−6、其余1e−5，纯文字样本不做视觉前向。 8×H100，综合训练≤5小时；这不是本地8GB训练配置。

**可核查结果。** TempCompass全任务55.9→61.2；仅grounding训练55.9→41.0。

**限制与反证。** 任务构造和原instruction混合影响迁移及原视觉能力；数据训练贡献不能归因于频率或单一内部算子。

**与本项目的差异。** 强证据表明视觉内容可读出与native推理可用之间有差距。可作训练型上界，不能替代同prompt/同输入的冻结内部修复验证。

<a id="p09"></a>

### P09. Frame2Freq: Spectral Adapters for Fine-Grained Video Understanding

**会议与读取范围：** CVPR 2026；method and architecture; main action-recognition tables; Supplement A.1 and training details; local axis audit。来源：[正式接收页面](https://openaccess.thecvf.com/content/CVPR2026/html/Ponbagavathi_Frame2Freq_Spectral_Adapters_for_Fine-Grained_Video_Understanding_CVPR_2026_paper.html)；[论文PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Ponbagavathi_Frame2Freq_Spectral_Adapters_for_Fine-Grained_Video_Understanding_CVPR_2026_paper.pdf)。本地：`outputs/literature_internal_mechanism_v2/09_frame2freq.pdf` / `09_frame2freq.txt`。

**实际张量/算子。** 输入X[B,T,N,D]，按空间位置/通道沿T做复数STFT；FCdown→STFT→频域及时间卷积→iSTFT→FCup残差。多尺度版拆通道、用多窗口后concat。附录对内部embedding沿时间FFT，以功率和类别ANOVA计算频段区分性。

```text
P_i(f)=mean_d |FFT_t(X_i)[f,d]|²; adapter uses iSTFT(Conv(STFT(FCdown(X)))).
```

**模型与参数。** 冻结CLIP或DINOv2 ViT-B/16、ViT-L/14骨干；分类head与adapter训练。 多尺度adapter约7.3M可训练参数，吞吐表314GFLOPs；不是7.3M全模型，也不是native VLM QA系统。

**训练数据与计算。** 五个动作识别数据集；60epochs、batch32、lr1e−3，训练adapter与分类head。 小数据4×A100，SSv2/Diving48用16×A100；实际总GPU-hours未核实。

**可核查结果。** Diving48 ST-Adapter90.4→Frame2Freq-MS92.2，+1.8pp。

**限制与反证。** 已有内部频谱诊断，不能说其只做频率模块而未分析hidden。类别监督频段区分性不是无监督因果证据；离散频率不自动是物理Hz。作者代码轴/布局疑点详见已有frame2freq_axis_audit。

**与本项目的差异。** 频域方向最直接前作；若继续频率路线，需原生VLM推理、阶段/跨模型复现和同容量时域控制，不能只复写STFT adapter。

<a id="p10"></a>

### P10. Language-guided Frequency Modulation for Large Vision-Language Models

**会议与读取范围：** CVPR 2026；method Eq11–13; §4.1; Tables1–2; appendix implementation。来源：[正式接收页面](https://openaccess.thecvf.com/content/CVPR2026/html/Ouyang_Language-guided_Frequency_Modulation_for_Large_Vision-Language_Models_CVPR_2026_paper.html)；[论文PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Ouyang_Language-guided_Frequency_Modulation_for_Large_Vision-Language_Models_CVPR_2026_paper.pdf)。本地：`outputs/literature_internal_mechanism_v2/10_lfm.pdf` / `10_lfm.txt`。

**实际张量/算子。** 按组平均vision中间层，对空间轴FFT得到F_k；额外CLIP text encoder产生语言特征T，用cos相似度调制频谱，再把IFFT后的结果和FFN后视觉特征混合，组间concat→2层MLP。这里是空间频率。

```text
S_k=cos(F_k,T); F̂_k=F_k⊙(1+α_kS_k); fuse with β Re(IFFT(F̂_k)).
```

**模型与参数。** Phi2-2.7B、Vicuna7/13B、Llama3-8B、Yi34B及InternVL2.5-1B/8B、Qwen2.5-VL7B。 模型标签不含可直接比较的完整精确总参数；额外text encoder、多层vision融合和训练成本需计入。

**训练数据与计算。** 指令微调更新全部参数。预训练一epoch、SFT一epoch；表列0.5M+0.6M或1.2M+1.5M等配方，正文数据命名与数量有歧义，不虚构每模型统一训练集。 NVIDIA A6000 48GB；未核实每模型GPU数/总GPU-hours。

**可核查结果。** InternVL2.5-1B：MMB70.7/66.3→73.1/68.3，MMStar48.8→52.3。 匹配CLIP/Vicuna7B配置的Table2 GQA62.0→64.6。

**限制与反证。** 不是仅训练轻量MLP的冻结插件；语言条件、训练数据、多vision层等混杂因素不能全部归功于频域。

**与本项目的差异。** 文献矩阵必须区分空间/时间频率。我们的无训练内部干预需与其全训练结果分栏，并验证同容量非频率对照。

<a id="p11"></a>

### P11. Too Late to Recall: Explaining the Two-Hop Problem in Multimodal Knowledge Retrieval

**会议与读取范围：** NeurIPS 2025；two-hop dataset; Grad×Delta attribution; MLP transplantation; appendices and linear probes。来源：[正式接收页面](https://papers.neurips.cc/paper_files/paper/2025/hash/4526cfacdbca6b6e184568dac91bf070-Abstract-Conference.html)；[论文PDF](https://papers.neurips.cc/paper_files/paper/2025/file/4526cfacdbca6b6e184568dac91bf070-Paper-Conference.pdf)。本地：`outputs/literature_internal_mechanism_v2/11_too_late.pdf` / `11_too_late.txt`。

**实际张量/算子。** 比较图像→实体→知识的VLM与已获实体名的纯文本LLM。对clean/corrupted hidden作Grad×Delta归因；把文本LLM早层实体MLP activation移植到VLM所选位置，另测同模型backpatch。

```text
Attribution=∑_d [∂KL(clean||corrupted)/∂H_corr,d]·(H_clean,d−H_corr,d).
```

**模型与参数。** 14组VLM/LLM骨干；重点LLaVA7/13B、LLaVA-MORE8B；Qwen2.5-VL7B/Gemma3-12B为较低退化组。 名义骨干尺寸，不等于包含vision后的精确总数；donor LLM的额外存储与前向也需计入。

**训练数据与计算。** 15k WIT+GPT4.1构造QA，剔除识别错误，取每模型1000有效样本；归因100答对样本；patching100 VLM错/LLM对样本。无VLM权重训练，另训练ImageNet50类/2500图线性probe。 A100 80GB；没有据此推算本地可行性或完整GPU-hours。

**可核查结果。** MLP transplantation恢复35%的性能差距，不是全测试集准确率+35pp。 backpatch约13%差距恢复，random约16%；不能只报最优移植列。

**限制与反证。** donor得到真实实体文本，属于额外oracle信息；筛选样本偏差明确。外部probe读出与原生推理能力不同；CoT改变harness。

**与本项目的差异。** 最接近FFN瓶颈→内部因果干预的论证模板。我们应在无答案/无entity oracle条件下识别FFN干扰，做随机/跨样本patch和原生答案验证。

<a id="p12"></a>

### P12. NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks

**会议与读取范围：** ICLR 2026；FFN pre/post definitions; covariance metrics; architecture/optimizer experiments; compute appendix。来源：[正式接收页面](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2764bbdbd9f36d9f1ec63f4ad22b24ff-Abstract-Conference.html)；[论文PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/2764bbdbd9f36d9f1ec63f4ad22b24ff-Paper-Conference.pdf)。本地：`outputs/literature_internal_mechanism_v2/12_nerve.pdf` / `12_nerve.txt`。

**实际张量/算子。** 非门控FFN比较W_up x+b与σ(pre)；SwiGLU比较W_gate x与σ(W_gate x)⊙W_up x。把[B,S,D]展成[BS,D]计算协方差特征值的熵/参与率/JS距离；已丢弃token顺序，不能当成时间频谱。

```text
Σ=(X−μ)ᵀ(X−μ)/(N−1); p_i=λ_i/∑λ; SE=−∑p_i log p_i; PR=(∑λ_i)²/∑λ_i².
```

**模型与参数。** GPT2-125M，Llama71M–1.3B从头训练；另有160M/350M优化器对比。 这里明确研究小型语言模型，非已验证的冻结VLM；型号容量不证明可对等迁移到多模态模型。

**训练数据与计算。** CodeParrot2.1B tokens/41k steps；Llama约1.1–2.2B tokens；另RoPE研究26B tokens。 24GB RTX3090；RoPE实验4GPU。总训练GPU-hours未独立核实。

**可核查结果。** RoPE vs NoPE实验PPL15.20 vs16.78；包含多个架构/优化器相关性比较。

**限制与反证。** eigenspectrum是特征协方差谱，非FFT；相关性不是冻结模型内FFN干预提高QA的证明。token池化也不构成独立样本统计。

**与本项目的差异。** 可移植的pre/post FFN统计（用Gram节省内存），但必须与native错误/修复因果配对。不能以SE高低直接判定干扰或选择需要增强的方向。

<a id="p13"></a>

### P13. Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding

**会议与读取范围：** CVPR 2025；head selection Eq3; grounding readout; main tables; supplement implementation。来源：[正式接收页面](https://openaccess.thecvf.com/content/CVPR2025/html/Kang_Your_Large_Vision-Language_Model_Only_Needs_A_Few_Attention_Heads_CVPR_2025_paper.html)；[论文PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Kang_Your_Large_Vision-Language_Model_Only_Needs_A_Few_Attention_Heads_CVPR_2025_paper.pdf)。本地：`outputs/literature_internal_mechanism_v2/13_localization_heads.pdf` / `13_localization_heads.txt`。

**实际张量/算子。** 最后文本query→视觉attention总质量阈值选头。视觉map按均值二值化、8连通分量，按分量面积分布计算spatial entropy；每样本选10低熵头，再按频次取最终3头，排除最前2层。不是普通token熵。

```text
H_spatial=−∑_i p(C_i)log p(C_i), p(C_i)=area(C_i)/∑_j area(C_j).
```

**模型与参数。** 10个LVLM，包括DeepSeek-VL1.3B、MiniGemini2B、其他6B/7B/13B。 1.3B/2B为模型标签；RES另用SAM，不能不计其参数/计算。

**训练数据与计算。** 无需权重训练；使用1000 RefCOCO train图文pair选头。Gaussian attention map产生box，SAM仅用于segmentation。 Supplement报告A6000 48GB；未核实总GPU-hours。

**可核查结果。** DeepSeek-VL1.3B RefCOCO val：REC73.2，RES56.3（后者含SAM）。

**限制与反证。** 输出是外部grounding readout，不是修改decoder后原生QA涨分；训练集选头可迁移程度需要测量。

**与本项目的差异。** 小模型也可存在稀疏视觉路径；用作head诊断和空间控制。不能把外部box/SAM成绩当成长期视频理解或内部干预收益。

## 3. 对实验设计的直接约束

1. **先分清算子作用位置。** VAR改softmax概率；head reallocation改AV后的head输出；delimiter改block后的residual；Too Late patch的是MLP activation；NerVE统计的是非线性前后特征。把这些统称attention增强会掩盖不同机制。[P01,P03,P04,P11,P12]
2. **测量不等于修复。** 删除高重要性head、从hidden训练probe、发现频谱差异，都不足以证明native答案会改善。必须在预冻结样本上执行内部干预，记录identity、随机/反向控制、基线答对→答错和答错→答对。[P05,P07,P08,P12,P13]
3. **FFN干扰应有可证伪定义。** 可先测 attention update 与 FFN update 的范数、夹角、对任务logit的局部作用及跨层稳定性；任何低相似度或高谱熵都只是候选统计。随后用不依赖正确答案的更新抑制/替换干预验证，避免使用oracle实体或按错误选样再宣称全数据提升。[P11,P12]
4. **训练预算、标签访问和输入条件一致。** RAR/T3/LFM/Frame2Freq与冻结基线分栏；Too Late的donor实体名、grounding中的SAM、额外CLIP text encoder均要显式记录。多种方法若读过同一个test再选最优，只能作探索结果。[P02,P04,P08–P11,P13]
5. **频谱必须标轴。** Frame2Freq是时间FFT/STFT，LFM是空间FFT，Frequency Entropy是token位置频率，NerVE是协方差特征谱；四者不是同一种频域。真实帧PTS、重采样、非均匀时间和窗口长度决定能否谈物理Hz。[P06,P09,P10,P12]
6. **跨模型需功能对齐。** 固定headID、tokenID和绝对层号不可直接跨家族复用；只能在真实tokenizer/config和明确tensor边界上移植。名称相近也不能忽略Q/K norm、RMSNorm位置、shared embeddings及MoE active/total差异。[P03–P05,P13]

## 4. 当前最值得验证的假设（不是论文已证结论）

第一优先是 **分隔/摘要token的相对更新抑制**：post-block residual正比例缩放可能主要改变后续attention/FFN更新相对旧状态的大小。应直接测缩放后、下一层norm前后、Q/K/V及各residual update，而不是按论文示意图预设Q/K一定立即放大。数学命题及先行工作另见 `docs/delimiter_scaling_mathematics_v2.md`（由主任务维护）。

第二优先是 **FFN把有用视觉证据推离可读出方向**：Too Late提供时机与MLP patching先例，NerVE提供pre/post非线性统计，但两者不证明当前视频模型存在相同故障。跨模型重现、无oracle选择与冻结测试改善三项缺一不可。

第三优先是 **head选择错误而非视觉质量总量不足**：VAR全头0分反例与RAR的global sink结果要求谨慎。只增加视觉attention或删除sink不能作为普遍安全干预。

## 5. 可追溯文件

- `docs/internal_mechanism_reading_sources_v2.json`：13篇结构化方法/结果记录、正式来源、PDF与文本SHA256、补充材料及模型metadata来源。
- `outputs/literature_internal_mechanism_v2/source_downloads.json`：原始论文下载/缓存记录；RAR为作者机构PDF，会议身份由ICLR正式页面确认。
- `outputs/literature_internal_mechanism_v2/support_downloads.json`：三个补充材料、官方代码和模型metadata（不含权重）。
- `docs/delimiter_scaling_implementation_audit_v2.md`：作者固定commit、tokenID、hook、默认参数公开程度及移植边界。
- `docs/internal_routing_data_selection_v2.md`：独立32视频/159题的冻结选择与完整性审计；本文不包含这些题的模型分数。

