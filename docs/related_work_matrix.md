# Spectral Memory Probe：近期正式论文对照矩阵

核查截止：2026-11-01。主矩阵只收录本轮直接读取正式论文集页面确认的 **22 篇 2025–2026 年 CVPR、ICCV、ICLR、NeurIPS 论文**。尚未独立确认最终会议身份的条目移入附录，不用 arXiv 的 “accepted” 声明替代正式会议证据。ICML 候选 XSkill 暂不进入主表；这不表示它未被录用。

研究问题：在未来问题未知、实际持久化存储预算固定的条件下，按视觉隐状态的时间轨迹选择少量真实视觉证据，能否帮助小 VLM 恢复短暂状态、方向及事件顺序？

计数口径：记忆方向主矩阵22篇；另有第10节独立FFN机制分区6篇，合计28篇近期正式论文。FFN分区不改变原主表的身份与编号，且不把参数事实召回论文归类为已验证的视频长期记忆系统；其他会场、经典和待确认项仍为9篇附录。

## 1. 核查与阅读边界

- **会议身份**：直接读取 CVF、ICLR proceedings、NeurIPS proceedings 的标题和会议元数据；对 CVPR 2026、ICLR 2025/2026、NeurIPS 2025 的新增重点项另核对官方总目录。来源记录见 [source audit](related_work_source_audit.json)。
- **机制证据**：主表机制至少来自正式摘要。Frame2Freq、FreqKV、FlashCache、FluxMem、OASIS、AKS、LSDBench/RHS 另读取正式 PDF 的方法部分；其余项不能据本表理解为已完成逐页全文审查。
- **实现证据**：`C1` 表示本轮访问作者关联仓库并读取 README，核对其安装、训练或推理说明；`C0` 表示本轮未确认可执行的作者实现，不等于证明没有代码。公开权重链接不等于权重已下载、许可已完整核验或结果已复现。
- **复现状态**：本表所有第三方方法均未在本项目运行。没有把本地最小实现冒充原论文复现，也没有把论文的成绩写成本项目结果。
- **预算口径**：GPU 活跃 KV 上限、总持久化存储、输入视觉 token、写入临时工作区是不同资源。不能只匹配其中一项就宣布公平比较。
- 保留旧稿 `R01–R18` 编号以兼容其他文档引用；编号不代表会议优先级。新增正式项为 `R19–R31`。

## 2. 主矩阵 A：频域表征与压缩

| ID / 正式论文 | 正式身份与原始来源 | 已核实机制与频率轴 | 对当前候选的具体影响 | 公开代码及复现条件 |
| --- | --- | --- | --- | --- |
| **R02 Frame2Freq: Spectral Adapters for Fine-Grained Video Understanding** | **CVPR 2026**，[CVF正式页](https://openaccess.thecvf.com/content/CVPR2026/html/Ponbagavathi_Frame2Freq_Spectral_Adapters_for_Fine-Grained_Video_Understanding_CVPR_2026_paper.html)；pp.24073–24083 | 冻结图像骨干后训练 adapter 和分类头；沿时间做 FFT，学习频带特定变换，补充中等速度运动信息。正式 PDF 明确有时间域与频谱路径，不应将它简化为只用功率谱。 | “时间频谱帮助细粒度、近对称动作区分”已有先例。当前候选若成立，区别是**无任务标签写入、真实帧锚点、未来问答效用**，而非分类适配。必须对照相同轨迹上的时域重建和频带加权；不能仅凭任务换成记忆声称新机制。 | **C1**：[作者仓库](https://github.com/th-nesh/Frame2Freq)，README含依赖、CLIP B/16与L/14权重配置及训练说明。需训练适配器、准备动作数据；不是可直接接入2B VLM的无训练记忆模块。 |
| **R03 FreqKV: Key-Value Compression in Frequency Domain for Context Window Extension** | **ICLR 2026**，[正式页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ac4c13a5c6ec1bba68c9ce1d908321d2-Abstract-Conference.html)；已从官方总目录确认 | 沿 **token序列轴**对 K/V 做 DCT，保留频域信息并迭代压缩增长的缓存。压缩算子无参数，但扩窗实验还包含短上下文训练，不能把全部实验叫作零训练。 | 频域压缩和长上下文已明确存在。当前候选选择**外部真实frame-pair**而不修改KV；频带相对重建损失也不同于直接低频压缩。这些是实现差异，仍需证明动态证据效用，不能声称首次频域记忆。 | **C1**：[作者仓库](https://github.com/LUMIA-Group/FreqKV)，README列出训练、SFT、LoRA合并、评测脚本，涉及Llama-2-7B及FlashAttention。迁移到视觉2B模型需要适配，不是本项目已实现的基线。 |
| **R19 FlashCache — Revisiting Multimodal KV Cache Compression: A Frequency-Domain-Guided Outlier-KV-Aware Approach** | **CVPR 2026**，[CVF正式页](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Revisiting_Multimodal_KV_Cache_Compression_A_Frequency-Domain-Guided_Outlier-KV-Aware_Approach_CVPR_2026_paper.html)；pp.39550–39560 | DCT低通与逆变换得到平滑 **Base KV**；按原KV相对Base KV的偏离保留Outlier KV，并用频域残差信息跨层分配预算。对象是多模态KV矩阵，论文实验含Qwen2.5-VL-7B和LLaVA-OneVision-1.5-8B。 | **最高风险机制重合**：频率残差、保留重要视觉信息、动态预算均已有。当前的分频带相对轨迹重建不是逐KV低通残差排序；但应加入同表征的低通残差/差分选择对照，证明收益不只是“保留变化大的位置”。 | **C0**：正式页、正文和所查作者摘要未提供本轮可验证的实现入口。不能因仓库未确认而省略方法比较；若自行实现应标“受其启发的简化基线”，不能标原方法复现。 |
| **R20 Language-guided Frequency Modulation for Large Vision-Language Models（LFM）** | **CVPR 2026**，[CVF正式页](https://openaccess.thecvf.com/content/CVPR2026/html/Ouyang_Language-guided_Frequency_Modulation_for_Large_Vision-Language_Models_CVPR_2026_paper.html)；pp.39175–39185 | 在语言引导下调制视觉频率，区分空间全局结构与局部细节，再交给语言模型；摘要讨论轻量投影。这里主要是**视觉空间频率**，不是物理时间频率。 | 不能笼统声称VLM此前没有显式频域机制；也不能把空间高频等同短暂事件。查询未知的写入与语言已知的调制应分开评价。其训练细节尚未全文核清，不把“plug-and-play”转述为完全无训练。 | **C0**：本轮未核实作者可执行仓库。只作为空间频域相关边界，不列为现成2B时间记忆实现。 |

## 3. 主矩阵 B：视觉记忆、流式压缩与时序证据

| ID / 正式论文 | 正式身份与原始来源 | 已核实机制 / 预算边界 | 当前候选必须体现的差异与公平对照 | 公开代码及复现条件 |
| --- | --- | --- | --- | --- |
| **R06 ReWind: Understanding Long Videos with Instructed Learnable Memory** | **CVPR 2025**，[CVF正式页](https://openaccess.thecvf.com/content/CVPR2025/html/Diko_ReWind_Understanding_Long_Videos_with_Instructed_Learnable_Memory_CVPR_2025_paper.html) | learnable queries与cross-attention读写指令相关动态记忆；记忆再指导高分辨率帧选择。 | “记忆引导选择真实帧”已有先例。写入时是否已知指令是关键条件；当前未来查询未知的写入不能与提前知道问题的方法直接并列而不说明。 | **C0**：本轮核实正式论文，但未确认作者实现的可运行入口。学习型方法需训练预算，不能拿未训练的结构近似当公平复现。 |
| **R24 Streaming Video Question-Answering with In-context Video KV-Cache Retrieval（ReKV）** | **ICLR 2025**，[正式页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/67a9b444cbcd647572c88194619f72d5-Abstract-Conference.html) | 滑窗编码，将历史KV移到RAM/磁盘，问题出现后检索并回载相关KV；可分离视频分析和问答。 | 其GPU活跃缓存受控，但历史RAM/磁盘KV仍要计费。当前若宣称固定总存储，必须展示真实证据包字节数；不能仅凭峰值显存较低就胜过ReKV。 | **C0**：正式摘要和PDF已读，作者仓库本轮未核实。搜索候选仓库返回404，不能据搜索摘要写“代码可复现”。 |
| **R25 InfiniPot-V: Memory-Constrained KV Cache Compression for Streaming Video Understanding** | **NeurIPS 2025**，[正式页](https://papers.neurips.cc/paper_files/paper/2025/hash/caef5f5e658aa1f7565f063a2cd99726-Abstract-Conference.html)；已从正式总目录确认 | 无训练、query-agnostic，缓存达到阈值即压缩；Temporal-axis Redundancy（TaR）去除时间冗余，Value-Norm（VaN）保留重要token；明确面向长度无关的硬缓存上限。 | **“未来问题未知+固定预算+流式”已被覆盖**。当前须验证频带均衡的轨迹重建比时间冗余/幅值排序更能保留短暂状态，而不是只证明自己也能限制缓存。还要分开外部帧存储和KV缓存预算。 | **C0**：正式论文存在已确认；本轮未取得可验证的作者代码入口。不能把第三方搜索到的仓库名当作公开代码已经可运行。 |
| **R07 Memento: Toward an All-Day Proactive Assistant for Ultra-Long Streaming Video** | **ICLR 2026**，[正式页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3b5f4587a0bdb81ecc6ce9d82320a5c2-Abstract-Conference.html) | Dynamic Memory、Query-related Memory Selection和Step-Aware Memory Attention；训练主动响应与长历史访问；Memento-54K/MementoBench最长视频达7小时。 | 稀疏记忆、查询检索和小时级测试已有。当前小VLM实验需分别报告写入选择、读出能力和证据年龄，不能拿短片功能样例代替小时级验证。 | **C1**：[作者仓库](https://github.com/XiangTodayEatsWhat/Memento)，`master` README列出Memento-8B adapter、训练/评测说明和数据入口。基座Llama-3.1-8B、SigLIP及Ego4D源视频需另取；标注开源不代替源视频许可。 |
| **R23 StreamingVLM: Real-Time Understanding for Infinite Video Streams** | **ICLR 2026**，[正式页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6445dd88ebb9a6a3afa0b126ad87fe41-Abstract-Conference.html) | attention sinks、短视觉窗口、长文本窗口构成紧凑KV；用短重叠片段SFT对齐流式推理。Inf-Streams-Eval平均视频超过2小时，要求逐秒视觉文字对齐。 | 窗口稳定运行与旧事件细节恢复不同。当前候选应证明远期视觉状态可恢复，不能仅凭连续运行不爆显存称长期记忆改善。训练过的流式模型和冻结2B reader也需分开。 | **C1**：[作者仓库](https://github.com/mit-han-lab/streaming-vlm)，实际README文件名是`readme.md`；已核实推理入口、训练数据和评测说明。未下载权重或运行。 |
| **R21 FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding** | **CVPR 2026**，[CVF正式页](https://openaccess.thecvf.com/content/CVPR2026/html/Xie_FluxMem_Adaptive_Hierarchical_Memory_for_Streaming_Video_Understanding_CVPR_2026_paper.html) | 无训练短/中/长期层级记忆；Temporal Adjacency Selection保留时间变化token，Spatial Domain Consolidation合并空间重复区域；阈值随场景统计自适应。 | “变化驱动写入”“多级记忆”“自适应压缩率”不能单独立创新。当前应和相邻差分、相似度去冗余比较，隔离分频带相对误差目标的增益；不能只比较均匀采样。 | **C0**：正式摘要及PDF方法已读，所查来源中未确认作者可执行仓库。可先建立清楚标注的TAS风格简化对照。 |
| **R22 OASIS: On-Demand Hierarchical Event Memory for Streaming Video Reasoning** | **CVPR 2026**，[CVF正式页](https://openaccess.thecvf.com/content/CVPR2026/html/Liang_OASIS_On-Demand_Hierarchical_Event_Memory_for_Streaming_Video_Reasoning_CVPR_2026_paper.html) | 无训练事件层级；先短上下文作答，不确定再按高层意图检索；PDF包含事件层级动态合并。摘要保证token成本与请求延迟受控，不应自动扩张为所有外部存储严格恒定。 | “事件记忆+按需检索+小上下文”已被覆盖。固定reader比较写入质量，并对读取次数、检索token和外部索引计费，避免把更强搜索策略误算成频谱收益。 | **C1**：[作者仓库](https://github.com/Solus-sano/OASIS)，README有评测命令，列出Qwen3-VL-8B与Qwen3-Embedding-0.6B依赖；不是默认2B本地设置。 |
| **R26 StreamingTOM: Streaming Token Compression for Efficient Video Understanding** | **CVPR 2026**，[CVF正式页](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_StreamingTOM_Streaming_Token_Compression_for_Efficient_Video_Understanding_CVPR_2026_paper.html) | Causal Temporal Reduction按相邻帧变化和显著性选token，限制每帧prefill；Online Quantized Memory以4-bit存储、按需检索解量化，限制活跃KV。 | 时间变化驱动的无训练输入压缩已有。比较时需分别列出prefill成本、活跃KV、量化历史总量；论文摘要中的bounded active memory不等于固定总历史存储。 | **C1**：[作者项目](https://yige24.github.io/StreamingTOM)指向[真实仓库](https://github.com/YIGE24/StreamingTOM)。README有LLaVA-NeXT/lmms-eval依赖与OQM配置；其他搜索返回的同名仓库不作为证据。 |
| **R11 Question-guided Visual Compression with Memory Feedback for Long-Term Video Understanding（QViC-MF）** | **CVPR 2026**，[CVF正式页](https://openaccess.thecvf.com/content/CVPR2026/html/Yamao_Question-guided_Visual_Compression_with_Memory_Feedback_for_Long-Term_Video_Understanding_CVPR_2026_paper.html) | QMSA依据问题联合选择当前片段与过去记忆，反复进行压缩和记忆反馈；摘要明确指出完整事件和temporal ordering任务。 | **“已有压缩只关注静态语义”不成立**。当前必须区别query-free写入和question-guided压缩；顺序恢复本身不是新问题。可作为知道问题的上界/另设协议，不能混入相同写入设定。 | **C1**：[作者仓库](https://github.com/FujitsuResearch/QViC-MF)，`master` README确认已发布7B adapter和评测代码，基于LLaVA-Video-7B-Qwen2；**明确不含训练代码**。不能写“完全可重训复现”。 |
| **R08 Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism（FlexMem）** | **CVPR 2026**，[CVF正式页](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Scaling_the_Long_Video_Understanding_of_Multimodal_Large_Language_Models_CVPR_2026_paper.html) | 无训练视觉KV记忆、双路径压缩写入与迁移、多种读取策略；覆盖离线长视频和流式任务。 | 当前是外部真实证据包，FlexMem直接复用潜在KV；要分别核算重编码和缓存成本。不能把冻结因果decoder的旧hidden states不变当作记忆抗干扰。 | **C1**：[作者仓库](https://github.com/city1517/FlexMem)，README给LLaVA-Video-7B-Qwen2及LongVideoBench/MLVU命令。需基座、数据和适配环境；论文单3090实验不代表本项目2B实现已复现。 |
| **R09 WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning** | **CVPR 2026**，[CVF正式页](https://openaccess.thecvf.com/content/CVPR2026/html/Yeo_WorldMM_Dynamic_Multimodal_Memory_Agent_for_Long_Video_Reasoning_CVPR_2026_paper.html) | episodic、semantic、visual三类记忆；query驱动迭代选择来源和时间尺度。 | 多尺度视觉证据保留已有。当前要固定reader和检索预算，证明证据选择而非更强agent造成收益；文本摘要不能是唯一强基线。 | **C1**：[作者项目](https://worldmm.github.io)关联[仓库](https://github.com/wgcyeo/WorldMM)。README默认记忆/检索/回答包含GPT API配置，并列Qwen3-VL-8B支持；需计入外部调用和构建成本，不是现成纯本地2B公平对照。 |
| **R10 VisMem: Latent Vision Memory Unlocks Potential of Vision-Language Models** | **CVPR 2026**，[CVF正式页](https://openaccess.thecvf.com/content/CVPR2026/html/Yu_VisMem_Latent_Vision_Memory_Unlocks_Potential_of_Vision-Language_Models_CVPR_2026_paper.html) | 短期精细视觉记忆与长期语义记忆，在长推理/生成时维持视觉grounding。 | latent vision memory和双时尺度已有；但其主要语境是**长生成中的视觉保持**，不是自动等同跨日视频事件记忆。匹配问题后再决定是否作主实验基线。 | **C1**：[作者仓库](https://github.com/YU-deep/VisMem)，README提供两阶段训练与推理命令；本轮未确认完整预训练权重可得性。 |
| **R14 Seeing the Scene Matters: Revealing Forgetting in Video Understanding Models with a Scene-Aware Long-Video Benchmark（SceneBench / Scene-RAG）** | **CVPR 2026**，[CVF正式页](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Seeing_the_Scene_Matters_Revealing_Forgetting_in_Video_Understanding_Models_CVPR_2026_paper.html)；已从官方总目录确认 | SceneBench关注场景级长程问答；Scene-RAG构建、检索跨场景动态记忆。 | 对照相似场景、证据年龄及干扰历史；性能下降不能直接解释为参数遗忘。当前开发集单例现象只能提出假设，不能替代场景级外部验证。 | **C0**：正式论文已核实，本轮未确认可执行的作者仓库与完整下载条件。同名3D/音频SceneBench不能混用。 |

### 3.1 固定预算关键帧与局部采样密度

| ID / 正式论文 | 正式身份与原始来源 | 已核实机制 / 预算边界 | 当前候选必须体现的差异与公平对照 | 公开代码及复现条件 |
| --- | --- | --- | --- | --- |
| **R30 Adaptive Keyframe Sampling for Long Video Understanding（AKS）** | **CVPR 2025**，[CVF正式页](https://openaccess.thecvf.com/content/CVPR2025/html/Tang_Adaptive_Keyframe_Sampling_for_Long_Video_Understanding_CVPR_2025_paper.html)；pp.29118–29128 | 联合考虑问题相关性与coverage。方法以时间bin分散选点、避免相邻冗余：相关性优先可退化为TOP，每bin取高分候选的BIN体现覆盖约束。这里的coverage不是本项目hidden-space k-center最大半径。 | 固定帧预算下结合覆盖与重要性已有先例。候选可区别为query-free写入、明确的特征距离硬约束及保相位轨迹失真；这些差异尚不证明新颖或有效。AKS使用问题相关评分，不能混入未来问题未知的同协议主比较。 | **C1**：[作者仓库](https://github.com/ncTimTang/AKS)，README给出BLIP/CLIP/SeViLA特征与评分、`frame_select.py`及LLaVA-Video/lmms-eval说明。需相应基座和LongVideoBench/VideoMME数据；本项目未运行。 |
| **R31 Does Your Vision-Language Model Get Lost in the Long Video Sampling Dilemma?（LSDBench / RHS / SGFS）** | **ICCV 2025**，[CVF正式页](https://openaccess.thecvf.com/content/ICCV2025/html/Qu_Does_Your_Vision-Language_Model_Get_Lost_in_the_Long_Video_ICCV_2025_paper.html)；pp.20889–20899 | LSDBench针对长视频局部高采样密度任务；RHS先根据问题定位，再局部密采样。其**SGFS子模块无需问题先验**：由初始均匀帧的cosine相似度与时间距离项构造权重，动态规划最小化所选相邻帧的权重和。 | query-free、减少冗余、兼顾时间分散及短动作取证已有直接相关工作。应把可独立运行的SGFS与完整RHS区分：前者是无查询选帧对照，后者有额外问题推理和再次访问视频成本。低重建误差不保证达到问题所需的局部采样密度。 | **C1**：[作者仓库](https://github.com/dvlab-research/LSDBench)，README含独立`sgfs_sampling`、RHS-Qwen2.5-VL和lmms-eval入口；本轮未运行。源视频需Ego4D许可及下载，公开QA标注不代替源视频获取条件。 |

## 4. 主矩阵 C：小 VLM 的正式先例

以下三篇支持“较小模型也可以有明确方法贡献”的判断，**不能证明2B模型上的任意记忆工程组合都有新颖性**，也不把高效架构、蒸馏和长期记忆视作同一个任务。

| ID / 正式论文 | 正式身份与原始来源 | 已核实工作 | 对2B记忆方向的含义 | 公开代码及限制 |
| --- | --- | --- | --- | --- |
| **R27 Align-KD: Distilling Cross-Modal Alignment Knowledge for Mobile Vision-Language Large Model Enhancement** | **CVPR 2025**，[CVF正式页](https://openaccess.thecvf.com/content/CVPR2025/html/Feng_Align-KD_Distilling_Cross-Modal_Alignment_Knowledge_for_Mobile_Vision-Language_Large_Model_CVPR_2025_paper.html) | 跨模态浅层匹配和视觉投影蒸馏；正式摘要明确**1.7B MobileVLM V2学生、7B教师**。 | 模型小并不排除研究价值；应提出可检验机制并用同规模强对照。此文不提供长期时间记忆证据，也不能作为当前方法有效性的论据。 | **C1**：[作者仓库](https://github.com/fqhank/Align-KD)。README说明是基于MobileVLM的增量代码，需合并上游代码和ZIP；不是独立一键安装的完整栈。 |
| **R28 LLaVA-KD: A Framework of Distilling Multimodal Large Language Models** | **ICCV 2025**，[CVF正式页](https://openaccess.thecvf.com/content/ICCV2025/html/Cai_LLaVA-KD_A_Framework_of_Distilling_Multimodal_Large_Language_Models_ICCV_2025_paper.html) | MDist转移多模态表示，RDist转移视觉token关系；三个训练阶段提升小MLLM。 | 支持“表示关系比仅扩大模型更值得研究”的相邻路线；不是无训练记忆压缩工作。摘要没给精确学生规模，不能擅自标成1B或2B先例。 | **C1**：正式PDF链接[作者仓库](https://github.com/Fantasyele/LLaVA-KD)，已核对`llavakd/`、`quick_inference.py`和评测说明；基座/视觉编码器/学生权重需另外准备。 |
| **R29 MaTVLM: Hybrid Mamba-Transformer for Efficient Vision-Language Modeling** | **ICCV 2025**，[CVF正式页](https://openaccess.thecvf.com/content/ICCV2025/html/Li_MaTVLM_Hybrid_Mamba-Transformer_for_Efficient_Vision-Language_Modeling_ICCV_2025_paper.html) | 用Mamba-2替换部分Transformer层，初始化与蒸馏改善效率与能力。 | 这是架构/蒸馏的高效VLM先例，不是“2B长期记忆已被验证”；可用于解释资源限制下应同时测真实延迟、峰值内存和任务质量。 | **C1**：[作者仓库](https://github.com/hustvl/MaTVLM)，含权重链接和训练/评测说明。README的`MaTVLM_0_25_Mamba2`是模型配置名，不能误读成0.25B参数量。 |

## 5. 当前候选机制与最接近工作的逐点区别

候选实现是：冻结小VLM，使用时间对齐的hidden-state轨迹，对候选真实帧/帧对作受预算限制的贪心选择；以保留相位的分频带相对重建误差评价锚点，固定首尾约束，最终reader只读取真正存下来的视觉证据。它并不直接回注或压缩KV。

| 对照对象 | 已有机制 | 当前可说明的差异 | 仍需证明什么 |
| --- | --- | --- | --- |
| Frame2Freq | 学习时间频带适配器，目标为动作分类 | 不训练adapter；用轨迹误差决定存哪些真实证据；未来问题未知 | 不是换任务产生的工程差异：必须优于同计算量的时域目标，并在独立动态QA上保持增益。 |
| FreqKV / FAEDKV | 频域变换直接压缩/更新KV | 外部帧证据选择；reader重编码存储帧 | 收益来自证据保留目标而非更大读出预算；写入和重编码成本均计费。 |
| FlashCache | 低通Base-KV的残差排序，加跨层预算分配 | 集合层面的轨迹插值重建；按频带能量归一化；选择帧而非KV | 对比低通残差排序、相邻差分、同一greedy时域MSE，排除“找变化大帧”的等价解释。 |
| InfiniPot-V / FluxMem / StreamingTOM | query-free压缩、时间冗余/显著性、容量控制 | 用不同频带的**相对重建误差**平衡短暂与持续变化 | query-free、压缩、固定预算都不是新增点；要以证据年龄和短暂状态恢复证明目标更好。 |
| ReWind / QViC-MF / OASIS | 指令/问题/意图引导记忆与证据选择 | 问题事后出现时的无查询写入；固定reader隔离写入收益 | 明确query可见性；已知问题的方法可作另外的上界协议，不能偷看未来问题当公平基线。 |
| AKS | 固定预算下联合问题相关性与时间bin覆盖 | 无查询轨迹目标；下一版可研究特征覆盖半径硬约束 | 不能把“覆盖+重要性”组合作为新点；时间bin覆盖与特征距离覆盖不能互相替代。 |
| LSDBench / RHS中的SGFS | 无问题先验的视觉多样性与时间间隔联合选帧，动态规划求解 | 集合轨迹插值失真与特征最大覆盖半径约束 | query-free不是区别；应在相同初采候选、8帧预算及reader下比较独立SGFS，并另外核算完整RHS的二次采样。 |

这张区别表是研究设计判断，**不是新颖性证明**。目前只能称为一个值得检验的有限预算采样目标；没有证据支持“首个频谱记忆”“首个动态证据记忆”或“足以发表顶会”。

覆盖半径约束下交换锚点的下一版构想，另见 [独立探索分析](next_iteration_literature.md)。它不修改已冻结的v1确认实验，也不因补入相关工作就事后更换主比较。

### 5.1 必做的等价性与预算检查

1. 使用单位正交归一化、正确处理实FFT两侧权重时，**全频段等权平方重建误差与时域平方误差由Parseval定理等价**。若两者选点不同，先检查实现、归一化、半谱权重、插值和候选集合，不能直接当研究发现。
2. 分频带相对误差改变了各频带的权重，因而可以产生不同偏好；需要比较“分频带归一化”“固定频带权重”“同greedy时域MSE”，把有效机制定位到目标而非FFT名称。
3. 仅用幅值/功率谱可能丢掉时间顺序和反转方向；候选保留复数相位时应做反转、循环位移、局部顺序扰动及无相位消融。单独的功率谱不能保证区分原序列与其反转。
4. 不能只比较锚点数：图像尺寸、帧对重复图像、时间戳、索引、编码格式、持久化hidden轨迹、检索索引都属于实际存储。临时写入工作区和后续readout预算另报。
5. 如果写入算法需要随视频长度增长的完整hidden轨迹，当前协议是**离线/有界片段压缩**；即使最终证据包大小固定，也不能称长度无关内存的在线流式算法。要做在线主张必须同时限制写入状态与工作区。
6. 当前小样本观察只用于开发：较多历史导致某一道最终状态题答错、保留后段答对，既可能是冗余干扰，也可能是采样位置、输入顺序或reader偏好。需matched-tail、等末帧、等帧数与独立留出视频验证，不能先把它归因为频域机制。
7. 做终点状态任务时，固定首尾与recency可能已解决大量题目；必须单独报告去首尾泄漏的中间短暂状态和跨事件顺序任务，避免将采样协议优势当作频谱优势。

## 6. 建议的可执行对照顺序

1. **同表示、同选择器**：相同hidden轨迹、相同候选帧与首尾约束下比较时域MSE、全频等权误差、分频相对误差、去相位误差；先验证等价性和反转性质。
2. **同持久化预算、同reader**：均匀、最近、随机、差分/变化量、k-center、低通残差、相同greedy时域重建。所有方法只把真正存下来的视觉数据交给reader。
3. **机制邻近基线**：依据最终方案选择FlashCache风格残差、FluxMem/TaR风格时间去冗余；自行改写到帧级时必须标成启发式，不冒充原文结果。
4. **正式系统基线**：先跑公开入口明确且与条件接近的方法，再考虑需要7B/8B、额外训练或API的系统；同时公布原系统预算和适配后预算。不能为适应本机而给某基线更少数据、另一个更多读取次数。
5. **外部泛化**：当前Perception Test样本只有8个视频、36道MCQA、1道官方memory题，永久作为开发集。扩展到独立长视频、多人/多场景后才能检验长期或跨主体泛化；源视频衍生片段不得跨训练/测试泄漏。

## 7. 附录 A：其他正式会场与必要经典理论

| ID / 论文 | 独立核实状态 | 保留原因与限制 |
| --- | --- | --- |
| **R04 FAEDKV: Infinite-Window Fourier Transform for Unbiased KV Cache Compression** | **Findings of EMNLP 2025**，[ACL Anthology正式页](https://aclanthology.org/2025.findings-emnlp.914/)；页面元数据为2025/11 | 频域KV压缩和频带配置的直接先行工作，即使不在本次优先会议也不能遗漏。“unbiased”是论文标题用词，不是本项目验证的普遍保证。这里不把它包装成CVPR/ICLR主会论文。 |
| **R01 Language Through a Prism: A Spectral Approach for Multiscale Language Representations** | **NeurIPS 2020**，[正式页](https://papers.nips.cc/paper_files/paper/2020/hash/3acb2a202ae4bea8840224e6fce16fd0-Abstract.html) | 隐状态频率与多尺度表示已有理论/实证先例；语言token轴与视频物理时间轴不同。只保留为概念来源，不占近期主表。 |

FFT的可逆性、Parseval等价和反转下功率谱不变是本项目必须满足的数学约束；它们不等于“低频天然就是长期记忆”的语义定理。后者仍需要任务相关实验。

## 8. 附录 B：预印本与正式身份尚未独立确认项

以下条目本轮重新读取了arXiv标题及元数据；**不计入22篇正式主矩阵**。方向性说明仅用于提醒竞争边界，不代表已完成全文审查、代码核验或最终会议验证。

| ID / 论文 | 本轮实际取得的证据 | 与本方向的关系 / 未解决问题 |
| --- | --- | --- |
| **R05 FreqDepthKV: Frequency-Guided Depth Sharing for Robust KV Cache Compression in Long-Context LLM Inference** | [arXiv:2607.06519](https://arxiv.org/abs/2607.06519)，未独立确认会议 | 沿网络深度讨论共享频率，不应混淆为视频时间频率；若加入跨层压缩，需再全文比较。 |
| **R12 MemLens: Benchmarking Multimodal Long-Term Memory in Large Vision-Language Models** | [arXiv:2605.14906](https://arxiv.org/abs/2605.14906)，comments仍为“Work in progress” | 多会话视觉证据的相邻评估，不能单独检验连续视频的时间频谱机制。 |
| **R13 MemLife: Curating and Reasoning over Long-Term Egocentric Video Memories** | [arXiv:2609.40195](https://arxiv.org/abs/2609.40195)，未独立确认会议 | 经验写入/第一视角记忆候选；MemOpt是该工作的方法名，不另算一篇。比较需核算文本与视觉证据成本。 |
| **R15 InsightEmb: Learning Action-Intent Embeddings for Agentic Insight Retrieval** | [arXiv:2608.04761](https://arxiv.org/abs/2608.04761)，未独立确认会议 | 任务推进/决策瓶颈导向的检索已有先行声称，不能把它简化为普通语义embedding，再以“更关心效用”立创新。 |
| **R16 RRM: Experience-Driven Reflective Retrieval Memory for Long-Horizon Multimodal Reasoning** | [arXiv:2607.28156](https://arxiv.org/abs/2607.28156)，未独立确认会议 | 事实证据与反思检索程序应区分；只有本方法确实包含经验检索循环时才列为可执行主基线。 |
| **R17 XSkill: Continual Learning from Experience and Skills in Multimodal Agents** | [arXiv:2603.12056](https://arxiv.org/abs/2603.12056)作者写“Accepted to ICML 2026”；本轮PMLR检索未取得对应正式页，OpenReview访问受到403验证限制 | 保留为待独立核会场项，不沿用作者声明升级。与2023年同名机器人XSkill消歧；不是当前帧级写入机制的最直接对照。 |
| **R18 LifeMem: Enabling Lifelong Experience Reuse for LLM Agents** | [arXiv:2609.12655](https://arxiv.org/abs/2609.12655)作者写“Accepted to EMNLP 2026 Main Conference”；未独立核正式论文集 | 既不属本次优先会议，最终正式身份也尚待核实；与R13 MemLife是不同论文。 |

检索候选 **FreqVLM、SpectraVLM、MIST、StreamMem、Convene** 未在本轮取得能够匹配其声称身份的正式论文页，因此不进入任何已核实机制论证；这不是对这些名称是否存在的否定判断。搜索摘要、搜索返回的arXiv号、拼接的仓库URL均必须回到原站核对标题。本轮实际发现部分搜索候选ID指向完全无关论文，已剔除。

## 9. 研究表述建议

可以提出：

> 在固定实际视觉存储预算、未来查询未知、reader保持不变的条件下，检验按时间隐状态轨迹的分频带相对重建误差选择真实视觉证据，是否比同计算量的时域重建与时间去冗余更好地保留短暂状态和事件顺序。

目前不能提出：

- 首次将频域方法用于VLM、KV缓存或长期记忆。
- 已有工作仅关注静态语义、没有研究完整事件与时间顺序。
- 固定预算、query-free写入或层级记忆本身就是新贡献。
- 功率谱能够普遍识别动作方向，或低频能量天然代表长期记忆价值。
- 由单个小VLM开发样例即可推出长期、多主体泛化或顶会录用前景。

真正值得推进的证据是：目标函数有可解释的非等价性；在严格相同资源下稳定胜过强时域对照；收益集中在预先定义的动态证据任务；且在独立视频、模型或主体上仍然成立。

## 10. 独立分区：FFN机制与参数记忆

以下6篇分别使用`F01–F06`编号，不计入22篇记忆主矩阵。它们用于界定FFN直觉与实验方法，完整设计见[FFN假说](ffn_hypothesis.md)。本轮均未复现；F01–F03读了选定官方PDF方法章节，F04–F06只核官方摘要。代码状态本轮未审计，不能把有论文或摘要代码链接当作实现已运行。

| ID / 正式论文 | 正式身份与原始来源 | 已核实机制 | 对视频FFN假说的限制 |
| --- | --- | --- | --- |
| **F01 Skip-Vision: Efficient and Scalable Acceleration of Vision-Language Models via Adaptive Token Skipping** | **ICCV 2025**，[CVF正式页](https://openaccess.thecvf.com/content/ICCV2025/html/Zeng_Skip-Vision_Efficient_and_Scalable_Acceleration_of_Vision-Language_Models_via_Adaptive_ICCV_2025_paper.html) | 分析visual-token FFN增量，训练时跳过部分冗余视觉token的FFN，加summary tokens，并在训练后跳过相关KV。 | 训练与summary-token适配是方法的一部分；不能推断冻结Qwen3-VL可无损删除FFN，也不证明短动作记忆由FFN独立承担。 |
| **F02 Too Late to Recall: Explaining the Two-Hop Problem in Multimodal Knowledge Retrieval** | **NeurIPS 2025**，[正式页](https://papers.neurips.cc/paper_files/paper/2025/hash/4526cfacdbca6b6e184568dac91bf070-Abstract-Conference.html) | 视觉实体形成与事实召回的两阶段问题；比较MLP/attention归因，以activation patching检验召回路径。 | 研究静态实体事实，不是视频episodic memory；层相关性不能替代干预，同层attention是必要对照。 |
| **F03 ParamMute: Suppressing Knowledge-Critical FFNs for Faithful Retrieval-Augmented Generation** | **NeurIPS 2025**，[正式页](https://papers.neurips.cc/paper_files/paper/2025/hash/90d1e515d74686ad148b469a3cecbbaa-Abstract-Conference.html) | 文字RAG中选出与不忠实回答相关的FFN，缩放激活，再配合知识适应和偏好优化。 | 可启发参数先验压过视觉证据的假说；不是无训练视频方法，不能用确认答案挑层，也不能把简单缩放称作原方法复现。 |
| **F04 Distributional Associations vs In-Context Reasoning: A Study of Feed-forward and Attention Layers** | **ICLR 2025**，[正式页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/533a89d7c2980218d82dbe3ea85f77ae-Abstract-Conference.html) | 受控合成任务及Pythia消融中，FFN更偏分布关联，attention更偏上下文推理。 | 分工倾向不能升级为普遍定理；FFN输入可已含时间上下文，不能据此说FFN不处理时间信息。 |
| **F05 MLP Memory: A Retriever-Pretrained Memory for Large Language Models** | **ICLR 2026**，[正式页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d72ae75abaa70a3b19c5d4f436c680d1-Abstract-Conference.html) | 预训练附加MLP模仿kNN检索，以概率插值接入decoder。 | 是额外训练的参数记忆模块；不支持冻结现成FFN能将新视频写入权重的说法。 |
| **F06 NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks** | **ICLR 2026**，[正式页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2764bbdbd9f36d9f1ec63f4ad22b24ff-Abstract-Conference.html) | 使用谱熵、有效维度及特征值分布描述FFN隐空间方差和非线性变化。 | 这里是协方差/特征值谱，不能等同时间DFT；方差再注入不代表恢复了短动作信息或长期记忆。 |
