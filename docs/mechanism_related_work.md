# 时间信号、位置背景与FFN因果作用：机制与相关工作核查

核查截止：2026-11-01。本文只做一手资料、数学与源码层面的审查，没有运行模型或GPU，没有读取确认QA成绩，也没有改变冻结协议或文献主表计数。选取8篇2025–2026正式论文，重点判断哪些现象已经被研究、哪些对照可以复用、当前研究还缺什么证据。所有作者方法均未在本项目复现。

**当前最有价值的问题是：在相同输入位置与因果上下文结构下，静态反事实能否提供可靠的时间信号对照；由此定义的分量是否在有通路的中层影响时间任务，并帮助有限预算证据选择。** 静态扣除不是“纯运动提取”的保证，FFN频谱变化也不是因果瓶颈的证据。最后decoder层视觉位置的FFN输出尤其需要作为结构性负对照。

## 1. 八篇直接相关的正式工作

### M01：Causality Matters，AAAI 2026

**完整标题**：Causality Matters: How Temporal Information Emerges in Video Language Models。

**来源与核验**：[AAAI正式PDF](https://ojs.aaai.org/index.php/AAAI/article/download/37856/41818)，首页确认AAAI-26，起始页9006；[正式DOI](https://doi.org/10.1609/aaai.v40i11.37856)。本轮读取官方首页及第5页的attention knockout方法，并阅读其[作者扩展版](https://arxiv.org/abs/2508.11576)第4–6节。曾遇到官方连接失败，随后直接取得正式PDF；会议身份不依赖arXiv录用声明。

**已经覆盖**：作者比较改变位置编码与改变帧顺序，利用分层attention knockout追踪帧间、帧到query、query到最终预测的信息路径。在所测VideoLM和任务中，早层帧间作用与中层query整合比“显式视觉位置编码决定一切”更能解释结果，并提出深层视觉交互/缓存截断策略。

**可复用的实验**：以目标token集合`T`和来源集合`S`定义额外attention mask，把`T×S`的logit设为负无穷；分别阻断帧间、帧到query以及query到输出路径。作者采用5层滑动窗口；本项目低资源版本若只干预一层，应明确不是同等设置复现。保留帧内容、位置ID及文本时间戳的独立记录，不用“逆序”一个词掩盖三种不同变换。

**碰撞风险与限制**：“因果attention构造时序信息”“深层视觉token未必直接驱动最终回答”均已有直接先例。本文不把这一结论升级为所有VideoLM的普遍定理；也不把作者的任务效应当成静态重复帧hidden漂移已经被完整解释。静态反事实扣除是否可靠仍是额外问题。

### M02：Video-LevelGauge，ICLR 2026

**完整标题**：Video-LevelGauge: Investigating Contextual Positional Bias in Video Language Models.

**来源与核验**：[ICLR正式页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4c2a220d89b4f88222222b46d9de1495-Abstract-Conference.html)，已核官方总目录；正式PDF第3节的probe/context设计片段已读。这里采用正式论文标题，不把搜索到的改题稿名替代正式身份。

**已经覆盖**：把相同的probe片段插入不同长度、不同类型的上下文和不同位置，保持问题可比，以此区分位置与上下文造成的表现变化。该工作研究head及neighbor-content等偏好，而非只汇报完整视频的平均成绩。

**可复用的实验**：同一短事件内容在前、中、后不同槽位出现，配套相同的无答案泄露问题；分别控制上下文长度和邻近内容。对本项目而言，先保持总帧数、分辨率、token格和prompt不变，再移动事件，比拿不同视频的不同位置直接比较更有解释力。

**碰撞风险与限制**：可控事件插入、位置偏差基准和邻近背景影响不是新发现。移动事件可能改变真实事件顺序或题意，必须限定问题是否位置不变，或者独立标注变换后的正确答案。它没有证明隐藏状态可按“内容+固定位置项”线性拆开。

### M03：Position Bias / SoFA，CVPR 2025

**完整标题**：Identifying and Mitigating Position Bias of Multi-image Vision-Language Models。

**来源与核验**：[CVF正式页](https://openaccess.thecvf.com/content/CVPR2025/html/Tian_Identifying_and_Mitigating_Position_Bias_of_Multi-image_Vision-Language_Models_CVPR_2025_paper.html)，pp.10599–10609；正式摘要及PDF第3–6页相关实验和第4节已读。

**已经覆盖**：作者把显式位置编码与inter-image causal attention区分开，比较causal、isolated和bidirectional图像间交互。SoFA在图像间因果和双向交互之间插值，同时保留文本片段的因果结构；其插值系数由小验证集决定。

**可复用的实验**：先用同内容置换评估位置敏感性，再采用“禁用图像间交互”的诊断检查漂移是否依赖跨帧历史。完整SoFA应依原实现核对mask与归一化语义，不能把任意attention mask混合声称为原方法复现。

**碰撞风险与限制**：减少位置偏差与抹去时间顺序是不同目标。将双向交互直接用于流式视频会改变可见的未来信息；原多图任务的有效性不能担保视频顺序任务不受损。单独发现“帧位置改变hidden”已不足以构成机制新意。

### M04：The Unseen Bias，ICLR 2026

**完整标题**：The Unseen Bias: How Norm Discrepancy in Pre-Norm MLLMs Leads to Visual Information Loss。

**来源与核验**：[ICLR正式页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/03de675250598f76340d459c8bbf75d5-Abstract-Conference.html)，官方目录与正式PDF已核；读取第3节假设/角速度分析、第4节部分实证及第5节方法。

**已经覆盖**：在Pre-Norm残差结构下，视觉和文本token的范数差可伴随不同的表示更新速度。作者在假设下分析更新向量相对原残差的旋转幅度，并在视觉投影后加入按文本embedding尺度初始化的LayerNorm；完整方法还处理训练中的梯度衰减问题。

**可复用的诊断**：同层分别统计视觉/文本token范数、支路增量范数、增量与残差的夹角，以及相邻表示的方向变化。必须同时记录RMSNorm之前与之后的量，不用归一化后的输入替代真实残差基底。

**关键限定**：理论使用更新幅值、期望夹角和旋转方向等假设。正文Ovis 2.5实例明确出现较大静态范数差但没有同等更新不对称，说明不能只凭范数差判定视觉损失。其修复包含新增层的训练与梯度补偿，不能描述为冻结Qwen3-VL的无训练推理修复。残差范数并非在任意网络中数学必然单调增长，负交叉项可以使它下降。

### M05：Frequency Entropy / Weighted RoPE，ICLR 2026

**完整标题**：Probing Rotary Position Embeddings through Frequency Entropy。

**来源与核验**：[ICLR正式页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0aee38a6fe9fffc8b658cfb1d872c1d5-Abstract-Conference.html)，官方目录及正式PDF第3、5节已读。

**已经覆盖**：对各二维RoPE pair的query范数沿token轴计算DFT/STFT与谱熵，再按熵阈值缩放对应query/key分量。其Weighted RoPE把选中的二维子向量乘以`alpha`，属于由频谱诊断指导的推理激活干预。

**可复用的算法骨架**：固定诊断集和频谱定义，计算归一化功率分布的Shannon熵，预先选维度，再在独立任务上比较缩放和同幅度对照。本项目只有8个视频时间格，不能直接沿用论文4096-token序列及1024长度STFT的配置。

**数学解释必须收紧**：该文第3.1节的观测量为`||q_t^(j)||₂`，并明确指出二维旋转保范数。因此在上游query固定时，单独施加/撤销标准RoPE不能改变该范数序列；观察到的谱结构还可能来自输入内容、上游计算或训练。其式(12)为`alpha R(theta)q`，这改变幅值而非旋转角，不等于减慢相位增长。本文据此只借鉴其频谱选维和干预范式，不沿用超出公式支持的直接位置因果解释。

**碰撞风险**：“频谱异常维度→衰减→下游测评”已存在；Q/K二维旋转轴、token序列谱与本项目frame-pair hidden时间谱必须区分。新意不能来自给普通激活缩放加上“频域”名称。

### M06：BOLD，ACL 2025主会长文

**完整标题**：Addressing Blind Guessing: Calibration of Selection Bias in Multiple-Choice Question Answering by Video Language Models。

**来源与核验**：[ACL Anthology正式页](https://aclanthology.org/2025.acl-long.162/)，Volume 1: Long Papers，pp.3216–3246；正式PDF第4节、第6–7页分解与校准公式已读。不是Findings论文。

**已经覆盖**：去掉视频、问题或有效选项中的某个要素，估计模型的选项位置先验；由校准样本估计全局prior，再作`softmax(log P_original - log P_prior)`。Weighted BOLD还需要确定分解项的权重。

**可复用的实验**：时间QA必须有no-video及选项顺序置换检查；测量干预后答案变化时，同时检查它是否仅改变字母/位置偏好。若使用校准prior，应固定独立校准集并与原始分数同时报告。

**碰撞风险与限制**：反事实/删输入估计背景偏差并扣除的范式已有。这里扣的是答案分布偏差，不是hidden时间背景；其“缺失关键信息时应均匀”的假设也不能直接移植到仍保留对象和场景的静态视频。因此静态参考预测未必是无信息先验，不能无条件从logits中扣掉。

### M07：FLoC，ICLR 2026

**完整标题**：FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding。

**来源与核验**：[ICLR正式页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a87d3a6b4884108c0dff392c1c7e2d82-Abstract-Conference.html)，官方目录及正式PDF第3节、Algorithm 1已读。

**已经覆盖**：query-agnostic、training-free的视觉token预算选择，以facility-location目标`sum_i max_{j in S} sim(i,j)`覆盖原token集合，用lazy greedy降低实际搜索开销。论文按时间块执行以控制开销，并把未来流式扩展与当前方法区分。

**可复用的算法**：对相同输入表征和候选池计算相似度矩阵，按最大边际覆盖增益贪心选择；维护优先队列可少算边际增益。改为frame-pair候选时必须标注为帧级适配基线，不能冒充论文原token压缩复现。

**碰撞风险与限制**：覆盖、多样性、预算和query-free coreset已是直接先例。其近似比针对代理覆盖目标，不是QA准确率；要使用标准单调次模近似结论，须满足非负、归一化目标与相应约束等条件。原始cosine可为负，具体实现必须交代空集和相似度处理。加入覆盖半径、事件配额或其他约束后，不能直接沿用同一个近似界。

### M08：StreamMind，ICCV 2025

**完整标题**：StreamMind: Unlocking Full Frame Rate Streaming Video Dialogue through Event-Gated Cognition。

**来源与核验**：[CVF正式页](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_StreamMind_Unlocking_Full_Frame_Rate_Streaming_Video_Dialogue_through_Event-Gated_ICCV_2025_paper.html)，pp.13448–13459；正式PDF第2.3–2.5节相关片段已读。

**已经覆盖**：学习型Event-Preserving Feature Extractor以状态空间结构结合当前帧与过去状态，生成perception token；Cognition Gate结合用户query决定是否调用LLM，另有perception memory与取样流程。

**可复用的实验思想**：把持续感知成本、事件状态保持、调用时机和最终回答分开评估；插入无关片段后测试是否仍能恢复目标事件状态。若自实现门控，应显式区分query-free写入与query相关响应。

**碰撞风险与限制**：“event-preserving”“感知记忆+事件门控”已有正式先例。名称不构成事件完备保存定理；EPFE、gate和表示对齐需要训练，也不能由每帧感知成本恒定推断所有历史存储恒定。

## 2. 这些文献尚未给出的结论

本轮没有核到一篇能直接证明：对冻结Qwen3-VL的某层hidden，减去同槽位的重复帧hidden，就能唯一分离真实运动；或者时间白化必然保留QA相关事件。检索未发现完全匹配不是“首次”的证据，尤其不能忽略M02/M03的位置控制、M05的频谱干预、M06的反事实校准和M07的预算选择这些已覆盖的组成部分。

已有矩阵中的FreqKV、FlashCache、AKS、SGFS，以及[FFN文献分析](ffn_hypothesis.md)中的Skip-Vision、Too Late to Recall继续构成直接边界，不作为本轮新增发现。Skip-Vision已研究visual-token FFN跳过；其对末视觉token预测后继文本的讨论，恰好说明必须明确读出位置。下面的严格负对照是计算图推论，不能包装为独立新理论。

## 3. 静态重复输入为何仍可能出现时间轨迹

把第`ell`层分支输出写为

\[
H_\ell(V,P,Q)\in\mathbb R^{T\times D}.
\]

`V`表示视觉序列，`P`包括显式位置ID、帧槽位和可见时间戳，`Q`表示固定prompt及此前文本上下文。即使所有帧内容相同，因果模型各槽位可访问的前缀长度、重复视觉token数量、相对位置及时间戳文本仍不同，因此`H(t)`不必恒定。若模型还有跨帧视觉编码或DeepStack，也必须纳入计算边界。

但这不是“因果attention必然制造漂移”的定理：在没有差异位置编码、没有异质前缀且所有可见value完全相同的特殊情形中，softmax加权仍可能得到相同value。静态轨迹应实际测量，不能预设其必然大于零，也不能把所有静态轨迹变化都归给RoPE。

在Qwen3-VL中至少分开三种位置因素：显式位置ID、因果可见性、嵌入prompt的数字时间戳文本。仅改变其中一个因素，不能宣称排除了所有位置/上下文因素。

## 4. 静态反事实扣除：定义、假设和可否定标准

### 4.1 配对输入

选定参考图像`r`，构造`R_r(V)`：把每个输入帧替换成同一参考图像，但保留原序列长度、尺寸、时间槽位、时间戳文本、prompt、token布局与dtype。重复某个含两张不同图像的frame-pair并不是真正静止，应作为另一类周期性控制；真正静态基线应在每个输入槽位放相同图像。

参考图像的内容来源PTS和它在合成输入中的指派时间槽位必须分别保存。不能将重复帧的真实来源时间误写成它被放置的时间，或因此触发processor重新采样。

令`C_T`去除时间均值，定义

\[
D_{\ell,r}=C_T\{H_\ell(V,P,Q)-H_\ell(R_r(V),P,Q)\}.
\]

它是**相对于指定静态参考的表示差**，不是已识别的纯运动。只有额外假设成立，例如

\[
H_\ell(V,P,Q)=B_\ell(P,Q)+M_\ell(V),
\]

且静态输入的`M_ell`不随时间变化时，才有`D=C_T M(V)`。真实attention、RMSNorm和MLP包含内容与位置/历史的交互，通常不能假设同一个`B`在动态和静态条件下保持不变。

若写成`H=B(P,Q)+M(V)+I(V,P,Q)`，扣除后仍有交互差`I(V,P,Q)-I(R_r(V),P,Q)`。它可能包含任务有用信息，也可能含重复输入造成的背景效应；不能仅凭频谱形状辨别。

### 4.2 三个必须通过的控制

1. **采集一致性与参考敏感性**：相同输入重复运行给出数值误差基线；预先固定首、中、尾等少量参考图像，不按QA挑参考。若现象仅对一个参考成立或幅值主要由参考选择决定，应否定“稳定运动分量”的解释。
2. **同事件换槽位**：在固定长度和背景中移动相同事件，区分跟随事件的分量与固定在输入位置的分量。事件顺序变化会改变标签时，须独立更新真值；不能既改变题意又要求答案不变。
3. **匹配与错配参考**：匹配静态参考与预先指定的错配参考同时检查。若任意扣除都产生相同效果，可能只是缩放、中心化或一般扰动；不能把它归因于去除了视频特定的位置背景。错配控制会额外改变语义，只作机制诊断，不作等价任务输入。

写入时问题未知，应保持中性`Q`；如果每道测试题都重新构造反事实或按其答案选择参考，已经变成query-aware方法，不能放回query-free写入协议。

### 4.3 FFN的配对增量

设`Y`为FFN前残差，`M`为FFN输出，`Z=Y+M`为FFN后残差，则

\[
C_T\{(Z_{\mathrm{dyn}}-Z_{\mathrm{static}})
-(Y_{\mathrm{dyn}}-Y_{\mathrm{static}})\}
=C_T(M_{\mathrm{dyn}}-M_{\mathrm{static}}).
\]

这是支路配对差的代数恒等式；实际BF16采集须加入已记录的舍入残差。它能避免把两条路径的不同DC基底混为“FFN新生时间信号”，但不提供该增量对答案的因果证据。

## 5. 时间白化改变的是度量，不是信息来源

若需要降低静态背景常见方向的权重，应先明确白化轴。以下是时间轴白化，不是隐藏通道协方差白化：

\[
K_{\mathrm{static}}=\frac{1}{RD}\sum_{r=1}^R
(C_TH_r)(C_TH_r)^\top,
\qquad W_\lambda=(K_{\mathrm{static}}+\lambda I)^{-1/2}.
\]

然后比较`C_T H`、`D_r`与`W_lambda D_r`。所有选择器和基线都必须使用相同变换，不能只给候选方法背景校准。

白化后的平方距离是一个数据依赖Mahalanobis度量。它不会凭空增加视觉信息；它可能抑制有用事件，也可能放大静态控制中低能量的数值噪声。正则项、协方差估计集、参考集合及秩截断应提前固定，不能依确认QA挑选。对当前8个时间格，谱分辨率很低，不宜把几个离散频点解释成稳定的物理运动频带。

如果`K`来自当前视频的无标签静态参考，这是query-free的测试时适配，但会增加多个前向及工作区成本；若从跨视频数据学习，须按源视频/主体分组留出。`W`通常混合所有时间槽位，使用未来状态，因此只能作为离线压缩或诊断，不能直接声称在线因果白化。

## 6. DC、RMSNorm和attention sink不能互相替代

- **时间DC**：同一hidden通道沿时间轴的均值组成的向量。
- **大激活**：某些token/通道的幅值很大，未必时间恒定。
- **attention sink**：某来源位置吸引较多attention质量，是路由现象；不等于该位置语义重要，也不等于一条时间均值向量。
- **RMSNorm**：逐token沿隐藏通道轴缩放，通常不减去时间均值，也不减去通道均值。

设FFN前后为

\[
X=\mathbf1\mu^\top+U,\qquad
M=\mathbf1\nu^\top+V,\qquad C_TU=U,\ C_TV=V.
\]

那么

\[
E_{\mathrm{DC}}(X+M)=T\|\mu+\nu\|_2^2,
\]
\[
E_{\mathrm{AC}}(X+M)=\|U\|_F^2+\|V\|_F^2+2\langle U,V\rangle.
\]

只要`mu`与`nu`的抵消足够强，AC占比可以显著升高，即使AC绝对能量持平或下降。因此必须报告DC、AC绝对量、交叉项和比值，不能只看比例。“DC抵消”也不能追溯解释一个本来已经排除DC的选择目标为何成功或失败。

RMSNorm是内容依赖的非线性缩放。一般有`C_T Norm(X) != Norm(C_T X)`，所以去DC、归一化与FFN的顺序不可交换。MLP即使逐token运算，也可对输入正弦通过非线性产生谐波；出现新高频不证明它新增了跨时间聚合或记忆。

若要把变化归因给attention sink，必须额外测量被关注位置及干预；仅凭范数/DC无法识别sink。首轮低资源实验若不采集attention路由，就应保持“未检验sink机制”的表述。

## 7. 最后decoder层视觉FFN是结构性负对照

在本项目固定的普通decoder-only读出中，最后一层为

\[
Y^L=H^{L-1}+A^L(H^{L-1}),\qquad
Z^L_t=Y^L_t+M^L(\operatorname{Norm}(Y^L_t)),
\]

答案logits读取最后文本位置`q`：

\[
\operatorname{logits}_q=W_{\mathrm{LM}}\operatorname{Norm}(Z^L_q).
\]

attention已经结束；之后的MLP、final RMSNorm与LM head均逐token。只修改视觉位置集合`I_v`上的最后层MLP输出，保持输入长度、位置及文本支路不变，则对`q notin I_v`有

\[
\frac{\partial\operatorname{logits}_q}{\partial M^L_t}=0,
\quad t\in I_v.
\]

最后层KV也由FFN前的状态生成。若没有其他跨token模块或额外状态更新，所有层缓存不变，后续自回归文本的计算也可归纳为不变。零置该视觉FFN、改变其DC或任意时间频带，都不应改变标准答案位置logits。这比“消融后统计上没有明显变化”更强：在上述条件下是计算图断路。

**适用边界**：不适用于读视觉位置logits来预测后继文本的训练损失、不适用于把视觉hidden交给额外跨token头，也不适用于外部选帧器使用这些hidden后又改变reader输入。若干预实际动了最后层attention输入、K/V、序列长度或文本位置，也不再是这个负对照。自定义跨token量化/归一化或其他共享状态应另行审查。

这里根据[固定transformers源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)推导结构；实际hook仍需通过原样与零置条件的答案logits逐项一致性验证，避免边界实现错误。本文不宣称已完成该实验。最后层视觉FFN出现很强的谱变化，完全可以与QA因果作用解耦，不能据此挑它作为瓶颈。

## 8. 有通路的中层如何检验频谱因果性

中层视觉MLP输出能通过后续attention流向文本，因此存在因果通路，但“存在通路”不等于该分量被任务使用。

建议只预先选择一个中层，先比较原样、温和的整支路缩放、指定非DC分量干预和同范数扰动。频带mask、强度与层位必须由开发规则确定；不得通过确认答案扫描到有利组合。保留最后层视觉FFN负对照和同层attention对照。

频域操作必须说明对象：真实token级`T×P×D`支路，还是空间池化后的`T×D`诊断。把池化向量的变化广播回所有patch，是另一种干预，不能称为恢复原token分量。全序列DFT还会跨未来槽位混合，只能用于机制诊断或离线操作，不能不加修改地部署为在线因果滤波。

判断标准是：干预是否在人工确认的时间任务上产生预期、可重复的答案margin变化，同时区别于一般缩放、格式偏好和同能量扰动。对顺序反转或事件换位，要明确新标签；对MCQA，要有M06提醒的选项置换与no-video检查。若仅仅影响所有题的语言能力，不能称为时间专用分量。

即使中层干预得到效应，也只能先称为特定分量对该计算路径和任务的因果贡献。将它改成记忆写入策略，仍需单独验证预算、选帧、独立QA和计算成本。

## 9. 从机制诊断到task-aligned coreset的边界

M07已经给出代表性/多样性覆盖的直接基线。一个可精确定义但尚未验证的扩展是

\[
F(S)=\sum_i w_i\max_{j\in S}s(g_i,g_j),\qquad w_i\geq0,\ s\geq0,
\]

其中`g_i`可以是原表征或静态校准后的表征。非负权重、固定相似度及标准基数预算下，这仍是熟悉的加权facility-location目标；增加权重本身不是新算法原语。

如果`w_i=||D_i||²`，它只是“相对静态参考的变化强度”代理，不是task-aligned。若权重由任务敏感性学习而来，需要独立训练任务分布、固定拟合过程及无问题写入映射；不能在测试时使用未来QA、正确答案或其梯度，再宣称query-free。训练和额外前向也应计费。

求和覆盖可能牺牲稀有短事件。增加事件配额、最大覆盖半径或顺序约束必须说明新的可行集合，不能继续直接引用无约束基数版本的近似界。任何数学覆盖保证都不是“足够证据支持QA”的保证：原候选没有采到动作、帧对不足以区分方向、reader不能利用证据，都可能令更优代理目标无效。

当前值得检验的最窄组合是：**可重复的静态校准是否改善了中层、具备任务因果作用的证据度量，并在相同真实帧预算下优于原表征的覆盖选择。** 对照至少包括原表征facility location/k-center、仅中心化的同目标，以及使用完全相同搜索与预算的静态校准版本。先用小候选池审计优化，再讨论扩展；不靠更复杂搜索包装不存在的表征收益。

## 10. 可交付证据与停止条件

| 层级 | 所需证据 | 不足时应停止的说法 |
| --- | --- | --- |
| 采集与轴 | 明确槽位、源PTS、position IDs、时间戳文本、分支边界、dtype及舍入；静态/动态布局匹配 | “测到的是物理运动频率”。 |
| 反事实有效性 | 多参考稳定、同事件移位、静态输入和错配参考控制 | “扣除项就是位置背景”“余量就是纯运动”。 |
| 数学非等价性 | 与中心化、缩放、普通时域二次型/距离的区别；DC与AC分开 | “FFT产生了新信息”“AC占比升高就是时间增强”。 |
| 因果性 | 最后层负对照通过；中层目标干预优于同范数扰动解释；答案位置偏好受控 | “某层谱大所以该层是QA瓶颈”。 |
| 记忆效用 | 同候选、同frame-pair/字节、同reader、同优化搜索；独立视频上比较 | “机制图好看就意味着有限记忆更好”。 |

本轮八篇来源已分别核到正式会议身份和所标明的方法片段；没有核查所有作者仓库的可执行性，没有复现论文成绩，也没有进行穷尽检索。当前可以形成严格的机制问题和实验定义，尚不能保证方法有效、研究新颖或满足任何顶会录用标准。
