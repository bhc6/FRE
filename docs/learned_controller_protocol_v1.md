# V1：小模型末层学习式内部控制器比较协议

日期：2026-11-01。状态：**FROZEN：研究设计、数据、方法、训练预算与统计规则已冻结。** 本协议的SHA256由外部运行清单保存，避免自引用哈希。源码、机器环境、资源proof及12个最终checkpoint另由执行清单逐项封存，在测试前必须全部满足下文要求。本文件没有宣称已经完成训练或新测试集评估。资源proof、短训练冒烟、正式200步训练、最终普通forward评估是不同完成状态，报告必须分别标明。

## 研究问题和可接受结论

先前单一源样例的构造端点试验发现：静态8/8正确，混合12/24正确；直接attention端点门控可改变内容方向，但末端门控损害静态可读性，未通过选择性修复门槛。本轮检验一个有界、学习式的模型内部读出修正能否在独立源视频上改善官方多选题，同时保留静态输入的基线行为。

本轮只研究固定Qwen3-VL-2B、统一16帧输入下的末层控制器。原始视觉编码和语言模型权重全部冻结。正结果首先支持“小模型内部控制器在此局部数据划分上的任务收益”；只有相对于容量匹配的时域控制器仍有稳定增量，才讨论本频谱结构的额外作用。数据时长不是长期记忆证据；不把官方memory标签的正确率直接等同长期记忆机制。

## 已冻结的数据范围

| 本地角色 | 独立源视频 | 官方真实QA | 静态蒸馏/保持样本 | 状态 |
| --- | ---: | ---: | ---: | --- |
| train | 32 | 142 | 64 | 来自此前已看的40视频，只允许训练/开发 |
| validation | 8 | 40 | 16 | 同样来自已看40视频；不称独立确认 |
| new test | 16 | 74 | 32 | 排除旧8+40源ID，按预定官方标签分层选取的新源 |
| seen smoke/endpoint | 原8；端点构造只用其中1源 | 不并入主表 | 原构造32题含静态8、混合24 | 仅工程和已看诊断 |

train/validation将旧40个video_id按 `SHA256("internal-controller-seen40-split-20261101-v1:" + video_id)` 排序，前32训练、后8验证。新16按 `internal-controller-heldout-20261101-v1:` 加video_id的哈希排序，从memory源与非memory temporal源分别选8个；后者实际均属sequencing选源层。排除cupgame和需要cut的源，不根据答案值或模型结果挑选。每个入选源的全部官方MCQA均保留。

这里所有源都来自官方Perception Test **valid** split；train/validation/test是本地实验角色，不能写成使用了官方隐藏测试集。source_video_id及视频文件SHA跨划分交集为0。主体/会话ID不可用，不能声称人物/会话隔离；精确哈希排重也不排除近重复场景或预训练污染。

帧统一为每源16点时间网格、max-side 512的离线图像，另保留原解码帧索引、真实PTS、展示时间槽和图像SHA。模型处理器预算、时间分组及像素规范对全部方法相同，不能混入旧实验不同输入harness的正确率。本轮新test源时长14.362–34.910秒，不称长视频或长期记忆基准。

### 文件与哈希

| 文件（均相对data/internal_controller_v1） | SHA256 |
| --- | --- |
| selection.json | 9195c66c2616ec119f66b7d154eb7ee0da3134d79c20ada82db53dcec726bcd3 |
| split_protocol.json | 1c2b8b2c00f0128bcc1b00445f9ab5f6f731741b7cb40ec03dbeeb51ca89fdc2 |
| all_videos.jsonl | 82ae854861191c7da4848e813817a3da0c0a7d59c277624900e160dac3fd2133 |
| all_questions.jsonl | 6c905265b6d22fa24477c039e76ae626b2121994a70573330462b0f11590ca32 |
| train.questions.jsonl | b1cb8b15865aa387bed3cefb7f6a922f41f0c1c7301fe590f2f562920b3c5845 |
| validation.questions.jsonl | 76e60bb8b3cc522dfd3aa6323782d753ef58caa78b7dbcbe864224f34d7d3504 |
| test.questions.jsonl | 10a5fe30f688a512bbb93b3ad049f2f02fae3b6a88da613354d05b42aa72e4ac |
| frames16/frames.manifest.jsonl | ec364f1918ed7aba6a41655f75ee9976d7c547a8f8e1fc9c074568f4a741139a |
| frames16/train.frames.manifest.jsonl | 952cba4ce0cda44ae79992f705fd8cec691942eae555c3eefee4a16c592c4a5d |
| frames16/validation.frames.manifest.jsonl | 82d93ca7f9acb6fc8062160676a2f7e20885298218a0fd8430bdda11aac6b9b9 |
| frames16/test.frames.manifest.jsonl | b15fded6c1eecda205b69b04e12d284f5a183d42d54aa8c48bab9d962582eca9 |

## 固定子集与副作用任务

主表同时报告全部74题、official temporal-tag 41题、official memory 18题，以及“描述类非时序标签子集”30题。标签可能重叠，不把子集当独立样本重复检验。

描述类非时序标签子集的过滤器固定为：`reasoning == "descriptive"`、`area != "memory"`，并且tags不含 `sequencing`、`action counting`、`change detection`。train/validation/test分别81/17/30题。这些标签不证明单帧即可回答。原manifest中的candidate-control 22/9/8题是不同标签规则，仅可另列附录；不能把它们与上述81/17/30混用。全部official_descriptive标签为124/33/67题，可另报告，但不能把67题称为无重叠的非时序控制。

每个源的视频静态样本固定使用其首条官方QA的同一提示/选项，将第一或最后一张展示图分别重复16次，保持原展示时间槽。首条QA取冻结manifest按 `(video_id, query_id)` 字符串排序后的第一条，不按基线得分选择。原问题在重复图像下可能失去语义前提，因此这些样本 `gold=null`，只蒸馏/比较原模型输出分布，**不计算静态正确率**。train64条用于训练，validation16与test32只用于相应角色的保持诊断。真正已有字词金标的静态正确率仅来自原seen端点构造8题。

“描述类非时序标签MCQ”不是自由生成描述。除非另有预先冻结的生成和盲评方案，本实验不声明caption质量或自由描述无损。

## 固定干预位置与共同计算

基座为 `Qwen/Qwen3-VL-2B-Instruct`，revision `89644892e4d85e24eaac8bacfd4f463576704203`，BF16、同一SDPA路径。只在最后一个decoder层27、产生首个答案logit的最后prefill文本query处，构造实际SDPA各query head的context单位下的有界delta；o_proj、残差、FFN、最终norm和lm_head全部冻结。实际注入为将delta转为BF16后单独执行 `o_proj(delta)`，再加到原生self_attn的BF16输出，然后走原生残差与FFN。它没有重新计算 `o_proj(c_native + delta)`；代数线性等价不代表BF16逐位等价。10%限制仅针对投影前每head context，不保证投影后残差同样满足10%。其余视觉和文本位置不回写，不进行层搜索，不更换KV cache策略。

实际Q/K遵守模型的head norm、RoPE、GQA和完整key mask。`a[h,t,p]`是从最后query与全部keys的同一logit/mask重建后、取出的visual attention质量；归一化不能仅在visual keys内部重新softmax。`V[h,t,p,d]`按实际GQA映射使用value。保留空间位置p，不先空间平均。重建attention不是SDPA内部导出的权重；与实际context的数值差单独记录。

实际原生SDPA context `c_native[h,d]`始终作为base，控制器仅加delta。若增益为0，普通模型forward必须保持原生输出；不能用重建context替换base后再把差异当控制器收益。视觉token不改；最后文本query的后续表示允许改变。Q head有16个、head_dim 128，KV有8个；GQA展开只用于模型计算，不制造独立统计样本。

### 时间和频谱贡献的确切定义

时域贡献：`B_time[h,t,d] = sum_p a[h,t,p] * V[h,t,p,d]`，T=8。

频谱贡献先沿T做 `rfft(norm="ortho")`，然后形成

`C[h,k,d] = sum_p conj(Fa[h,k,p]) * FV[h,k,p,d]`。

8个实坐标固定为

`B_spectral = [Re C0, 2 Re C1, -2 Im C1, 2 Re C2, -2 Im C2, 2 Re C3, -2 Im C3, Re C4]`。

频谱8坐标不能直接全部相加声称等于原视觉context；正确重建权重为 `[1,1,0,1,0,1,0,1]`。必须以CPU数值检查验证该加权和与时域visual contribution总和的Parseval对应。

time的逐坐标gain对应逐时间位置的对角加权；上述cross-C谱坐标gain对应循环lag相关结构，它们不是相同函数类。这区别于“完整FFT后接一个可吸收换基的dense线性层”。等参数不等价于等表达能力。若a在时间上完全均匀，cross-C可仅剩DC，不能声称它必然保留顺序；该退化情况进入CPU检查。这里只描述8个已观测时间格的结构，不将其频点命名为原视频快速运动频率。

### 四种可训练方法和identity

| 方法 | 条件输入/实际delta basis | 训练参数 | 角色 |
| --- | --- | ---: | --- |
| identity | 增益为0，保留原生context | 0 | 所有方法的基准 |
| time_mlp | post-head-norm、post-RoPE query + B_time；delta也使用B_time | 18,584 | 容量和训练预算匹配的时域控制 |
| spectral_mlp | 同一query + signed B_spectral；delta使用signed B_spectral | 18,584 | 频谱结构候选 |
| spectral_phaseblind | 同一query + magnitude条件输入；delta仍使用真实signed B_spectral | 18,584 | 仅条件器的cross-phase消融 |
| constant_spectral | 8个共享可学习增益；delta使用signed B_spectral | 8 | 无query/content条件器的固定谱算子 |

三个MLP均为共享于所有head的 `1152 → 16 → 8`，包括bias：`(128+8*128)*16+16+16*8+8=18,584`；中间非线性SiLU，控制器参数与运算FP32，epsilon固定1e-8。q取实际SDPA收到的最后位置query，已经过head norm和RoPE，并非单独的pre-RoPE q_norm。basis条件特征除以同head原生context RMS，分母下限1e-8；q不再做此RMS缩放。用于产生delta的basis保持原尺度。delta限幅分母也使用1e-8，原生零context允许的delta为0。

phaseblind条件坐标严格为 `[abs C0, 2 abs C1, 0, 2 abs C2, 0, 2 abs C3, 0, abs C4]`。它删除条件输入中的复cross-phase及实频点符号，保留每复频点幅值；实际delta仍乘完整signed谱坐标，q和基座亦可能包含相位/顺序信息。因此**不能称power-only模型、全模型去相位或完整相位必要性检验**。constant_spectral的增益固定，但其basis仍随输入变化；它不是给所有样本加同一个常数向量，也不冒称与18,584参数模型同容量。

每head的增益固定为 `0.25 * tanh(controller_output)`，先产生delta，再对每head独立限制 `||delta_h||2 <= 0.10 * ||c_native_h||2`。共享MLP的最后线性层weight/bias全0；constant8参数全0；初始行为应为identity。任何额外缩放、clip、归一化、频带或层选择必须在冻结前写入配置，不能按newtest结果增加。

## 固定训练预算

四个可训练方法各3个seed，共12个正式训练run。固定seed `{20261101,20261102,20261103}`。identity不是重复训练的3个独立模型；比较时可与每个训练seed配对，但统计上仍是同一基线。

每run固定200个optimizer步骤，batch=8，其中6个真实QA、2个静态保持样本。分别从真实QA池和静态池对case均匀、有放回抽样；并非源视频均匀。同一seed下所有方法使用相同批次样本ID顺序，独立NumPy default_rng在每方法开始重置为该seed，采样RNG与torch初始化RNG分离；不得因方法初始化消耗不同随机数而使训练数据次序变化。

AdamW，learning rate=1e-3，weight decay=1e-4，betas=(0.9,0.999)，eps=1e-8；每步以max-norm=1.0裁剪控制器梯度，记录裁剪前范数。训练启用确定性算法并关闭TF32。只训练控制器，不向冻结权重传递参数梯度；冻结tail仍须对控制器输入保留反向计算链。记录每步CE、各项KL、delta ratio penalty、总loss、梯度范数及是否有限；NaN/Inf为运行失败，不能静默跳过batch或只保留成功seed。

损失设计固定为：

`L = mean_real CE(all original options, gold) + 0.1 mean_real KL(p_base || p_ctrl) + 1.0 mean_static KL(p_base || p_ctrl) + 1e-3 mean_sample,head delta_ratio^2`。

真实CE和真实KL各在6个真实样本求均值；静态KL在2个静态样本求均值；delta ratio平方在8个样本和16个head求均值。确切正则为 `mean(delta_after_cap.square().sum(-1) / native_context.square().sum(-1).clamp_min(1e-12))`。KL温度为1，覆盖每题**全部原始3选项**，方向为baseline teacher到controller，不是只对正确/错误两个选项做二分类。静态项不使用gold。KL teacher固定为normal native forward得到的原3选项logits之softmax，而非同batch identity-tail分布。

最终checkpoint固定为step200；**不按validation或test挑选步数、seed或方法的最好checkpoint**。validation用于报告开发表现、资源/数值和副作用诊断，不用于事后修改这12个run的超参。如果工程错误要求重跑，保留旧失败记录、修改原因和源码差异；修改后全方法按同一新冻结版本运行。不得在看到newtest后改实现/超参再把同一test称未见确认。

## 8GB资源和缓存—普通forward一致性

先在已看训练源执行资源proof：模型加载、一个完整缓存样本、一个训练batch及短步反向、普通forward启用控制器、卸载恢复。保持现有ResourceGuard、两CPU线程、0.65 CUDA allocator、原RAM/温度阈值及300秒单任务上限；若不能在边界内完成，按事先写定的分片保存/恢复执行，不放宽阈值。缓存、训练和正式评估严格串行占用GPU。

缓存只保存复现实验所需的固定状态/原生context、视觉贡献输入、实际post-head-norm和post-RoPE query、尾部残差和基线分数及其哈希；gold仅留在监督sidecar，不是控制器特征。缓存提取模型冻结、不更新权重。训练仅装载冻结最后层tail、最终norm及所需输出权重和小控制器，避免全2B模型optimizer状态。缓存文件大小和CPU RAM也属于资源账目。

BF16 GEMM随batch形状可能产生数值差，因此cached-tail分数不得直接作为论文主表。必须在固定开发样本报告：identity normalforward对native的完整词表误差；identity cached-tail batch1/batch8对normalforward的误差；启用各控制器时cached-tail与完整普通forward的选项分数、预测一致性。报告最大绝对误差、RMS误差、选项margin误差和答案变化，不能只展示平均误差或恰好一致的题。

冻结tail按 `BF16(o_proj(delta)) + native_attention_output`、加decoder_input、post-attention norm/MLP、最终norm、选定答案行的BF16 unembedding顺序重放。batch形状及仅保留输出权重的3行都会造成算术近似。协议不预设replay误差或flip的通过容差，也不根据test设置容差；只要前向和梯度有限、可微且工程校验通过即可运行，但必须报告全部误差与flip。近似训练目标不能被称为精确native目标；此限制在论文中保留。

**全部validation、newtest和seen endpoint主评估使用完整模型的普通forward、同一输入、同一控制器checkpoint。** 默认每次一个问题，避免方法之间batch形状差异。缓存可用于训练和辅助数值诊断。普通forward的identity wrapper需保持native全词表相等；缓存/批处理的非零差异须完整保留，不能称逐位等价。

## 最终评估与测试集一次性执行

在开始任何newtest评分前，冻结并保存：本协议、数据/帧/缓存哈希、模型revision、全部训练/评估源码、12个step200 checkpoint、optimizer/采样/seed配置、正常forward接口、子集规则、bootstrap规则、输出schema和资源上限。取得test cache不等于允许单独查看baseline/方法成绩；尚未冻结时不得按newtest输出修订设计。

正式newtest作一次统一评估job：identity和全部4×3seed方法，对74真实QA和32静态保持条件全部运行。单题以所有原始选项的答案标记logit作受限MCQ评分，选项标记必须验证映射唯一；不能删除难选项、重写正确选项或改成men/enm二选一。保留原选项次序、所有选项logits、确定性argmax及精确平局。此读出不是无约束文本生成。

资源分片或中断恢复只能重跑缺失的既定case/method/seed，不能依据部分test成绩增删方法或提前停止有利子集。完整性以预声明矩阵齐全为准；不完整就标partial。实现bug导致test重新开放时，保留版本和已看状态；后续调参结果只能称开发结果，新的独立结论需另一未见集合。

seen端点32题在全部模型上完整评估，静态8题和混合24题分开列。静态8/8是此前定义的功能保持门槛，不以新增蒸馏样本的KL替代。若某seed破坏该门槛，原样报告，不挑最好seed。该诊断来自已看单一源，不并入newtest或bootstrap样本数。

## 统计单位、主表和成对区间

独立计数单位是source video。74题、16个head、8个时间格及3个seed均不是74×16×8×3个独立样本。主表每项列出源视频数、题数、每seed指标及seed平均；同时报告：

- micro accuracy：全部题目的正确数/题数。
- video-macro accuracy：先在每个视频内部求QA正确率，再对含该子集题目的视频等权平均。
- 全部74、temporal41、memory18、描述类非时序30；各子集另外记录实际参与的视频数与标签交集。
- 静态保持：test32个无gold条件的baseline→controller选项KL、top-1改变比例、margin/分布漂移和扰动ratio；按视频配对汇总，不报准确率。

**预声明唯一主要比较为全部74条真实QA的video-macro accuracy上 `spectral_mlp - time_mlp` 的3-seed平均差。** 同时报micro并列出各seed值。每个可训练方法与identity作同video/case的成对差值，以及spectral对phaseblind/constant的消融差值，均为补充比较；重叠子集为探索诊断。若仅优于identity而不优于time，不支持频谱结构额外收益。报告所有3个seed，不以最好seed作为主结果。

bootstrap固定10,000次，RNG seed=20261101，95% percentile区间。主单位为视频cluster；每次在newtest的两个选源stratum内分别有放回抽8个源，保留该源全部问题，所有方法使用完全相同的抽样索引。这样保留预定8 memory-source + 8 sequencing-source构成，不把本地挑选的16源外推为整个benchmark随机样本。

分别报告两种区间：

1. **固定3个训练seed的视频bootstrap**：每次只重采样视频，指标先对3个seed平均；该区间条件于这3次训练。
2. **video×seed配对双因素bootstrap**：每次全局有放回抽3个seed索引，并按上述规则抽视频；所有方法共用seed和video索引。不能在每个视频内部独立抽seed，因为同一个训练模型覆盖全部视频。identity按相同抽样权重比较，但不伪造3个独立identity模型。

三个seed不足以准确估计广泛训练随机性，16个test cluster也使区间不稳定；第二种区间作为补充描述，不宣称保证覆盖率。子集bootstrap若某次没有可评分题，明确计为undefined并报告有效重复数，不填零。主表不做逐题独立t检验、不把重叠子集当重复确认、不在多个指标中挑一个显著结果。区间、所有单seed值和绝对计数同时公开。

## 论文表格与机器记录

至少输出下列长表，字段名由评估接口最终确认，语义不可变：

1. **逐case评估表**：run/protocol/checkpoint/source SHA、split、method、seed、video_id、case_id、condition(real/static_first/static_last/seen_endpoint)、原question/options及option token IDs、gold或null、全部option logits、prediction、correct或null、平局、各标签、native/identity分数、运行耗时、输入与帧哈希。
2. **每seed指标表**：method/seed/split/subset、video数、QA数、正确数、micro与video-macro、静态KL/改变率、seen端点分组结果、失败状态。identity只保留一个实际执行记录并标明配对复用。
3. **配对比较表**：method/reference/subset、point difference、全部seed差值、视频bootstrap区间、双因素bootstrap区间、bootstrap seed/重复数/有效数、参与video与类别层、是否来自完整矩阵。
4. **训练与资源表**：trainable/frozen参数量、各seed200步loss轨迹、真实/静态采样计划SHA、optimizer完整参数、dtype/library/driver/GPU型号、每阶段wall time、峰值allocated/reserved GPU内存、CPU RAM、温度、缓存大小、训练和所有评估总成本，包含失败proof和重跑成本。

主要结果表包含identity、time、spectral、phaseblind、constant全部方法；资源差异不能用参数少或相同token数代替实测。论文中要区分每run计算与整个研究总计算，也要说明误差条反映哪些随机性；这些记录要求与NeurIPS Paper Checklist的实验统计和计算资源条目一致（核查日期2026-11-01，官方来源：`https://neurips.cc/public/guides/PaperChecklist`）。

## 执行前与测试前封存检查

- 执行清单核验固定seed、采样规则、native KL teacher、AdamW参数及loss归一化与本协议一致。
- CPU检查覆盖cross-C重建、uniform-attention退化、phaseblind条件化范围、identity及梯度有限性。
- 缓存/训练/普通forward评估JSONL及全部源码SHA随每run保存，资源proof失败不得进入正式训练。12个step200 checkpoint齐全并封存后才执行统一test矩阵。
- test全方法完整评估前保持不读测试性能；协议冻结完成及资源proof完成均不等于正式实验已完成。

本轮允许无增益、时域更好、静态副作用或seed不稳定的结果。若频谱方法没有独立增量，结论应收缩到实际证据，不以新名字包装相同基线，也不以开发/seen端点成绩代替newtest。
