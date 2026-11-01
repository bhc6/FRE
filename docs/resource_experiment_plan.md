# 公开资源冷启动实验方案与资源选择审计

核查日期：2026-11-01。本文保留公开资源冷启动设计。当前已完成样例下载与真实 Qwen3-VL-2B 试点，执行结果见 [小模型试点](small_model_pilot.md)；没有证明频谱记忆提升。来源仅使用作者项目、官方代码/数据与论文；硬件可行性以本机实测结果为准。

## 1. 资源选择结论

已选推进路线是：**已下载的 Perception Test 官方 sample 做接口与短时恢复冒烟检查 → EPIC-KITCHENS 的真实第一视角视频和动作时间标注发展主体可追踪的实验主线 → NExT-QA/LongVideoBench 做独立公开 QA 泛化**。前两者不能互相替代：sample 太小，EPIC 动作标注又不是现成问答。本文保留一个 NExT-QA 冷启动配置作为可选、无需自建答案的外部 QA 分支，不将其覆盖已选主线。[R1–R3,R5,R6,R10–R12]

本地已读取 `data/perception_test_sample/provenance.json`、`videos.jsonl`、`memory_questions.jsonl`：共有 **8 个视频、36 道官方 MCQ、71 段动作标注，只有 1 道 `area=memory` 的题目**。这些是导入后实际数量，不是全数据集统计。视频包为 225,370,639 字节，标注包为 2,305,618 字节；完整哈希与媒体检查见同目录 provenance/media_checks 文件。资源 JSONL 尚未包含真实 utility，也不是现有探针可直接训练的 manifest。

| 资源 | 已核实的一手证据 | 能否生成成对效用 | 适合承担的角色与限制 |
| --- | --- | --- | --- |
| Perception Test official sample | 官方下载入口与本地导入结果一致；主体/会话未知，MCQ 保存原始零基答案索引 | 可以在接入读出器后用真实答案生成 with/without；目前只有任务 GT，没有效用 | 当前接口与短时语义恢复 smoke。36 QA/1 memory 题不足以训练并验证一般长期记忆结论。[R10，及本地 provenance] |
| EPIC-KITCHENS | 官方动作标注提供 participant/video 标识、动作时间区间、verb/noun/narration；官方媒体支持逐视频获取 | 动作顺序可以按规范派生并审计；状态、方向、自然 QA 不能从动作名称直接臆造 | 正式第一视角主线候选；先一段短片验证解码与标注对齐，再真实长片，最后扩展多 participant 后才检验跨主体泛化。[R11,R12] |
| NExT-QA MC | 官方仓库发布 QA 与视频获取入口；官方 `val.csv` 含 `video,question,answer,qid,type,a0…a4`，`answer` 是选项索引 | 可以：重新运行固定读出器的 with/without 条件，再用公开正确选项评分；CSV 没有 memory utility | 可选公开 MCQ 冷启动/外部泛化。时间题优先，描述题作对照。官方数据约 5,440 个视频、平均 44 秒；不能称全天或 lifelong 评估。[R1–R3] |
| NExT-GQA | 作者提供带视觉证据定位的 VideoQA 资源，研究答案与视觉证据是否一致 | 可以用于同一 QA 的读出评估；证据区间只能作为额外评价，不等于边际效用 | 检查正确答案是否来自相关证据。与 NExT-QA 有源数据联系，不能把二者当独立数据集随机拆分。[R4] |
| LongVideoBench | 官方加载器区分 `lvb_val.json` 与 `lvb_test_wo_gt.json`；输入含视频、字幕、问题和候选答案 | 有公开 GT 的 validation 可做本地成对评估；无 GT 的 test 不用于本地效用监督 | 第二轮长时域验证。官方协议包含字幕；若采用 video-only，必须明确这是自定义视觉实验，并与带字幕版本分别报告。[R5,R6] |
| MVBench | 官方数据目录包含 `action_sequence.json`、`state_change.json`、`moving_direction.json`；论文说明任务来自不同已有资源 | 在具体子任务确认答案字段、视频映射与下载成功后可生成；此次目录核查不等于所有媒体均已取到 | 方向/状态/顺序迁移诊断。不同子任务源数据可能与其他数据集重叠，按原始来源去重。短片不能强行拉伸成 16 秒频域样本。[R7,R8] |
| Ego4D NLQ | 官方任务输出是能看见或推断答案的时间窗口 | 直接可构造定位型 reward，例如时间窗 IoU/Recall；不能未经额外答案标注就构造标准 QA exact-match | 后续第一视角定位/证据检索研究。QA 与定位 reward 必须分成不同运行，不能混用评分单位。[R9] |

以上“可以”指公开监督足够定义评分规则，仍需要模型实际推理、载荷与基础记忆哈希、固定评估设置及输出日志。公开 QA 正确答案不是记忆效用；答案选项编号不能作为 `utility`，证据窗口重叠也不能冒充 QA 收益。

## 2. 已选资源主线的具体执行顺序

### 2.1 Perception sample：只验证接口与短时恢复

先对 8 个视频完成真正解码时间戳、视频/动作边界和问题索引核验。现有唯一 memory 问题位于 `video_1580`、`perception_test:video_1580:mc:1`，询问打乱后的字母顺序在结尾是否与开头相同；官方选项是 `I don't know / Yes / No`，零基正确索引 2，对应 `No`。该实例可验证起始/结束视觉证据的比较，不是长期多会话记忆证据。

这些 sample 约 25–35 秒，开头/结尾问题必须覆盖全片。不能取最前面的 16 或 32 帧后声称模型已看完视频；可以对全时长显式重采样为约 2 FPS，并核实开始与结束状态仍可见，或对全片均匀取帧后根据实际时间间隔重设谱频带。模型 temporal patch size 为 2 时，约 52–70 帧对应约 26–35 个时间格；具体值按实际时长、边界与可整除规则确定。

官方平均 frame rate 约 29.6–30.0 FPS，不足以证明逐帧 PTS 等间隔。当前媒体检查只证明 8 个视频的首/中/末抽查共 24 帧可解码、帧数匹配，不等于验证全部 PTS。后续必须读取原始 PTS；若显式重采样到规则时间网格，保留输入 PTS、输出网格、取样/插值规则与重复帧记录。不能直接给不规则解码帧配上 `arange(T)/2` 冒充真实均匀采样。

对此题预定义空记忆、只看开头、只看结尾、同时保留开头与结尾、等成本重复末帧五个条件；所有预测必须由固定读出器实际生成。按共同模板运行其余 35 道题以验证答案解析与评分，分别报告 memory/non-memory 任务。不能把这 36 道题扩增成几百个独立样本，也不以样本内训练/测试表现判断频谱有效。

当前 `questions.jsonl` 的 `answer_index/answer_text` 只进入评分器；`utility_status=not_evaluated` 保持真实。`subject_id/session_id=null` 不补假身份。将视频、问题、动作资源转换成帧 manifest、隐藏表示、实际读出日志后，才可以产生 probe manifest；已有资源导入成功不代表这些下游接口已经完成。

### 2.2 EPIC：从真实动作时间标注建立任务

资源选择已经明确到官方单视频：短片 `P03_15.MP4` 的媒体大小为 39,548,685 字节；长片 `P01_01.MP4` 为 6,217,695,941 字节，约 27.54 分钟。此处记录的是协调代理已核实的官方媒体资源规格；是否下载/解码完成以资源清单与本地文件校验为准，不能仅凭 HTTP 入口声称跑过长视频实验。[R12]

使用官方 annotation 的 `participant_id` 作为真实去标识化主体标识，`video_id` 作为源视频分组。会话身份若没有明确字段则保持未知，不能把视频编号自动升级成真实会话。短片和长片用于工程/任务验证；只有两个 participant 时不能作有说服力的跨主体泛化结论。

分三个任务层次：

1. **可规范派生的动作顺序**：选取时间区间不重叠、在给定上下文内能唯一指代的两个正标注动作，依据官方时间边界产生“哪个先发生”的答案。题目使用动作描述，不暴露时间戳、原始顺序 ID 或答案；问题左右顺序随机化且类别平衡。设置最小间隔并排除重叠/歧义动作，保留来源 action ID 与派生规则版本。监督称“由官方动作时间标注派生”，不是“官方 QA”。
2. **需要额外核验的状态/方向**：`open`、`take`、`put` 等 verb/noun/narration 不足以证明视频最终状态、物体身份或具体动作方向。只有在起止帧/片段可见且经逐例视觉复核后新增关系标签，并保存证据时间窗；未核实的题目不纳入。可由 VLM 提议，但其自答不能自动成为 ground truth。
3. **后续真实延迟查询**：沿同一真实视频逐段观察，以 0/4/16 个新增片段及对应实际经过秒数为预设 delay，实施真实写入/淘汰并重新读出。同一个来源视频的所有 delay 条件归属同一 split；回答比较使用保留的完整证据 GT。

首轮完整研究目标可设为至少 10 个有官方标识的 participant、200–500 个经核查的独立事件/关系对，按主体隔离 train/validation/test；这是后续扩展的工作量假设，不是当前已获得样本数，也没有 power 保证。若实际可用主体不足，则明确降级为跨视频可行性试验，不虚构身份或切重叠窗口增加独立组数。

资源路径下载、动作表获取与标签派生均由项目继续自主执行，无需把“请用户提供数据”作为下一步。扩展媒体规模应按当前计算/存储资源测量结果分批进行；不要为了预设大数据量先下载全部 EPIC 原片。

### 2.3 统一读出与停止条件

Perception、EPIC 派生题与外部 benchmark 都采用下一节的实际 with/without/readout 原则，但分开 `dataset/task/reward/protocol` 运行，不拼成一张自称同分布的结果表。EPIC 派生 QA 的题型构造、视觉复核方式和来源 ID 必须公开，以便区分 benchmark GT 与新建研究标签。

如果小样本读出器不能稳定解析答案或无法消费记忆载荷，先修接口；如果只有静态帧便能完成任务，先补充真实动态诊断。只有已经评估的答案输出才能生成 `utility`，不能从动作标签、官方 answer 编号或文件哈希推算效用。

## 3. 可选公开 MCQ 分支：NExT-QA 研究设定

### 3.1 样本与划分

1. 先以 NExT-QA 官方公开分区和原始 `video` 标识建立来源表；查询 ID 使用 `nextqa:<video>:<qid>`，避免只用局部 `qid` 造成冲突。官方 CSV 未提供可靠的被摄主体或真实拍摄会话标识，不能从视频编号推断身份。[R3]
2. 初始目标为 **200 个源视频、每个最多 2 道问题**，其中 120 个训练、40 个验证、40 个测试。可从官方 train 选择开发训练视频，从官方 val 选择互斥验证/测试视频，保存原分区。该结果称“自定义 pilot 划分”，不冒充官方 benchmark test 分数。
3. 这只是计算和人工复核规模假设，不是最低统计样本数、power 保证或显著性承诺。达不到 200 个有效视频时按真实数量报告；不能用重复窗口凑“独立样本数”。
4. 按时长、公开问题类型、视频是否可解码和文件完整性选择样本；选择规则在评估前固定。不得依据测试集 full-video 是否答对、频谱特征强弱或最终方法收益选择样本。模型正确的子集可作为补充诊断，但主结果保留全部纳入样本。
5. 优先覆盖 previous/next 类时间问题；描述题单列。原类别名称并不保证问题真正需要动态信息，必须通过后述静态/顺序对照验证。动作方向和状态转移若样本不足，留到 MVBench 专项补充，不能给原始题目杜撰类别或答案。
6. 原片、裁剪、采样率变化、时间反转、所有问题与干扰变体必须跟随同一源视频分区。没有真实主体标识时仅报告 **跨视频泛化**，不报告跨人/跨家庭/跨会话泛化。

### 3.2 候选记忆与表示

采用候选片段为单位的首轮设定：源视频时长至少 16 秒；以 16 秒窗、16 秒步长切分，末尾若不足一窗，则加一个以视频末端为终点的完整 16 秒窗并去重。末窗可与前窗重叠，需保留重叠率；这会产生冗余，正好用于检验真实集合收益的非加性。长度不足者在首轮频域实验中单列，不补成伪造时间序列。

每窗以真实时刻 2 FPS 预采样 32 帧，保持共同空间预处理。若模型配置的 temporal patch size 为 2，则得到 16 个时间格、格率 1 Hz，适配现有默认 `(0,0.125] / (0.125,0.25] / (0.25,0.5]` Hz 频带。该时间格数必须由实际 `video_grid_thw` 验证，不能仅根据模型名称硬编码。多尺度、其它采样率后置，不在首轮同时搜索全部组合。

特征提取使用冻结、固定 revision 的 VLM 与不含问题/答案的统一提示。小模型试点已采用独立 Qwen3-VL-2B 适配器，预声明层为 0/14/28；Qwen2.5-VL 保留另一适配器。若需要选择最优层，只能用训练/验证数据选择后冻结，不根据测试集选择。

区分两个对象：

- **特征载荷**：每个片段的 hidden-state NPZ，用于 semantic/temporal/spectral 特征；不把 future query 送入提取。
- **真正的记忆载荷**：读出器之后能访问的预采样帧像素、源时间戳、片段边界与必要索引。文件或规范打包内容须可哈希。仅存文件路径、在测试时无限制回读原视频，不符合预算约束。

这一路线可以用现成 VLM 读取视觉片段，因此能评估“频谱指导选哪些片段”。它尚未实现对 hidden NPZ 的直接读出；不能把 hidden 频带干预结果当作已经通过模型消费的记忆。

### 3.3 固定基础记忆与标签生成

首轮基础记忆取空集合 `M0`，问题在观看结束后才呈现。对同一问题的所有候选分别估计：

```text
R(q,M) = 1[固定读出器的选项编号等于公开 GT]
U(m|q,M0) = R(q,M0 ∪ {m}) - R(q,M0)
```

确定性 MCQ 的单次 utility 属于 `{-1,0,1}`，允许负值。固定 `do_sample=false`、模型/处理器 revision、统一模板、最大输出长度与解析规则。第一版要求输出一个合法选项标记，解析失败按错误计分并单独报告；不能删除失败样本。模型输入含问题与全部候选答案，不含正确选项索引、参考答案文字标记、证据 GT 或对错反馈。

对每个 `(query,base_memory,seed,protocol)` 的 without 结果可以缓存复用，但为各候选保存对应成对日志；不能缓存一次 with 结果后用于另一种候选载荷。随机解码另设实验，采用至少三个配对种子并先按查询平均；种子数量是稳定性设计，仍不代表统计充分性。

第二个基础记忆条件可使用固定一条既有片段，但选择须不依赖候选收益且候选不在 base 中。它与空 base 分别报告，并让 `memory_state_id/base_memory_hash` 标识确切快照。首轮空 base 结果只能说明单片段条件价值，不能证明在真实更新记忆库中的边际收益已被学会。

`query_after_event=true` 在此表示实验协议将查询放在候选观察之后；它不意味着原公开视频包含真实用户数小时后的提问。应保存 `query_presentation_policy=after_source_video`，明确这是离线构造的 post-observation QA。

### 3.4 固定预算下真正读出

候选成本按读出器处理后的视觉 token、时间/边界标记和随载荷保存的额外描述统计。第一轮预设 memory-token 预算 `{1536,3072,6144}`，以试运行确认至少一档允许非空选择；若不适配实际分辨率，先在开发样本上改预算，随后冻结，禁止根据测试准确率选预算。另报实际保存字节数、解码延迟、总推理 token 和显存峰值。

每种方法在相同预算下输出候选集合，按源视频时间排序给同一读出器；相邻片段间的真实间隔和边界必须保留，不能拼接后假设时间连续。重新计算 `R(q|selected_memory)`，并同时保留空记忆基线。当前背包的 `sum_individual_utility_proxy` 和 additive oracle 仅用于诊断，不能替代此集合重评估。

至少比较：空记忆、固定预算均匀覆盖、recency、随机选择、query–segment 语义相关性、时域变化/运动强度、M1–M6。随机基线重复多个选择种子并报告方差。全视频读出可作参考，但因上下文负荷不同不必是数学上界；不要称无条件 oracle。

## 4. 防止“静态捷径”冒充时间恢复

| 控制 | 实施方式 | 可以支持的结论与限制 |
| --- | --- | --- |
| 纯语言 | 只给问题与候选答案 | 估计题目/语言先验；不能用它的答错题作为唯一测试集 |
| 单帧/重复帧 | 选择预声明中心帧；另将同帧重复至与动态载荷相同的视觉 token 成本 | 区分单幅证据与动态贡献；重复帧也会产生输入分布变化 |
| 保帧集合、乱序 | 同一批帧随机打乱，保存实际控制条件 | 检验使用顺序的敏感性；不能自动把 drop 全解释为语义时间因果关系 |
| 原视频反放 | 重新采样/编码反放视频，而不是直接反转旧 hidden | 反放可能改变问题正确答案，必须有人工/受控有效新 GT 后才能报恢复准确率；没有新 GT 时只报输出敏感性 |
| 保序时间模型 | M5/M6 加训练预算匹配的 TCN/GRU/小型时间 Transformer | 检查频域收益是否只是补上时域建模；后几项尚需实现 |
| 证据定位 | 使用 NExT-GQA 的定位标注评价选中时间段覆盖 | 衡量是否保留相关视觉证据；定位 GT 不进入选择器，也不替代 QA utility。[R4] |

将顺序、状态、方向、描述类分别报告；不存在有效标签的类别不报零分假装已测试。语义均值由因果 VLM 提取时可能已含时序信息，不能把 M1 说成严格静态编码。频谱分支包含局部顺序摘要，也不能称纯功率谱。

## 5. query-free 写入单列，不能从检索结果换个名字

完成 retrieval pilot 后，再进行独立 `write` 实验：流按时间到达，在每次写入决策时只能读取已经出现的载荷、时间与当前记忆。未来问题、候选答案及官方题型均不得进入写入特征。训练标签可以事后用同一预声明未来问题集合计算期望收益，但这些问题必须在训练/测试各自所属的视频分区中，不能跨分区共享视频内容。

同一写入决策组的所有候选使用完全相同 `label_query_ids` 与均匀权重，设置 `query_id="__expected__"`、`label_scope="expected_future_queries"`；上下文仅来自 `write_observable`。先保持与检索实验相同读出器和成本，训练不同选择头，并重新评估最终保留集合。

公开 benchmark 的后验问题集合只定义本实验所选择的查询分布，并非自然用户未来需求。首轮有限片段的离线写入排序也不是在线 lifelong 方法；需要实现 `Update(M_t,new_segment,budget)`、在每个时刻真正淘汰/压缩、保存快照并在延迟查询时读出，才可扩展相应结论。

## 6. 干扰与长期记忆的边界

冻结因果 decoder 的旧 token hidden states 不因追加后缀而重算；磁盘上的特征文件也不会因时间经过改变。应测量持续更新后的记忆内容、选择/淘汰结果或新查询的 readout。

第一轮可在固定目标片段后插入预声明数量的其它片段，保持输入长度与候选成本控制；这属于 **构造流的读取/选择干扰**。不同人物短片的拼接不能称同一主体长期经验。第二轮在更长真实视频上，以新增片段数、经过秒数和 token 负荷分别报告延迟，再讨论长期保留。

干扰选择采用训练阶段固定的语义/动态相似度，匹配时长和运动幅度；不得按测试回答失败情况反选干扰。表示频带移除仍需真正接受变换后表示的 readout、identity 往返、等重建误差与随机子空间控制。只有原始视觉片段读出器时，不把 NPZ 干预当成端到端因果证据。

## 7. 可选 NExT-QA 分支的建议预注册配置

以下是建议配置语义，**不是现有 CLI 可以直接读取的配置文件**；修订预算/层时应在开发阶段记录版本。

```json
{
  "protocol_id": "nextqa_temporal_memory_pilot_v1",
  "task_mode": "retrieve",
  "split_unit": "source_video",
  "identity_claim": "cross_video_only",
  "target_video_counts": {"train": 120, "validation": 40, "test": 40},
  "max_queries_per_video": 2,
  "minimum_source_seconds": 16,
  "window_seconds": 16,
  "stride_seconds": 16,
  "tail_policy": "one_complete_end_aligned_window_then_deduplicate",
  "input_fps": 2,
  "expected_temporal_patch_size": 2,
  "verify_actual_grid": true,
  "spectral_bands_hz": [0.0, 0.125, 0.25, 0.5],
  "stft_window_grids": 16,
  "ordered_points": 16,
  "model_family": "Qwen2.5-VL",
  "model_revision_policy": "pin_resolved_commit_before_run",
  "extraction_query_conditioned": false,
  "base_memory_policy": "empty",
  "query_presentation_policy": "after_source_video",
  "reward_metric": "mcq_exact_match_0_1_v1",
  "decoding": {"do_sample": false, "max_new_tokens": 8},
  "memory_token_budgets": [1536, 3072, 6144],
  "primary_comparison": "M4_vs_M2",
  "primary_outcome": "selected_set_qa_accuracy_difference",
  "uncertainty_unit": "source_video",
  "test_hyperparameter_tuning": false
}
```

首轮主结果是相同成本下所选集合的 QA accuracy 差，按源视频成对 bootstrap；频域 utility probe 的 MSE 改善是机制筛查指标，不能替代主结果。报告 NDCG、MAE、Spearman、正/零/负效用比例、问题类型分层、解析失败率及实际计算成本。先固定一个主要预算和层，其余预算/层作为补充或探索性结果，避免把多重搜索后的最好点当预注册结论。

## 8. 与当前代码对接所需修改

此表是从当前 v1 读取器/探针/提取器审计得到的接口要求；不是已经实现的清单。只要下列 readout 缺失，项目仍不能自产真实 memory-utility 标签。

| 位置/接口 | 必要适配 | 验收条件 |
| --- | --- | --- |
| 公共数据导入器 | 读取官方列，核验选项索引，定位原视频，保存公开分区、来源与排除原因；先按源视频划分再生成窗/问题 | 同源数据无跨 split；没有人工虚构 GT、主体或会话 |
| `data.load_samples` 与 `extract_qwen` 输入校验 | 允许未知的 `subject_id/session_id` 缺失或 null；只有按该字段分组时才要求有效真实标识；公共 QA 运行显式 `--group-key video_id` | 不把 video ID 复制成 subject ID，不用一串随机 ID 伪装身份；缺主体的默认 subject split 清楚报错 |
| provenance family | 保留规范 source-video ID；裁剪/变换不产生新的独立来源身份 | 两个不同路径的同一原片不会进入不同划分 |
| 视频预采样 | 解码真实 PTS、固定采样率、记录抽帧/缩放策略和失败，不用文件帧序号冒充秒 | `timestamps_s`、帧数与实际模型时间格一致 |
| memory readout evaluator | 固定问题模板、模型 revision 与解码规则，真正消费所选帧载荷，输出预测选项与 reward，再生成 with/without 日志 | gold answer 只在评分器可见；内容哈希对应实际可访问载荷 |
| set evaluator | 读取各方法的 selections，按相同预算重新推理整组选中记忆 | 报告真实 `R(q|set)`；不会用单条 utility 的和代替 |
| 查询上下文 | 检索模式允许从问题与全部选项提取固定 query embedding；写入模式不允许未来 query/option 特征 | 所有 M1–M6 用相同合法 context；基础 embedding 与谱增量对照完整 |
| 输出与哈希 | 除特征 NPZ 外，归档实际帧载荷、基础记忆、完整输入/输出及协议配置；路径重定位后仍可验证 | `labels`、`validate`、重复运行结果可追踪到真实候选与 readout |

如果当前主分支已完成其中某项，应以实际代码验收结果更新状态；不要为了兼容旧 loader 在数据层临时伪造主体 ID。

## 9. 计算安排与推进条件

先完成已下载的 8 个 Perception sample 与 EPIC 短片的载荷/读出冒烟检查，包括完整样本、空记忆、部分片段、格式失败及来源核验，再扩大到主线与独立泛化分支。对于 `Q` 个问题、每题 `C_q` 个候选、一个确定性 seed，空 base 标签生成可复用 without 读出，主要推理次数为 `sum_q(1+C_q)`，另加集合重评估、对照与失败重试。按实际片段数计算，不预先承诺分钟数或 GPU 显存。

只有真实标签具有可测变异、读出器能消费载荷、正/负收益可重现后才训练探针。若大量题目在空记忆下已正确，仍报告完整结果并增加独立动态诊断；若完整视觉输入也几乎不能完成目标问题，优先检查读出/采样和任务匹配，不能靠挑选测试题制造“可恢复性”。

进入长期流实验的条件是：至少一种预算下真实集合收益相对强时域/语义基线有预声明的实际意义，静态捷径和选择成本不能解释该收益，并且方向/状态/顺序类别有真实监督。未达到时，先报告可行性或负结果，不直接推进复杂 lifelong memory 架构。

## 一手来源

- [R1] NExT-QA 作者项目：<https://doc-doc.github.io/docs/nextqa.html>。用于数据构成、任务类别、平均时长与公开分区信息。
- [R2] NExT-QA 官方仓库：<https://github.com/doc-doc/NExT-QA>。用于多项选择 QA、视频获取入口与标注发布状态。
- [R3] NExT-QA 官方 validation CSV：<https://raw.githubusercontent.com/doc-doc/NExT-QA/main/dataset/nextqa/val.csv>。已直接核实表头、五个选项、答案索引、video/qid/type；它不含主体/会话身份或 memory utility。
- [R4] NExT-GQA 作者仓库：<https://github.com/doc-doc/NExT-GQA>。用于视觉证据定位与 QA 捷径诊断的适用性。
- [R5] LongVideoBench 官方仓库：<https://github.com/longvideobench/LongVideoBench>。核实 validation/test 无 GT 文件区别、视频/字幕输入形式。获取和使用遵循作者公布条件；此次未下载完整视频包。
- [R6] LongVideoBench 作者论文：<https://arxiv.org/abs/2407.15754>。用于长时域视频与字幕交错 QA 的研究目标。
- [R7] MVBench 官方数据目录：<https://huggingface.co/datasets/OpenGVLab/MVBench/tree/main/json>。核实动作顺序、状态变化、运动方向子任务存在；此次单个 state-change JSON 内容通过网页工具未能打开，不能宣称完成其字段与媒体可用性逐条验证。
- [R8] MVBench 官方 CVPR 论文：<https://openaccess.thecvf.com/content/CVPR2024/papers/Li_MVBench_A_Comprehensive_Multi-modal_Video_Understanding_Benchmark_CVPR_2024_paper.pdf>。用于时间任务分类与多来源构建背景。
- [R9] Ego4D 官方 Episodic Memory 任务说明：<https://ego4d-data.org/docs/benchmarks/episodic-memory/>。核实 NLQ 的输出是答案可见/可推断的时间窗口，而非现成多项选择答案收益。
- [R10] Perception Test 官方仓库：<https://github.com/google-deepmind/perception_test>；sample 官方视频包：<https://storage.googleapis.com/dm-perception-test/zip_data/sample_videos.zip>；官方标注包：<https://storage.googleapis.com/dm-perception-test/zip_data/sample_annotations.zip>。已下载文件哈希、实际题数、主体未知情况记录于本地 `data/perception_test_sample/provenance.json`，不是从完整数据集宣传数字推断。
- [R11] EPIC-KITCHENS 官方标注：<https://github.com/epic-kitchens/epic-kitchens-100-annotations>；项目：<https://epic-kitchens.github.io/2024>。用于官方动作时间区间、participant/video 标识和原始任务范围。
- [R12] EPIC 官方短片：<https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/P03/P03_15.MP4>；官方长片：<https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/P01/P01_01.MP4>。文件大小/时长由资源核查代理确认；本文没有独立运行这两个视频的模型评估。
