# 数据与标签协议 v1

日期：2026-11-01。目标是让“发现频谱差异”“预测边际效用”“证明记忆收益”对应不同证据，避免相互替代。本协议与 `spectral_memory.data`、`labels`、`extract_qwen` 的实际接口一致。

## 1. 研究单位与时间

一个样本表示一个候选事件、一个已固定的基础记忆状态，以及一个检索查询或预先声明的未来查询分布。真实视频必须先按主体/会话/源视频切分，随后才产生窗口、干扰变体、查询和反事实对。

源视频时间用秒表示。事件为 `[event_start_s,event_end_s]`；每个 hidden 时间戳必须位于其中。用真实采样时刻，禁止把不规则帧间隔替换为 `arange(T)`。相邻帧预处理需要降采样时，应显式记录抗混叠处理、重采样参数及原时间戳。

Qwen2.5-VL 的一个时间格通常合并多帧；以模型配置中的 `temporal_patch_size` 为准。例如输入 2 FPS、每格两帧时，隐状态采样率是 1 Hz，Nyquist 是 0.5 Hz。不能把输入帧率直接当隐状态采样率。提取器保存各帧组中心时间及实际 grid 时间间隔。

初始试验可先准备 200–500 个人工复核事件、至少 10 个独立主体/会话，覆盖顺序、动作方向、物体状态、短暂事件和静态场景。该规模只是可行性试验建议，不是统计充分性保证，也不要求第一步采集 100–300 小时。

## 2. 两种任务不可混用

| 模式 | 决策时可见信息 | 标签 | 允许进入特征的上下文 |
| --- | --- | --- | --- |
| `retrieve` | 当前查询、候选、基础记忆 | 给定查询的边际收益 | 从查询与基础记忆得到的瓶颈/任务向量 |
| `write` | 已发生的流、当前记忆、事先已知任务分布 | 未来查询分布下的平均边际收益 | 仅在写入时可见的变量 |

写入阶段不能输入未来查询、答案、未来视频、后验正确率或“是否应该保存”的标注。`context_source=write_observable` 是显式声明，代码无法证明特征的因果可得性；需要人工审计。检索瓶颈也不能由金标准答案或候选加入后的结果构造。

写入样本设 `query_id="__expected__"`、`label_scope="expected_future_queries"`，预声明 `label_query_ids`。同一 `evaluation_group` 中所有候选使用相同未来查询集合与均匀权重；不同查询重复次数可不同，先按查询平均随机种子，再等权平均查询。需要非均匀分布时必须扩展实现并记录权重，不能通过重复日志暗中加权。

## 3. 预采样帧 manifest

UTF-8 JSONL，每行一个事件。示例中的路径与哈希需要替换为真实值：

```json
{"sample_id":"p01_s01_e01_q01","candidate_id":"p01_s01_e01","video_id":"v01","subject_id":"p01","session_id":"p01_s01","event_start_s":0.0,"event_end_s":16.0,"frames":["frames/000.jpg","frames/001.jpg"],"timestamps_s":[0.0,0.5],"query_id":"q01","memory_state_id":"base01","evaluation_group":"base01_q01","cost":256,"cost_unit":"decoder_tokens","candidate_hash":"SHA256_OF_EXACT_MEMORY_PAYLOAD","base_memory_hash":"SHA256_OF_BASE_MEMORY","task_mode":"retrieve","context":[1,0,1,0,0],"context_source":"query_and_base_memory"}
```

这里仅展示两帧以说明字段；实际至少八个完整时间格。若每格两帧，至少 16 帧，帧数须能被两整除。输入帧路径相对 manifest 所在目录解释。提取器保留标注元数据但不把查询或答案加入固定提取提示。

不要假装文件名或实体 ID 就是 SHA-256。候选哈希绑定真正传给记忆读出器的序列化 payload：可能是视觉 token、帧列表/图像内容的规范编码，或压缩记忆。特征 NPZ 的哈希与记忆 payload 的哈希含义不同；只有确实直接读取 NPZ 作为记忆时才相同。

## 4. 隐状态 manifest 与 NPZ

| 字段 | 约束与含义 |
| --- | --- |
| `sample_id` | 每行唯一；同事件不同查询可有不同 sample ID |
| `candidate_id` | 同一个事件/记忆载荷的稳定标识；同决策组内不能重复 |
| `subject_id/session_id/video_id` | 全局唯一、去标识化；相同原视频的全部变体共享 video ID |
| `event_start_s/event_end_s` | 源视频内绝对秒数 |
| `hidden_path` | 相对当前 manifest 或绝对 NPZ 路径 |
| `query_id/memory_state_id/evaluation_group` | 标识任务与固定基础记忆；一个决策组只含一个查询/分布与记忆快照 |
| `cost/cost_unit` | 正整数成本及全实验一致单位；真实 token/字节/固定块，不混用 |
| `context/context_source` | 可选固定长度数值向量及来源；同一运行维度一致 |
| `utility` | 有符号边际收益，允许负值；运行探针时必需 |
| `candidate_hash/base_memory_hash` | 标签生成必需；绑定被评估的候选和基础记忆 |
| `model_id/model_revision/layer_index/pooling` | 一个运行只能一种模型版本、层和池化方式 |
| `split` | 可选；全部填写 train/validation/test 或全部不填 |
| `memory_payload_path` | 可选本地载荷路径；提供时 loader 核验文件 SHA-256 |
| `label_query_ids/label_scope` | 写入模式必需，见上文 |

NPZ 使用 `allow_pickle=False` 加载，必须包含：

```text
hidden_states: float[T,D], T>=8, finite
timestamps_s: float[T], strictly increasing and uniformly sampled
```

提取器按层输出 `manifest.layer_N.jsonl`。原输入的 `timestamps_s` 仍代表帧时间；`hidden_timestamps_s` 和 NPZ 中时间代表池化时间格。探针读取 NPZ 时间。

不要将七层当作七倍独立数据。先在训练/验证主体上选层，再在测试主体上确认；跨模型需要单独拟合或预声明映射，不可混合任意 hidden 维度。事件长度变化时频率分辨率也变化，报告长度分层结果。

## 5. 成对反事实标签

先固定候选 `m` 不在其中的基础记忆 `M`，再评估：

```text
U(m | q,M) = R(q,M ∪ {m}) - R(q,M)
```

固定模型版本、提示模板、解码设置、查询、基础记忆和随机种子，仅改变候选加入这一项。加入候选引发上下文截断会改变其他内容，应明确记录并设计等长度/等成本控制。删除已保存记忆则是另一个估计量，不能与 addition utility 混合。

每行评估记录：

```json
{"sample_id":"p01_s01_e01_q01","query_id":"q01","memory_state_id":"base01","condition":"without","seed":17,"reward":0.0,"evaluator_id":"fixed_readout","evaluator_revision":"PINNED_COMMIT","prompt_hash":"TEMPLATE_SHA256","decoding_hash":"DECODING_CONFIG_SHA256","reward_metric":"exact_match_0_1_v1","base_memory_hash":"SHA256_OF_BASE_MEMORY","candidate_hash":"SHA256_OF_EXACT_MEMORY_PAYLOAD","candidate_in_base":false,"query_after_event":true}
```

同一 `(sample_id,query_id,memory_state_id,seed)` 还需要 `condition="with"` 的第二行。`prompt_hash` 指固定提示模板而非包含不同候选的整个最终提示；最终输入也应另行归档。`decoding_hash` 应覆盖 temperature、top-p、输出长度等设置。

`labels` 命令拒绝缺对、重复、候选/基础记忆哈希不匹配、跨种子混用评分协议；输出保存 `utility_protocol`。同一实验不能混合 0–1 accuracy 与百分制 reward。`candidate_in_base` 与 `query_after_event` 是外部评估器必须保证的声明，当前工具不会读取完整记忆库来证明它们。

随机解码建议至少三个配对种子；确定性模型可一对，需额外评估评分不确定性。优先结构化答案、顺序/状态精确匹配；若采用 LLM judge，应固定版本、隐藏方法名、交换候选顺序并抽样人工复核。保留原答案和失败原因，不仅保留 reward。

## 6. 防泄漏划分

默认按 `subject_id` 约 60/20/20 划分，使用排序后的主体集合和固定种子，至少五个主体。`session_id`、`video_id` 可用于其他明确声明的评估设定，但不能称为跨主体泛化。显式 split 不可混填。

程序检查同主体分组、源视频、决策组与相同 hidden 内容不能跨 split；所有标准化/降维统计只拟合 train，alpha 只看 validation。混合不同模型/层/utility 协议、同组不同基础记忆哈希或不同写入查询集合也被拒绝。

代码无法自动识别不同文件名下的同源视频、人工复制后改写的特征、家庭成员共用场景或近重复剪辑。上游需要建立 provenance family，把原片、裁剪、加速、干预及所有问题归在同一组。先 split 再检索干扰素材；查询模板与事件类别若要求 OOD，也应独立隔离。

## 7. 干扰实验对象

冻结因果 decoder 的旧 token hidden states 不会因追加后缀被重写。固定保存到磁盘的 NPZ 也不会自行遗忘。因此不要把同一缓存向量前后差异当作干扰实验。

为每个条件明确真正可变化的量：持续更新的记忆状态 `M_t`、压缩后 payload、检索排序或新查询的 readout。采用 `M_{t+1}=Update(M_t,x_{t+1};budget)`，在每个 delay 查询并记录真实输出；若全量缓存不更新，则研究的是读取干扰，不能称为存储表征覆盖。

用语义相似高/低 × 动态相似高/低的 2×2 配对设计，加无干扰条件。每组匹配片段长度、帧率、token 数、事件数量、视觉运动幅度和目标距查询的时间。加入静态等长度 distractor 控制位置/上下文长度。动态相似度用训练阶段固定的频率范围和层，不能根据测试恢复率选干扰片。

区分真实时间延迟、流中新增 token 数、基础记忆大小和实体竞争：它们不是同一变量。建议报告多个 delay 与预算，而非一个“长期”点。

## 8. 干预与成本

当前 `intervene` 实现 identity、反转、打乱、保单通道功率的相位随机化、频带移除。它作用于已记录表征；相位随机化不保证跨通道关系不变。反转 hidden 序列也不等于“把原视频反放后重新运行因果 VLM”。两类实验应分开。

干预输出改变 sample/candidate ID 和 payload hash，移除旧标签，并标记必须重新评估。必须把变换后的表示接到实际使用它的记忆读出器，重新生成成对日志。若读出器不能消费 hidden NPZ，应先实现并验证该接口，不能把原视频答案收益直接复制过来。

频段移除需配套等重建误差的随机子空间、时域遮挡/平滑、随机频段、能量匹配、identity 往返，以及同幅度下的相位对照，控制分布外破坏。代码提供干预原语，尚未实现完整读出器与这些全部对照。

预算使用实际保存且未来可访问的信息总量：视觉载荷、索引、时间戳、位置标记、语义摘要、相位/频谱元数据都计费。不能只数选中条目而忽略不同载荷长度，也不能在省略存储成本时偷偷访问原视频。

## 9. 数据管理与人工复核

只使用允许当前用途的数据，记录版本、下载/访问日期和使用条件；受限的第一视角数据不要随仓库发布。主体 ID 去标识化，原帧与标签存放在忽略目录。发布时提供生成脚本、schema、划分哈希、去标识化错误类型与可共享样例。

每个阶段抽查时间对齐、对象身份、动作方向、最终状态、跨查询一致性和记忆载荷。先确认目标任务需要时序信息，再分析频谱；仅靠静态帧即可回答的题目应单列，不用于宣称动态恢复能力。
# 本次真实数据补充（2026-11-01）

现有真实实验分为原 8 视频/36 QA 开发集与另取的 40 视频/182 QA 确认集。两者视频 ID 和视频文件 SHA 均不重叠；不能据此声称参与者独立，subject_id/session_id 仍为 null。确认集按官方 area=memory 是否存在分层各取 20，包含 32 个 memory QA，属于富集样本。

确认选择规则、官方裁剪核对、文件哈希及主分析定义见 [冻结协议](spectral_memory_protocol.md)，下载和规范化实现见 spectral_memory/prepare_public_subset.py。规范化后的每道问题、选项、答案及区域标签均已逐项对照官方 JSON。不要根据当前模型答题结果筛选或重标。

图像 manifest 同时保存重采样目标 timestamps_s 与实际解码 source_frame_timestamps_s；前者定义模型的均匀隐藏时间网格，后者才用于和官方动作区间比较。记忆 NPZ 的 frame_indices 指向原 64 帧采样槽位，不是 MP4 帧号。所有压缩载荷按同样 512×288 画布存储，4/8/16 帧的真实 NPZ 大小分别为 1,770,300 / 3,539,820 / 7,078,860 bytes。

完整官方动作标注覆盖这 40 视频，其中 32 视频提供 246 动作，8 视频列表为空。空列表表示未提供实例，不表示物理上没有动作。短动作 ≤2 秒的阈值在覆盖评分前冻结；148 个短动作中 144 个在原观察中可见，4 个无法由后续选择恢复。

确认结果已可见，后续修订方法不得继续将该 40 视频称为独立确认集。原始视频/标签与每轮新生成的记忆、预测、源码快照须分目录保留；数据集许可与来源说明见 data/perception_confirm/ATTRIBUTION.md。

## FFN 开发诊断补充

FFN 分支实验使用原 8 个开发视频，单独从源视频重新采 16 帧。其 8 个隐藏时间格不等同于原 64 帧实验的 32 个格，也不能按数组索引合并。`frames.manifest.jsonl` 保存原视频哈希、源 PTS、重采样目标、源帧号及图像路径；`probe.json` 保存图像和代码 SHA-256、模型 revision、dtype、输入网格、显存峰值、六个残差恒等检查及提示词。

`branches.npz` 包含 `timestamps_s` 和 `layer_{index}_{branch}`，后者为 `[T,D]` 的视觉 token 空间均值。分支为 decoder_input、attention_delta、pre_ffn、ffn_input、ffn_delta、post_ffn；另存 `[T]` 的空间二阶矩和最大 token RMS，以及实际 FFN 加法舍入的空间均值。采样发生在真实 MLP 之后，不能从已池化的输入重算 MLP 冒充该分支。`post_ffn` 在外层 DeepStack 更新前立即克隆。

这些文件不包含 QA 答案，不能直接作为 utility 标签。频率分组和能量归一化在独立诊断报告中明示；空间二阶矩不能恢复所有 patch 的时间动态。完整规则见 `ffn_development_protocol.md`。

