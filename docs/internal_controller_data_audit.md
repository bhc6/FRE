# 小型内部干预实验的数据审计与来源划分

日期：2026-11-01。范围：CPU、官方 HTTP 资源和原始 QA 标注；本轮没有加载模型、使用 GPU 或读取任何模型预测。本文记录可复现的数据准备，不报告模型效果。

## 1. 已获得的数据及其用途

现有 8 个样本视频和 40 个旧确认视频均已在此前研究中被观察过，不能重新称为独立测试。本轮从官方 Perception Test validation 下载了 **16 个新的源视频和对应全部 74 道官方选择题**。旧 40 源按固定 ID 哈希划为 32 train / 8 validation；新 16 源独立保留为本地 held-out test。旧 8 源仅作 smoke，不进入统一训练、验证、测试清单。

| 本地角色 | 视频数 | 全部 QA | 官方 memory | 指定时序标签并集 | 官方 descriptive | 描述对照候选 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train，旧源 | 32 | 142 | 23 | 53 | 124 | 22 |
| validation，旧源 | 8 | 40 | 9 | 19 | 33 | 9 |
| test，新源 | 16 | 74 | 18 | 41 | 67 | 8 |
| 合计 | 56 | 256 | 50 | 113 | 224 | 39 |

这些类别重叠。时序标签并集定义为官方 `tag` 含 `sequencing`、`action counting`、`change detection` 中任一项。新 test 中分别为 27、10、5 题，并集 41；18 道 memory 题均在该并集中。

**Descriptive 不等于静态。** “描述对照候选”仅按不属于 memory、不含上述时序标签、同时含 object recognition / object attributes / place recognition / part recognition / material 中任一标签定义；没有通过单帧标注或模型结果验证其单帧可解性，不应报告为已证明的静态对照。

若主表需要 `descriptive_only`，可预先固定更直接的标签过滤器：`reasoning == "descriptive" and area != "memory" and not (set(tags) & {"sequencing", "action counting", "change detection"})`。train / validation / test 分别有 **81 / 17 / 30** 题。它表示“描述类且无指定时序标签”，仍不证明静态可解。原“描述对照候选”的 22 / 9 / 8 题中，仅 21 / 7 / 7 题的 reasoning 本身是 descriptive；不能混用这两种定义。该过滤器由冻结标注派生，不修改主清单或其哈希。

这 56 个视频的官方原始 split 均为 `valid`。文中的 train / validation / test 是本地研究划分，不能冒称官方 benchmark train/test，也不能以这 16 个分层选择视频的结果代替全量 benchmark 分数。

## 2. 官方接口与实际下载范围

当前官方 README 与下载的原始文件保存在 `data/internal_controller_v1/sources/`。一手来源：

- 官方说明：<https://github.com/google-deepmind/perception_test>。
- Train MCQA：<https://storage.googleapis.com/dm-perception-test/zip_data/mc_question_train_annotations.zip>，本次实际 87,873 bytes；解出 1,955 视频、7,392 题。
- Valid MCQA：<https://storage.googleapis.com/dm-perception-test/zip_data/mc_question_valid_annotations.zip>，本次实际 204,076 bytes；解出 5,260 视频、19,140 题。
- Train 视频归档：<https://storage.googleapis.com/dm-perception-test/zip_data/train_videos.zip>，HEAD 为 28,404,087,974 bytes；本轮没有下载该归档的视频。
- Valid 视频归档：<https://storage.googleapis.com/dm-perception-test/zip_data/valid_videos.zip>，HEAD 为 75,375,143,328 bytes。
- 官方结尾裁切规则：<https://storage.googleapis.com/dm-perception-test/misc/cut_frame_mapping_train.json> 和 <https://storage.googleapis.com/dm-perception-test/misc/cut_frame_mapping_valid.json>。

本轮使用官方大 ZIP 的 central directory 和 HTTP Range 抽取个别原视频成员，而非下载整个归档，也没有假定存在经验证的独立 MP4 URL。复用 `spectral_memory.prepare_public_subset.HttpRangeReader`，每个请求验证 HTTP 206、精确 Content-Range 和一致 ETag；每个成员验证压缩目录大小、解压后大小和 ZIP CRC32，再计算内容 SHA256。

冻结成员的压缩体积合计 255,377,296 bytes，原视频合计 254,727,671 bytes。实际视频获取阶段成功 Range 合计 255,843,205 bytes，包含目录和成员头；含重试预留为 255,843,226 bytes。检查阶段的少量 central-directory 请求另记于 `inspection_archive_ranges.jsonl`。未下载整个 75 GB 归档，因此不伪造其整体 SHA256；保留 ETag、每个 Range 哈希、每个源视频哈希。

下载上限固定 800,000,000 bytes，未达到上限。源文件许可按已存官方 README 为非代码材料 CC-BY-4.0；代码许可不能替代数据许可。

## 3. 选源冻结与去重

选源 seed：`internal-controller-heldout-20261101-v1:`。

在官方 valid 标注中，排除旧 8+40 的全部 48 个 video ID、需要末尾裁切或缺少裁切规则的视频、cup-game 视频、无官方 MCQA 视频。保留视频均 `official_cut_frame=-1`，不通过忽略裁切来引入结尾答案泄漏。

候选分成两组：含官方 memory 题的源；不含 memory 题但含上述指定时序标签的源。各组按 `SHA256(UTF8(seed + video_id))` 升序选前 8。筛选只使用 ID、官方题型标签、裁切规则与 cup-game 元数据，未使用答案值、模型输出、模型对错或基线准确率。实际第二组 8 源均含至少一道官方 sequencing 题。

冻结的 16 个 ID：

- Memory：`video_8108`、`video_6184`、`video_2418`、`video_3020`、`video_11070`、`video_5425`、`video_687`、`video_5320`。
- Nonmemory temporal：`video_604`、`video_309`、`video_1567`、`video_7303`、`video_10049`、`video_1001`、`video_8376`、`video_3017`。

每个选中视频的**全部官方 MCQA**都保留。`official_questions.jsonl` 保存原始 question 对象；规范化清单保留题目、选项顺序、answer index、area、reasoning、tag。统一 256 题全部与原始官方 JSON 逐字段核对，全部视频的问题数量亦一一核对，没有按题目难易或模型表现删题。

完成校验：新 16 与旧 48 的源 ID 交集为 0、视频内容 SHA256 交集为 0；新 16 内部重复内容 SHA256 为 0。统一 56 源也无重复 ID 或内容 SHA256。这个检查排除明确身份和逐字节重复，**不证明不存在视觉近重复、参与者/会话重合或预训练污染**。官方这些元数据不提供 participant/session ID，相关字段保持 null。

## 4. 旧源训练/验证划分

Split seed：`internal-controller-seen40-split-20261101-v1:`。对旧 40 源按 `SHA256(UTF8(split_seed + video_id))` 升序排列，前 32 个 train，后 8 个 validation。整个源视频及其所有问题始终属于同一个 split。划分不读取标签值或模型输出；旧源已见的事实不会因重新划分而消失。

Validation IDs：`video_1024`、`video_6532`、`video_3007`、`video_341`、`video_1414`、`video_10267`、`video_8221`、`video_275`。全部 train/test ID 和源哈希见 `split_protocol.json`。后续学习代码应只从 `train.questions.jsonl` 拟合，使用 `validation.questions.jsonl` 进行预先定义的选择，完整方法冻结后再运行新 test；该协议记录数据角色，不冒充已执行了训练冻结。

## 5. 可直接读取的接口

所有文件位于 `data/internal_controller_v1/`：

| 文件 | 用途 |
| --- | --- |
| `selection.json` | 新 16 选源规则、ID、预计字节、官方来源哈希 |
| `provenance.json` | 实际下载、CRC/Range记录入口、来源去重结果 |
| `videos.jsonl` / `questions.jsonl` | 仅新 16 / 74 题，原始获取清单 |
| `official_questions.jsonl` | 新 74 题的原始官方 question 对象 |
| `all_videos.jsonl` / `all_questions.jsonl` | 统一 56 源 / 256 题，含本地 split |
| `train.videos.jsonl` / `train.questions.jsonl` | 32 源 / 142 题 |
| `validation.videos.jsonl` / `validation.questions.jsonl` | 8 源 / 40 题 |
| `test.videos.jsonl` / `test.questions.jsonl` | 16 源 / 74 题 |
| `split_protocol.json` | ID 划分、QA 一致审计和各清单哈希 |

问题字段沿用旧 `questions.jsonl`：`video_id, video_path, query_id, question, options, answer_index, answer_text, area, reasoning, tags`。统一清单另加 `split, usage_role, video_sha256, audit_categories`。`benchmark_split` 保留 `valid`，不被本地角色覆盖；`video_path` 在统一清单内为已验证的绝对路径。

冻结的核心 SHA256：

- `selection.json`：`9195c66c2616ec119f66b7d154eb7ee0da3134d79c20ada82db53dcec726bcd3`。
- `split_protocol.json`：`1c2b8b2c00f0128bcc1b00445f9ab5f6f731741b7cb40ec03dbeeb51ca89fdc2`。
- `all_videos.jsonl`：`82ae854861191c7da4848e813817a3da0c0a7d59c277624900e160dac3fd2133`。
- `all_questions.jsonl`：`6c905265b6d22fa24477c039e76ae626b2121994a70573330462b0f11590ca32`。
- `test.questions.jsonl`：`10a5fe30f688a512bbb93b3ad049f2f02fae3b6a88da613354d05b42aa72e4ac`。

## 6. 16 帧时间网格

`assemble.py frames` 复用项目现有 `prepare_video_frames`，从原视频直接导出 16 个覆盖首尾的均匀目标时间点，每点选择最近的真实解码帧，保留原始 PTS 与源帧号。空间长边最多 512、保持宽高比、JPEG quality=95。未施加时间低通，频谱解释必须考虑稀疏采样与混叠；不能把源视频 FPS 直接当作 hidden temporal groups 的采样率。

CPU 导出使用单进程，数学库、OpenCV 与 FFmpeg decoder 各限制为 2 个工作线程；没有 PyTorch/GPU。每个源视频在导帧前重新核对内容哈希，每帧另存 SHA256。接口保持与现有帧处理一致：`frames, timestamps_s, source_frame_timestamps_s, source_frame_indices, event_start_s, event_end_s, sampling`。

导出目录是 `frames16/`；主清单 `frames16/frames.manifest.jsonl`，按 split 的清单为 `frames16/train.frames.manifest.jsonl`、`frames16/validation.frames.manifest.jsonl`、`frames16/test.frames.manifest.jsonl`。完整完成状态、最终清单 SHA256、视频实际时间范围记录于 `frames16/preparation.json`，不能把导出过程中逐条写入的中间主清单当作完整数据。

本轮已完成 **56/56 源、896 张帧图像**。独立重读验证全部帧 SHA256、每源 16 个不同源帧号、覆盖解码首尾、目标时间网格等间隔，全部通过。`torch_imported=false`、`gpu_used=false`。全部源实际时间范围为 10.036–35.031 秒，新 test 为 **14.362–34.910 秒**；这些短视频不能支持分钟级或长期记忆结论。

- 完整帧清单 SHA256：`ec364f1918ed7aba6a41655f75ee9976d7c547a8f8e1fc9c074568f4a741139a`。
- Train 帧清单 SHA256：`952cba4ce0cda44ae79992f705fd8cec691942eae555c3eefee4a16c592c4a5d`。
- Validation 帧清单 SHA256：`82d93ca7f9acb6fc8062160676a2f7e20885298218a0fd8430bdda11aac6b9b9`。
- Test 帧清单 SHA256：`b15fded6c1eecda205b69b04e12d284f5a183d42d54aa8c48bab9d962582eca9`。

可复现入口：`python data/internal_controller_v1/acquire.py inspect`、`acquire.py acquire`、`assemble.py manifests`、`assemble.py frames`。这些入口拒绝覆盖已有冻结状态；本轮数据已准备时应直接消费清单，不重复执行覆盖。

## 7. 评估侧官方标签契约

新增纯标准库 helper `data/internal_controller_v1/evaluation_contract.py`，不改任何冻结清单。`load_evaluation_cases(split)` 按预设 SHA256 检查选源 / split 协议、相应 QA / 帧清单及官方 `sources/mc_question_valid.json`；由 query ID 精确查到源视频内的官方题 ID，并逐字段核对题目、选项顺序、0-based answer_index、area、reasoning、tags，验证 QA—帧的 video ID / 内容哈希 / 路径一致。

运行 `python data/internal_controller_v1/evaluation_contract.py --split all` 已通过 256 题、56 源的完整契约检查，再次确认 descriptive_only 为 **81 / 17 / 30**。它返回每题的规范化 `question`、冻结帧 `source`、`official_question` 和固定 `subsets`，不读取模型预测。评估时仍须在加载图像后核对逐帧哈希。

评估输出字段建议及训练公平性源码审计另见 `docs/internal_controller_training_audit.md`。静态构造的 official gold / correct 必须为空；只有原真实视频可使用原官方答案计算正确率。
