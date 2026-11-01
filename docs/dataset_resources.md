# 视频、问答与长期记忆数据资源核查

核查日期：2026-11-01。目标是在现有机器上先跑通真实视频与真实问答，再扩展到连续第一视角记忆。以下区分“论文或网页提到资源”“目录可列出”“匿名内容请求成功”和“已经完整下载”。本文件记录资源检索和元数据核验；项目实际本地下载、解码与实验状态以 README 和运行报告为准。

## 优先选择的两项资源

### A. 立即运行：Perception Test 官方样本

选择理由：官方直接提供小样本视频和真实答案，无需账号；同一标注还提供动作起止、物体轨迹及视频时间信息。不是从第三方镜像推断可用性。

| 资源 | 官方下载地址 | 本轮核实结果 |
| --- | --- | --- |
| 视频样本 | [sample_videos.zip](https://storage.googleapis.com/dm-perception-test/zip_data/sample_videos.zip) | 匿名 HEAD 200；225,370,639 字节，约 214.93 MiB；支持 HTTP Range |
| 样本标注 | [sample_annotations.zip](https://storage.googleapis.com/dm-perception-test/zip_data/sample_annotations.zip) | 匿名 GET 成功，内存解包；内含 `sample.json`，解压后 19,931,707 字节 |
| 训练裁切规则 | [cut_frame_mapping_train.json](https://storage.googleapis.com/dm-perception-test/misc/cut_frame_mapping_train.json) | 官方 README 指定用于删除会泄露答案的结尾帧 |
| 验证裁切规则 | [cut_frame_mapping_valid.json](https://storage.googleapis.com/dm-perception-test/misc/cut_frame_mapping_valid.json) | 同上；应按样本所属原始 split 应用 |
| 文档及许可 | [官方仓库](https://github.com/google-deepmind/perception_test) | 代码 Apache-2.0；其他材料 CC-BY-4.0。不要将代码许可代替数据许可 |

解析得到 **8 条视频、36 个 multiple-choice QA、1 个 grounded QA**；其中 4 条来自 train、4 条来自 valid。视频约 25.57–35.06 秒。样本不是完整 benchmark，且集中在相似的做茶场景，不能据此宣称跨场景、跨主体或长期记忆泛化。

推荐第一条时序样本：`video_1580`，原始 valid，约 34.22 秒。其 MCQA `id=1` 是样本中唯一原始 `area=memory` 标注，考察打乱后字母排列与开头是否一致；`id=2`、`id=3` 分别考察末尾和开头的字母顺序。这些都有官方正确选项。`video_9253`、`video_8241` 也包含开头和末尾排列问题，但不能把这些题自动改写成官方 memory 类别。所有答案索引为 **0-based**。

数据读取结构：

```text
sample.json
  video_id
    metadata
      split, video_id, frame_rate, num_frames, resolution
      is_cup_game, is_camera_moving
    mc_question[]
      id, question, options, answer_id, area, reasoning, tag
    grounded_question[]
    action_localisation[]
    object_tracking[]
    point_tracking[]
    sound_localisation[]
```

`frame_rate` 是浮点值，不能统一假设 30 FPS；动作/轨迹标注的 `timestamps` 单位为微秒，`frame_ids` 为帧号。完整任务格式由[官方 README](https://github.com/google-deepmind/perception_test#the-dataset-in-more-detail)说明。当前样本的 `metadata` 不含 participant/session ID，应按原视频分组；不得把视频 ID 冒充主体 ID。

建议用途是“视频解码 → 实际时间戳采样 → 隐状态 → 时序特征 → 原始 QA 答案读出”的工程闭环，以及少量真实动态案例。先使用官方 train/valid 区分做 smoke，不用 8 条视频拟合后声称有效的统计测试。将样本拼接成长流只构成**合成干扰实验**，不能标为真实连续生活记录。

### B. 连续第一视角：EPIC-KITCHENS-100

选择理由：原视频可按单视频匿名下载，提供参与者 ID、连续视频 ID、动作起止和语义标签；可以控制真实同一会话内的事件、时间距离及干扰。它不原生提供记忆 MCQA，需要明确另建有来源的派生问题，并保留动作标注与问题的对应关系。

| 资源 | 官方来源 | 本轮核实结果 |
| --- | --- | --- |
| 视频下载器 | [epic-kitchens 官方下载脚本](https://github.com/epic-kitchens/epic-kitchens-download-scripts) | 支持 `--videos --specific-videos VIDEO_ID --output-path PATH`；避免默认全量下载 |
| 动作训练标签 | [EPIC_100_train.csv](https://raw.githubusercontent.com/epic-kitchens/EPIC-KITCHENS-100-Annotations/master/EPIC_100_train.csv) | 匿名 GET 成功 |
| 动作验证标签 | [EPIC_100_validation.csv](https://raw.githubusercontent.com/epic-kitchens/EPIC-KITCHENS-100-Annotations/master/EPIC_100_validation.csv) | 匿名 GET 成功 |
| 视频元数据 | [EPIC_100_video_info.csv](https://raw.githubusercontent.com/epic-kitchens/EPIC-KITCHENS-100-Annotations/master/EPIC_100_video_info.csv) | 本轮解析为 700 条视频，总时长约 100.028 小时；含 duration、fps、resolution |
| 标注与许可说明 | [官方 annotations 仓库](https://github.com/epic-kitchens/EPIC-KITCHENS-100-Annotations) | 数据 CC-BY-NC-4.0；仅按其非商业用途条款使用 |

已核实的单视频路径：

| 用途 | 视频、来源与下载 | 时长 | 大小与标注 |
| --- | --- | --- | --- |
| 极小第一视角对齐检查 | [P03_15.MP4](https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/P03/P03_15.MP4) | 10.477 秒 | HEAD 200，39,548,685 B；官方 train CSV 内 3 个动作，`participant_id=P03` |
| 新采集版本对齐检查 | [P07_106.MP4](https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/P07/videos/P07_106.MP4) | 16.2 秒 | HEAD 200，47,917,136 B；5 个动作标注 |
| 一条连续会话 pilot | [P01_01.MP4](https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/P01/P01_01.MP4) | 1,652.153 秒，约 27.54 分钟 | HEAD 200，6,217,695,941 B；329 个动作标注 |

三个地址均返回 `Accept-Ranges: bytes`。大小来自本轮 HTTP HEAD，时长来自官方 CSV，动作数量来自本轮筛选 train+validation CSV。短样本只验证第一视角提取和标注对齐；27 分钟连续会话才适合逐步拉长同会话记忆延迟。不要从截取的十几秒样本声称长期效果。

`P03_15` 的来源是上表原始 `EPIC_100_train.csv`，不是根据文件名猜出的主体标签。三行分别为 `P03_15_0`（3.82–4.75 秒，pick up lid）、`P03_15_1`（4.73–6.19 秒，place lid）、`P03_15_2`（6.69–7.84 秒，turn off hob），三行的 `participant_id` 均明确为 `P03`；validation CSV 没有该视频。动作区间存在轻微重叠，不能强制假定动作严格不重叠。

标注字段包括 `narration_id, participant_id, video_id, narration_timestamp, start_timestamp, stop_timestamp, start_frame, stop_frame, narration, verb, verb_class, noun, noun_class`。官方说明帧索引为 0-based；部分 narration timestamps 缺失，不能用其替代动作起止。完整 train+validation 本轮解析为 76,885 条标注、34 个 participant ID，这不是整个数据集的参与者总数。

可构建“某动作之前发生什么”“最后一次操作某物体何时发生”等事件顺序/时间定位任务。仅凭动作名不能可靠构建“物体最终在哪里”或“煤气是否关闭”这类状态真值；必须检查视频或另有状态标注。按 participant 划分需要多个主体；只有一条 `P01_01` 时只能做 pipeline pilot。视频编号并不自动证明跨会话的真实日期顺序。

## 其他候选资源的逐项核查

### LongVideoBench

- **一手入口**：[作者仓库](https://github.com/longvideobench/LongVideoBench)、[作者 HF 数据集](https://huggingface.co/datasets/longvideobench/LongVideoBench)、[HF API](https://huggingface.co/api/datasets/longvideobench/LongVideoBench)。作者 README 明确认可该 HF 地址。
- **访问状态**：本轮 API 返回 `gated="auto"`。目录可匿名列出，但直接读取 HF README 被拒绝并要求认证；不能把公开文件列表写成无需账号可下载。访问需接受该数据集条款及账号认证。
- **内容**：`lvb_val.json` 约 1.31 MB，`lvb_test_wo_gt.json` 约 4.89 MB，字幕归档约 117 MB；test 文件名明确表示无公开 ground truth。视频与字幕可以共同支持长上下文理解，但字幕是否可见要成为实验设置。
- **下载成本**：视频是 31 个连续 TAR 分片，本轮 API 合计 **161,564,180,480 B**，大多数每片 5,242,880,000 B。不是 31 个独立 ZIP；官方流程先合并再解包，会产生额外磁盘占用。不要把只下载第一片当作完整数据。
- **许可**：作者声明 CC-BY-NC-SA-4.0，视频原版权仍属于原作者；HF gate 列出相应确认项。
- **能力边界**：适合长视频问答、字幕/视觉混合证据和视频级分组；不能从其常规 QA 自动获得持续生活会话的主体、时间轴或未来查询写入协议。本轮受 gate 限制，没有逐条核实所有标注字段，因此不宣称其提供精确事件证据时间。

### Video-MME

- **一手入口**：[作者仓库](https://github.com/BradyFU/Video-MME)。作者 README 直接链接 [LMMS-Lab HF 资源](https://huggingface.co/datasets/lmms-lab/Video-MME)，因此这里是作者认可的分发入口；当前 API 返回规范 ID `lmms-eval/Video-MME`，旧入口仍能访问。
- **真实 QA**：[test parquet](https://huggingface.co/datasets/lmms-lab/Video-MME/resolve/main/videomme/test-00000-of-00001.parquet)，405,167 B；HF schema 确认 2,700 行，字段含 `video_id, duration, domain, sub_category, url, videoID, question_id, task_type, question, options, answer`。`duration` 是 short/medium/long 类别，不是精确秒数。
- **规模**：官方说明 900 视频、254 小时，范围 11 秒至 1 小时；其中 long 为 30–60 分钟。20 个 `videos_chunked_XX.zip` 合计 **100,994,322,505 B**，大多约 5 GB；[字幕 ZIP](https://huggingface.co/datasets/lmms-lab/Video-MME/resolve/main/subtitle.zip) 约 7.5 MB。
- **访问/许可**：API `gated=false`；作者明确只允许学术研究，禁止商业使用，原视频版权属于视频所有者，并限制未经批准的再分发等。不能根据其他 HF 元信息推断它是任意用途的开放视频集。
- **可用于**：跨视频长短分层 QA、时域/频域表示和读出预算比较。原生 schema 没有主体/真实连续会话 ID，也没有每题标准证据区间；不能把长视频 QA 得分直接等同长期记忆写入质量。
- **小量策略**：可在服务器支持 Range 时读取 ZIP central directory 并提取个别成员，但本轮没有实际执行大归档的局部提取。读取应校验 HTTP 206、Content-Range、压缩后/解压后大小和 CRC；不应遇到服务器忽略 Range 就悄悄下载整档。

### EgoSchema

- **一手入口**：[作者下载仓库](https://github.com/egoschema/EgoSchema)。本轮读取 [questions.json](https://raw.githubusercontent.com/egoschema/EgoSchema/main/questions.json)、[subset_answers.json](https://raw.githubusercontent.com/egoschema/EgoSchema/main/subset_answers.json)、[uid_to_ego4d.json](https://raw.githubusercontent.com/egoschema/EgoSchema/main/uid_to_ego4d.json) 成功。
- **可用标签**：5,031 条问题，公开答案只有 500 条。每题含 `q_uid, google_drive_id, question, option 0 ... option 4`；映射提供 Ego4D `video_uid, starting_sec, ending_sec`，例如某条为 240–420 秒。应将来自同一原始 Ego4D 视频的片段放在同组，不能只按 `q_uid` 随机划分。
- **官方下载**：作者提供 [Kaggle competition](https://www.kaggle.com/competitions/egoschema-public/overview)（账号、CLI 认证和接受规则）、[官方 Google Drive 文件夹](https://drive.google.com/drive/folders/1SS0VVz8rML1e5gWq7D7VtP1oxE2UtmhQ?usp=sharing)，以及读取 `uid_to_url.json` 的 Wasabi 下载脚本。README 明确临时 URL 每 7 天过期；不能保证任一旧映射当日有效。
- **许可/成本**：作者当前 README 没有统一视频再许可声明；源视频来自 Ego4D，应核对对应源数据条款。仅有第三方 HF ZIP 并不足以证明其许可。本轮未核实作者 Google Drive 全档字节数，不填造精确下载成本。
- **能力边界**：适合 3 分钟左右的第一视角长程视频理解；有原视频和起止映射，但没有现成跨日生活流。500 个公开答案适合离线测试；其余题不能用“模型自己答出的答案”补成 gold labels。官方定位为零样本评测，不应拿公开子集训练后报告为标准 benchmark 分数。

### Ego4D

- **一手入口及访问**：[官方 Start Here](https://ego4d-data.org/docs/start-here/)。视频与正式标注都要求先签署数据协议、获得 AWS 凭证；文档提示审批约 48 小时、凭证有效期 14 天。当前不属于无需账号可立刻完整获取的数据。
- **最相关任务**：[Episodic Memory](https://ego4d-data.org/docs/benchmarks/episodic-memory/) 包括 NLQ（问题对应证据时间窗）、VQ（物体最后出现时的时空轨迹）、MQ（指定活动全部时间窗）。官方描述约 74K queries / 800 小时；这些是定位和记忆评估任务，不全是选择题答案生成。
- **元数据**：NLQ 有 `video_uid, clip_uid, video_start_sec, video_end_sec, query` 及标注的证据区间；VQ 有 `query_frame`、`response_track` 等。主体信息应从批准获取的正式 metadata 映射，不能将 video UUID 当人 ID。
- **下载方式**：[官方 CLI 文档](https://ego4d-data.org/docs/CLI/)，使用 `ego4d` 工具按 UID/数据类型选取子集。全量官方文档估计 canonical videos 约 7 TB、benchmark clips 约 1 TB、annotations 约 2 GB，不应默认全量下载。
- **能力边界**：这组任务对真实第一视角“过去发生了什么、物体最后在哪”更直接。仍需固定读写可见时间、预算和 query arrival；原始定位 benchmark 不能自动证明在线记忆策略。

### Ego-Exo4D

- **一手入口**：[官方页面](https://ego-exo4d-data.org/)。官方说明 740 camera wearers、5,035 takes、1,286.3 小时 ego+exo 视频；这是多机位总时长，不能全部解释成第一视角连续时长。
- **内容**：同步第一/第三视角、take、参与者及多模态传感信息；专家评论、参与者 narrate-and-act、原子动作描述。重点是熟练活动、技能步骤和多视角，不是现成生活记忆 MCQA。
- **访问/许可**：需单独签署 Ego-Exo4D 协议；已经签过 Ego4D 也不能直接沿用。官网提示审批约两天，之后通过官方 CLI 选取数据。
- **成本与适用性**：多视角和传感器让全量代价大；应按 take 与 ego 视角选取，实际大小以下载清单为准。本轮未核实固定全量字节数。适合物体状态/程序步骤的多视角机制研究，不作为当前无需账号 pilot。

### STAR

- **一手入口**：[STAR 官方网站](https://star.csail.mit.edu/)，包含 interaction、sequence、prediction、feasibility 问答、situation graphs 和 functional programs。
- **官方标注**：[STAR_val.json](https://star-benchmark.s3.us-east.cloud-object-storage.appdomain.cloud/Question_Answer_SituationGraph/STAR_val.json) 本轮 HEAD 200，60,041,967 B；[split_file.json](https://star-benchmark.s3.us-east.cloud-object-storage.appdomain.cloud/Question_Answer_SituationGraph/split_file.json) 实际读取为 train 45,731、val 7,098、test 7,377 个条目。
- **时间映射**：[Video_Segments.csv](https://star-benchmark.s3.us-east.cloud-object-storage.appdomain.cloud/Situation_Video_Data/Video_Segments.csv) 实际读取含 `question_id, video_id, start, end`；[关键帧索引](https://star-benchmark.s3.us-east.cloud-object-storage.appdomain.cloud/Situation_Video_Data/Video_Keyframe_IDs.csv) 可进一步对齐片段。
- **视频与许可**：官网指向 [Charades 原始视频](https://prior.allenai.org/projects/charades)；该页面在本轮返回 403，未验证当日可用视频直链或下载大小。STAR 仓库 LICENSE 为 Apache-2.0，但这不自动覆盖 Charades 视频。因此不以未经核实镜像替代原视频路径。
- **能力边界**：很好地覆盖短事件顺序和预判；有程序及片段时间用于构建受控关系测试，但没有天然长生活史或个人跨会话元数据。

## 原研究方案中的新资源是否真的可用

### Memento / Memento-54K / MementoBench

已定位并读取[作者仓库](https://github.com/XiangTodayEatsWhat/Memento)（默认分支 `master`）、[数据格式](https://github.com/XiangTodayEatsWhat/Memento/blob/master/docs/data.md)和作者链接的 [liarzone/Memento-54K](https://huggingface.co/datasets/liarzone/Memento-54K)。HF API 为 `gated=false`，标注 CC-BY-4.0；代码 Apache-2.0。

官方数据格式给出：train 53,626 对话 / 4,426 视频，test 198 对话 / 40 视频。每条含 `video_uid` 和按时间排序的 `conversation`；turn 有 `role, content, time, task`，时间单位为从原视频开始计的秒数。视频 UID 清单不能当作人员清单。任务含 appear/disappear/counting/duration/ordering/changing 等。

可直接获取[测试 JSONL](https://huggingface.co/datasets/liarzone/Memento-54K/resolve/main/data/test.jsonl)及[测试视频 UID](https://huggingface.co/datasets/liarzone/Memento-54K/resolve/main/uids/test_video_uids.txt)。**源视频不在数据仓库内**，作者明确要求通过 Ego4D 获取。代码开源和标注开放不能替代该视频访问步骤。作为后续真实流式记忆评估较合适，但当前不能声称全部资源已具备。论文所称 MementoBench 在该发布中对应测试对话协议，不应凭名称再虚构一个独立视频下载包。

### SceneBench / Scene-RAG

[论文原文](https://arxiv.org/html/2603.27259v3)明确链接[作者数据仓库 SinerChen/SceneBench](https://huggingface.co/datasets/SinerChen/SceneBench)。HF API `gated=false`，数据卡声明 CC-BY-NC-SA-4.0，并说明源视频版权归原所有者和再分发限制。

数据卡描述完整 release 有 8,507 QA、六种任务。**HF Dataset Viewer 的 `data/test.jsonl` 仅有 5 条展示记录**，不是完整 benchmark。已读取第一条，字段为 `task_type, video_id, question, candidates, answer, timestamp`，`timestamp` 可含多个不连续证据区间，例如 `01:40-02:40,39:05-40:40`。

完整数据为 62 个 `SceneBench.tar.part-000...061`；本轮 API 合计 **663,160,924,160 B**，每个完整分片约 10 GiB。官方解压布局含六种任务 JSON 和各任务视频目录。当前机器不适合全量镜像；若研究组确实需要该基准，应先获得可索引的特定视频成员和对应完整 QA。不要将 5 条 viewer 记录、没有匹配视频的元数据加载成功，写成完成 benchmark 复现。

### MemLife / MemOpt

[作者论文 HTML](https://arxiv.org/html/2609.40195v1)明确评估 SuperMemory-VQA、EgoLifeQA，并构建 SuperMemory-LVQA、EgoLife-EQA 等扩展设置。它是记忆系统与写入优化方法，不能把论文简称当成已经发布的独立视频数据包。

本轮未在所读作者论文中核实到 MemLife 独立数据/视频的正式下载入口、分发许可或可立即调用的完整处理清单。因此状态是“**论文与评估设置已核实，独立资源发布未核实**”，不是断言永远不开放。若接入其评估，必须继续逐个核查上述基础数据集及扩展构造协议，而不是从第三方同名仓库推断。

## 与当前实验接口的接入约定

1. Perception Test 先保留官方 question/options/gold answer 原文与原始 ID，建立 `(video_id, question_id)` 主键；前端给 reader 的内容不得包含 `answer_id`。
2. 基于真实视频生成的 hidden states 与真实官方 QA 配对，只能称“真实数据 smoke/pilot”。用于训练的 counterfactual utility 还需要同一 query、相同背景记忆及配对 reader 调用的两次收益；QA 标签本身不是效用标签。
3. EPIC 产生的派生问题必须保存生成规则、来源动作 ID、答案证据时间和审计状态；不要与原生人工 QA 混为一组。物体最终状态需要额外核实。
4. 真实连续视频内的间隔是自然延迟；不同视频拼接的干扰应明确标记 synthetic。不能用后者代替跨日现实评估。
5. 下载 manifest 保存来源 URL、获取日期、许可证来源、内容哈希、原始 video/subject/session 映射；下载前核对 `Content-Length` 和剩余空间，只选择本次需要的成员。
6. 当前资源优先级是 **Perception Test 8 视频真实 QA闭环 → EPIC 单连续会话动态实验 → 多主体 EPIC → 获准后的 Ego4D/Memento**。Video-MME 可补充真实长视频 QA；LongVideoBench 需账号 gate；SceneBench 的大归档不进入当前全量下载计划。

所有访问状态和文件大小均是 2026-11-01 的核查快照。未注明实际成功读取的链接只代表官方提供入口，不代表已经下载或可在无凭证条件下访问。
