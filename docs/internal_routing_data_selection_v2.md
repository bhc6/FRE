# 内部机制新轮确认集：32源数据审计

2026-11-01。状态：**名单、视频和16帧输入全部准备完成；新集未进行模型评估。** 这批数据预留至候选机制、强度、比较方法及主指标冻结后使用。

## 选源规则与规模

种子 `internal-routing-heldout-20261101-v2:`。排除此前已看旧8+40+16共64源；排除 cupgame、官方 cut rule 非−1或缺失、无官方MCQA的源。分别在至少含一道 memory 题的源与不含 memory 但含 temporal 标签题的源内，按 `SHA256(UTF8(seed + video_id))` 升序取前16。候选池为388/1214源，不以答案值、模型得分、视频字节数或下载结果挑选。

| 选源层 | 源数 | 保留官方题数 |
| --- | ---: | ---: |
| memory | 16 | 72 |
| nonmemory temporal | 16 | 87 |
| 合计 | 32 | 159 |

保留每源全部原题、选项顺序及答案。题级标签统计：official memory 30；temporal 标签并集88；official descriptive 148；`descriptive_only` 71。后者严格为 `reasoning == descriptive and area != memory` 且不含 sequencing/action counting/change detection 标签，不等同严格单帧可解。

memory源：7419、3440、10434、7171、4513、10535、7726、97、5086、4508、10942、4893、1978、8834、154、6010。

nonmemory temporal源：11162、9948、1218、6580、2586、5558、2795、923、5563、5324、4162、7736、5582、9700、2108、9877。完整ID均带 `video_` 前缀，顺序为层内哈希排序。

## 获取、抽帧与独立校验

官方视频ZIP为75,375,143,328 bytes。只使用HTTP Range获取32个已锁成员，不下载整包。所选压缩成员332,912,035 bytes，实际视频文件332,273,382 bytes。选源目录探查465,094 bytes；获取阶段成功Range共333,378,761 bytes，含重试预留请求341,767,369 bytes，均低于1.5GB上限。服务器HTTP206、Content-Range、ETag、每成员CRC和文件SHA逐项校验。

新32与旧64的源ID及精确视频SHA交集均0，新32内部无重复SHA。独立审计重新读取所有视频、512张JPEG、159道规范化QA及原题JSON，核对每个原字段、完整覆盖、帧hash、均匀目标时间网格、16个独立源帧索引及首尾覆盖，全部通过。

每源16帧、max-side512、JPEG95；目标时间均匀，保留实际解码PTS和源索引，不做时间低通。解码时长 **11.400–34.974秒**。单进程，OpenCV/FFmpeg/数学库各2线程；未导入Torch，未使用GPU。与v1使用同一 `prepare_video_frames` 实现。

## 交接路径与锁定哈希

所有路径相对 `data/internal_routing_v2/`：

| 文件 | SHA256 |
| --- | --- |
| `selection.json` | `c4f06cfd0088de99202c3fb607a18d36ba2a0376c66e2ec5b6903c416211ceca` |
| `freeze_manifest.json` | `32b5f9fa6efa3a4b68fe1cf093331f4f5339ffd110bcd537ba207b7cf6ac9989` |
| `videos.jsonl` | `fa625f4d3f965a41b4bf5e271cc663b4952fe22851b638f810607d9186ef0585` |
| `questions.jsonl` | `f03ca3523b93f410f7efd4ae641b5f9d6c55c2b7562e8d9b24cf68f4b2f0c482` |
| `official_questions.jsonl` | `d6e2435c438bf1944923527e62364af9b5e9cbf6c483d35f5b9d2a595f84499a` |
| `frames16/frames.manifest.jsonl` | `22b9006df9e5f77d746f664eafbcfa36e1c613b507d59896d632618a1d0ef814` |
| `frames16/preparation.json` | `3a51b27f3241f3bc8f469582c0a5e18a967311b74100456c93d06f27ad4412d6` |
| `provenance.json` | `f95288ac5fb6a9244082bda2fbb89cc798d9d5790f052fb0f40cff914b14f6bb` |

`dataset_integrity.json` 保存全部输入、代码和帧准备文件哈希。复核命令：`python data/internal_routing_v2/audit_dataset.py`。选择脚本与获取脚本拒绝覆盖已有状态；v1冻结文件没有更改。

数据均来自官方 **valid**，`test` 只是本地角色；没有官方隐藏测试集结果。没有主体/会话隔离元数据，精确哈希排重不排除近重复或预训练污染。短片不能作为长期记忆证据。来源及CC-BY-4.0许可保存在新数据目录 `sources/README.md`，原官方问题JSON的SHA为 `174d3fa7ffa0958ae1d61311f85747dd5d0a9ae3d1e21c504dd4a47e723f78c3`。
