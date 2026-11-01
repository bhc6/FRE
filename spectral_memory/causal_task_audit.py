"""Audit the original development questions and recorded answers without model loading."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re


EXPECTED_VIDEOS = {
    "video_10993", "video_1580", "video_2564", "video_3501",
    "video_6860", "video_8241", "video_874", "video_9253",
}
PILOT_CONDITIONS = ("no_memory", "first_half", "last_half", "uniform_full")
SELECTORS = (
    "uniform", "recent", "random", "temporal_difference", "kcenter",
    "temporal_reconstruction", "spectral_power", "spectral_balanced",
)
EXPECTED_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_category(question):
    text = question["question"]
    if text == "What is the order of the letters at the end?":
        return "endpoint_last_letters"
    if text == "What was the order of the letters at the beginning?":
        return "endpoint_first_letters"
    if text.startswith("How many letters changed position"):
        return "endpoint_change_count"
    if question["area"] == "memory":
        return "endpoint_equality_memory"
    if text.startswith("What action not related"):
        return "distractor_action"
    if text.startswith("What can you say about the temperature"):
        return "apparent_temperature"
    if text.startswith("Which of the following statements"):
        return "object_spatial_relation"
    if text.startswith("Is the camera"):
        return "camera_motion"
    if text.startswith("Where is the person"):
        return "place"
    if text.startswith("What is the person preparing"):
        return "preparation_action"
    raise ValueError(f"Unreviewed question text: {text}")


def compact_pilot(row, question):
    raw = row["raw_answer"]
    prefix = re.match(r"^\s*[\[(]?([A-Z])(?:[\]).:\s]|$)", raw)
    prefix_index = ord(prefix.group(1)) - ord("A") if prefix else None
    if prefix_index is not None and prefix_index >= len(question["options"]):
        prefix_index = None
    return {
        "condition": row["condition"], "raw_answer": raw,
        "parser": row["parser"], "official_reward": row["reward"],
        "posthoc_prefix_index": prefix_index,
        "posthoc_prefix_correct": prefix_index == question["answer_index"],
        "posthoc_prefix_is_primary_score": False,
        "option_logits": None, "gold_margin": None,
        "logits_absence_reason": "Free-generation pilot did not save option logits.",
        "frame_count": row["frame_count"], "timestamps_s": row["timestamps_s"],
        "candidate_hash": row["candidate_hash"], "forward_id": row["forward_id"],
        "input_prompt_sha256": row["input_prompt_sha256"],
        "visual_tokens": row["generation"]["visual_tokens"],
        "input_tokens": row["generation"]["input_tokens"],
    }


def compact_readout(row, question, dataset):
    response = row["response"]
    logits = response["option_logits"]
    answer = question["answer_index"]
    require(len(logits) == len(question["options"]), "Option count mismatch")
    require(all(math.isfinite(value) for value in logits), "Nonfinite option logits")
    predicted = max(range(len(logits)), key=logits.__getitem__)
    require(predicted == response["answer_index"], "Stored argmax mismatch")
    require(int(predicted == answer) == row["reward"], "Stored reward mismatch")
    require(response["raw_text"] == chr(65 + predicted), "Stored answer letter mismatch")
    require(response["model_revision"] == EXPECTED_REVISION, "Unexpected model revision")
    other = max(value for index, value in enumerate(logits) if index != answer)
    return {
        "dataset": dataset, "method": row["method"], "budget_pairs": row["budget_pairs"],
        "actual_pairs": len(row["selected_pairs"]), "selected_pairs": row["selected_pairs"],
        "raw_answer": response["raw_text"], "answer_index": predicted,
        "reward": row["reward"], "option_logits": logits,
        "option_token_ids": response["option_token_ids"],
        "gold_margin": logits[answer] - other,
        "argmax_tie": logits.count(max(logits)) > 1,
        "payload_sha256": row["payload_sha256"], "memory_bytes": row["memory_bytes"],
        "visual_tokens": response["visual_tokens"], "input_tokens": response["input_tokens"],
        "forward_id": response["forward_id"],
        "reused_identical_payload_forward": row["reused_identical_payload_forward"],
    }


def counterfactual_plan(question, endpoints):
    category = audit_category(question)
    base = {
        "identity": {"status": "valid_original_target", "answer_index": question["answer_index"]},
        "visual_ffn_only": {
            "status": "valid_fixed_input_fixed_target",
            "answer_index": question["answer_index"],
            "restriction": "Only visual token positions; no text or answer token modifications.",
        },
        "reverse_sequence": {
            "status": "unscored_without_new_annotation", "answer_index": None,
            "reason": "An arbitrary reversal may change action semantics; original-video wording also conflicts with displayed-order relabeling.",
        },
        "endpoint_preserving_interior_permutation": {
            "status": "unscored_without_new_annotation", "answer_index": None,
            "reason": "Same frame multiset does not establish unchanged action semantics.",
        },
        "repeat_first_frame": {"status": "unscored_without_new_annotation", "answer_index": None},
        "repeat_last_frame": {"status": "unscored_without_new_annotation", "answer_index": None},
    }
    if category.startswith("endpoint_"):
        base["endpoint_preserving_interior_permutation"] = {
            "status": "conditional_endpoint_target_invariance",
            "answer_index": question["answer_index"],
            "requirements": [
                "Explicitly score the displayed sequence rather than original-video order.",
                "Preserve first and last evidence frames and confirm their readable states.",
                "Retain the original question text; flag any presupposition violated by the edit.",
            ],
        }
    if category in {"endpoint_last_letters", "endpoint_first_letters"}:
        states = endpoints[question["video_id"]]
        original_position = "last" if category == "endpoint_last_letters" else "first"
        opposite_position = "first" if original_position == "last" else "last"
        for condition, source_position in (
            ("reverse_sequence", opposite_position),
            ("repeat_first_frame", "first"),
            ("repeat_last_frame", "last"),
        ):
            state = states[source_position]
            require(state["answer_text"] in question["options"], "Derived state absent from options")
            base[condition] = {
                "status": "conditional_annotation_derived_endpoint_target",
                "answer_index": question["options"].index(state["answer_text"]),
                "answer_text": state["answer_text"],
                "label_source": "derived_from_official_endpoint_question_not_official_edited_video_label",
                "source_query_id": state["query_id"],
                "changes_original_answer": state["answer_text"] != question["answer_text"],
                "requirements": [
                    "Use the same explicit displayed-sequence prompt across identity and edits.",
                    "Verify that the copied endpoint frame really displays the annotated state.",
                    "Keep presentation timestamps and input token count fixed while logging source PTS separately.",
                    "Pair reversal requires stable state within both endpoint pairs; exact frame reversal and pair reversal are distinct.",
                ],
            }
    elif category in {"endpoint_change_count", "endpoint_equality_memory"}:
        base["reverse_sequence"] = {
            "status": "conditional_endpoint_comparison_invariance",
            "answer_index": question["answer_index"],
            "reason": "Endpoint equality and Hamming change count are symmetric; require visible endpoint states and a displayed-sequence referent.",
        }
        for condition in ("repeat_first_frame", "repeat_last_frame"):
            base[condition] = {
                "status": "invalid_original_question_presupposition", "answer_index": None,
                "reason": "The original wording presupposes shuffling; static duplication removes it. Do not silently replace the gold label with zero/Yes.",
                "optional_separate_task": "A rewritten first/last displayed-frame comparison can have construction-derived labels, but is a separate synthetic task.",
            }
    return base


def recovery_status(v1, pilot, tail):
    full = next(row for row in v1 if row["method"] == "full_reference" and row["budget_pairs"] == 4)
    empty = next(row for row in v1 if row["method"] == "no_memory")
    compressed = [row for row in v1 if row["method"] in SELECTORS]
    losses = [row for row in compressed if full["reward"] == 1 and row["reward"] == 0]
    rescues = [row for row in compressed if full["reward"] == 0 and row["reward"] == 1]
    local_success = [row["method"] for row in tail if row["method"] != "no_memory" and row["reward"] == 1]
    pilot_success = [row["condition"] for row in pilot if row["condition"] != "no_memory" and row["official_reward"] == 1]
    if losses:
        status = "reference_success_with_compressed_failures"
    elif full["reward"]:
        status = "reference_and_all_compressed_conditions_correct"
    elif rescues or local_success or pilot_success:
        status = "reference_failure_with_other_visual_success"
    else:
        status = "no_recorded_visual_condition_recovers_answer"
    return {
        "status": status, "full64_reference_correct": bool(full["reward"]),
        "full64_gold_margin": full["gold_margin"],
        "no_memory_correct": bool(empty["reward"]),
        "compressed_failures_after_reference_success": [
            {key: row[key] for key in ("method", "budget_pairs", "gold_margin", "payload_sha256")}
            for row in losses
        ],
        "compressed_successes_after_reference_failure": [
            {key: row[key] for key in ("method", "budget_pairs", "gold_margin")}
            for row in rescues
        ],
        "strict_pilot_visual_success_conditions": pilot_success,
        "matched_tail_visual_success_conditions": local_success,
        "causal_evidence_loss_established": False,
        "limitation": "Full reference is 64 sampled frames, not complete evidence or a perceptual oracle. Success/failure contrasts change content and context; no QA-grounded evidence interval was established by action-coverage overlap.",
    }


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_report(report):
    summary = report["summary"]
    lines = [
        "# 原36题因果任务与可恢复性审计", "",
        "本审计仅读取已有记录，未加载模型或运行 GPU。开发全体预先限定为原8视频的全部36题；已有结果已看过，因此这是开发方案冻结，不是追溯性预注册或独立验证。", "",
        "## 结论与边界", "",
        f"官方字段逐题核验通过；自由生成试点 {summary['pilot_records']} 条、v1读出 {summary['v1_records']} 条、固定尾部诊断 {summary['tail_records']} 条。完整条件输出保存在本报告逐题部分和 JSON，未删去负例。",
        "",
        f"64帧参考答对 {summary['full64_correct']}/36，4对帧 uniform {summary['uniform4_correct']}/36，spectral_balanced {summary['spectral4_correct']}/36。明确端点/顺序/变化题共8题；唯一官方 memory 题在无视频下也答对。",
        "",
        "video_1580 末态字母题的完整64帧参考也错，不能称为‘完整证据可恢复但压缩丢失’。最后一帧对参考答对仅表明某一局部输入可读；其 margin=+0.25，且与16帧预算不同。固定尾部同预算条件仍错，尚未确定干扰机制。",
        "",
        "‘full_reference正确、压缩条件错误’仅是待解释的配对现象。没有问题级最小证据区间和匹配干预，不能由官方动作覆盖把错误归为问答证据被删除；更不能把归一化时间能量占比上升解释为保留了有用信息。",
        "",
        "## 最小可运行干预", "",
        "1. 全36题、固定原16帧、固定模型/提示/时间位置/像素预算，先重复 identity 与 no-op hook；保存所有合法选项 logits、token ids、mask 和输入哈希。",
        "2. 第27层仅视觉位置 FFN 输出置零作结构负对照：本层 attention 已完成，后面只有逐token残差/归一化/LM head，因此单次前向的末端答案 logits 应完全相同。若变化，先否决 hook/mask/缓存实现；不要称为机制。",
        "3. 第13层仅视觉位置 FFN 可进入后续 attention，才存在通向答案的路径。先固定一个强度（例如残差保留系数0.5）；同时保留均值、消融时间残差，并与同扰动范数的随机时间正交旋转对照比较。全token FFN 消融会直接改变答案位置，不回答视觉记忆机制问题。",
        "4. 内容顺序控制固定同一组16帧：identity、整帧对逆序、固定seed17的中间帧对置换且首尾不动。逐帧倒放另列，因其改变对内运动。保留展示时间格递增，源PTS另存；禁止将逆序原PTS错误喂给读者。",
        "5. 重复同一已验证末态静态帧16次作零像素变化控制，输入token预算与时间位置一致。重复最后两帧仍可能含帧间运动，须单独命名；重复帧在因果decoder中仍可产生位置/上下文导致的隐状态变化。",
        "",
        "内部FFN干预可沿用原题金标，因为输入任务不变。像素顺序编辑则需统一提示说明‘问题指所呈现序列’，否则原提示中的 original video 仍指原视频，不能自动换金标。6道首末字母题可从官方配对端点标签派生候选反事实；2道计数/同序题重复静态后违反‘已打乱’预设，不可给原问句直接改成0/Yes。其余题仅做未评分敏感性记录，直至独立建立编辑后标签。",
        "",
        "## 可评分指标", "",
        "主诊断 margin = z_gold − max(z_other)，选项tie沿用原argmax并单独报告。BF16 logits 离散，小幅margin变化不可当显著提升。全36题保留正确率、视频宏平均与所有逐题margin；8个端点题及2个反转会翻标的端点题是事前按官方语义确定的诊断子集，不能替换全体结果。",
        "",
        "合法反事实配对同时报告：原始金标在原序列的margin、派生金标在编辑序列的margin、二者同时答对的比例。仅原答案概率下降不算跟随正确反事实。无视频和选项顺序轮换控制检查答案先验；轮换须所有预定任务统一执行。",
        "",
        "## 开发门槛与否决规则", "",
        "- 完整性门槛：全部36题/全部预定条件完成，输入哈希与预算一致；身份/时间位置/视觉mask核验通过。identity和末层负对照须答案logits逐元素不变；否则停止解释中层效应。",
        "- 可恢复性门槛：先验证端点帧可读且候选反事实标签合法。若新固定16帧参考没有可重复的视觉依赖错误，或模型只靠无视频先验答题，则不能启动‘修复记忆’主张。",
        "- 任务门槛：冻结一个干预与强度后，全36题净修复数至少为正，视频宏平均不下降；按原有temporal_state定义的8题无净退化。逐题给出修复与损伤，不能只报margin最好的个例。此门槛仅决定是否值得继续，不能证明泛化。",
        "- 机制门槛：中层作用必须超过重复identity波动，且不能被同扰动范数随机时间控制或简单时域操作完整解释。若只有最后一帧或recent有效，应归为末态读取策略；若静态输入同样有效，不能称为动态选择性。",
        "- 频谱专属性门槛：同输入、同保存字节、同选择预算，与kcenter、temporal_centered、temporal_difference、uniform、recent及保序低秩/时域重建比较；方法必须改善真实QA而不只是谱能量/覆盖。若频谱等价的时域实现效果相同，只能主张该结构约束，不能主张频域创造信息。",
        "- 否决：完整参考都不懂且局部证据也不可读；编辑目标无法合法定标；收益只来自格式解析改变、答案泄露、更多帧/像素/token、选例或多重试验；任一出现均停止对应因果主张。",
        "",
        "## 新确认样本方案", "",
        "原8视频和已查看的40视频均已成为开发历史，绝不重新叫作未见测试。只在开发门槛通过后，从尚未推理的视频建立排除ID与文件SHA清单；固定种子、任务定义、方法/强度、主要预算、读者、主比较和停止规则，然后才读取模型输出。",
        "",
        "先由不见模型输出的标注过程核验官方问题、最小证据帧与反事实标签；完整参考可读性作为事先定义的分层指标，不能按某方法成功再筛题。保留原始全体与可读层两套分母。确认按视频分组；人数未知不能称跨主体。正式样本量根据最小有意义差异和视频内相关性预先估计，不以看到显著性为停机点。多层/多强度只在开发探索，确认仅一项主比较；使用新的第二小模型作迁移。",
        "",
        "## 端点任务原始结果", "",
        "| 题目 | 官方答案 | 无视频 | full64 margin | uniform4 margin | spectral4 margin | 解释 |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["questions"]:
        if not row["audit_category"].startswith("endpoint_"):
            continue
        indexed = {(entry["method"], entry["budget_pairs"]): entry for entry in row["v1_readout"]}
        recovery = row["recoverability"]
        lines.append(f"| {row['query_id']} | {cell(row['official']['answer_text'])} | {indexed[('no_memory', 0)]['raw_answer']} | {indexed[('full_reference', 4)]['gold_margin']:.3f} | {indexed[('uniform', 4)]['gold_margin']:.3f} | {indexed[('spectral_balanced', 4)]['gold_margin']:.3f} | {recovery['status']} |")
    lines.extend(["", "## 参考正确而4对压缩错误的全部题目", "", "| 题目 | 方法 | 参考margin | 压缩margin |", "| --- | --- | ---: | ---: |"])
    for row in report["questions"]:
        recovery = row["recoverability"]
        for loss in recovery["compressed_failures_after_reference_success"]:
            if loss["budget_pairs"] == 4:
                lines.append(f"| {row['query_id']} | {loss['method']} | {recovery['full64_gold_margin']:.3f} | {loss['gold_margin']:.3f} |")
    lines.extend(["", "## 全部36题与全部原始条件", "", "P为原144次自由生成主评分，F为格式失败；只保留原结果。v1表每个方法均列2/4/8对条件，单元格为原答案/正确位/margin；full_reference实际始终32对。JSON保留实际帧对、字节数、logits、token数、forward_id与载荷哈希。", ""])
    for row in report["questions"]:
        official = row["official"]
        lines.extend([
            f"### {row['query_id']}", "", cell(official["question"]), "",
            "；".join(f"{chr(65 + index)}. {cell(option)}" for index, option in enumerate(official["options"])), "",
            f"官方答案：{chr(65 + official['answer_index'])} / {cell(official['answer_text'])}；area={official['area']}；reasoning={official['reasoning']}；tags={cell(', '.join(official['tags']))}。审计类别 `{row['audit_category']}` 为本报告分类，未替换官方标签。", "",
            "| pilot条件 | 原始输出 | 原主评分 | parser |", "| --- | --- | ---: | --- |",
        ])
        for entry in row["pilot"]:
            lines.append(f"| {entry['condition']} | {cell(json.dumps(entry['raw_answer'], ensure_ascii=False))} | {entry['official_reward']} | {entry['parser']['status']} |")
        lines.extend(["", "| v1方法 | 2对 | 4对 | 8对 |", "| --- | --- | --- | --- |"])
        indexed = {(entry["method"], entry["budget_pairs"]): entry for entry in row["v1_readout"]}
        empty = indexed[("no_memory", 0)]
        lines.append(f"| no_memory | {empty['raw_answer']}/{empty['reward']}/{empty['gold_margin']:+.3f}（0对） | — | — |")
        for method in ("full_reference", *SELECTORS):
            values = [indexed[(method, budget)] for budget in (2, 4, 8)]
            display = [f"{entry['raw_answer']}/{entry['reward']}/{entry['gold_margin']:+.3f}" for entry in values]
            lines.append(f"| {method} | {' | '.join(display)} |")
        if row["matched_tail"]:
            lines.extend(["", "固定尾部诊断（实际帧对数不同的reference单列）：", "", "| 方法 | 实际对数 | 原答案 | 正确 | margin |", "| --- | ---: | --- | ---: | ---: |"])
            for entry in row["matched_tail"]:
                lines.append(f"| {entry['method']} | {entry['actual_pairs']} | {entry['raw_answer']} | {entry['reward']} | {entry['gold_margin']:+.3f} |")
        lines.extend(["", f"可恢复性：`{row['recoverability']['status']}`；压缩导致证据丢失的因果结论：**尚未建立**。", "", "| 干预 | 定标状态 | 候选金标 |", "| --- | --- | --- |"])
        for condition, details in row["counterfactuals"].items():
            answer = details.get("answer_index")
            lines.append(f"| {condition} | {details['status']} | {chr(65 + answer) if answer is not None else '未赋值'} |")
        lines.extend(["", "条件标签只在JSON列出的前提核验后启用；未定标题不报告编辑后准确率。", ""])
    lines.extend(["## 输入来源与完整性", "", "| 文件 | SHA-256 |", "| --- | --- |"])
    for name, source in report["sources"].items():
        lines.append(f"| {name} | `{source['sha256']}` |")
    return "\n".join(lines) + "\n"


def build_report(root):
    paths = {
        "official_annotation": root / "data/perception_test_sample/annotations/sample.json",
        "questions": root / "data/perception_test_sample/questions.jsonl",
        "pilot_records": root / "outputs/qwen3_qa36_pilot/records.jsonl",
        "pilot_config": root / "outputs/qwen3_qa36_pilot/run_config.json",
        "v1_records": root / "outputs/spectral_memory_dev_readout/records.jsonl",
        "v1_config": root / "outputs/spectral_memory_dev_readout/config.json",
        "v1_status": root / "outputs/spectral_memory_dev_readout/status.json",
        "tail_records": root / "outputs/matched_tail_readout/records.jsonl",
        "tail_config": root / "outputs/matched_tail_readout/config.json",
        "frames_manifest": root / "data/perception_test_frames64/frames.manifest.jsonl",
        "audit_source": Path(__file__).resolve(),
    }
    sources = {name: {"path": str(path.resolve()), "sha256": digest(path)} for name, path in paths.items()}
    questions = read_jsonl(paths["questions"])
    require(len(questions) == 36 and len({row["query_id"] for row in questions}) == 36, "Expected original 36 unique questions")
    require({row["video_id"] for row in questions} == EXPECTED_VIDEOS, "Development video membership changed")
    annotation = read_json(paths["official_annotation"])
    pilot_config = read_json(paths["pilot_config"])
    v1_config = read_json(paths["v1_config"])
    require(pilot_config["questions_sha256"] == sources["questions"]["sha256"], "Pilot question hash mismatch")
    require(v1_config["questions_sha256"] == sources["questions"]["sha256"], "V1 question hash mismatch")
    require(pilot_config["frames_manifest_sha256"] == sources["frames_manifest"]["sha256"], "Frame manifest hash mismatch")
    require(read_json(paths["v1_status"])["status"] == "complete", "V1 readout incomplete")
    known = {row["query_id"]: row for row in questions}
    groups = {}
    for name in ("pilot_records", "v1_records", "tail_records"):
        records = read_jsonl(paths[name])
        grouped = defaultdict(list)
        for row in records:
            require(row["query_id"] in known, f"Foreign question in {name}")
            require(row["official_answer_index"] == known[row["query_id"]]["answer_index"], "Record gold mismatch")
            grouped[row["query_id"]].append(row)
        groups[name] = grouped
    require(sum(map(len, groups["pilot_records"].values())) == 144, "Pilot record count changed")
    require(sum(map(len, groups["v1_records"].values())) == 1008, "V1 record count changed")
    require(sum(map(len, groups["tail_records"].values())) == 66, "Tail record count changed")
    endpoints = defaultdict(dict)
    for question in questions:
        category = audit_category(question)
        if category in {"endpoint_first_letters", "endpoint_last_letters"}:
            position = "first" if category == "endpoint_first_letters" else "last"
            endpoints[question["video_id"]][position] = {
                "answer_text": question["answer_text"], "query_id": question["query_id"],
            }
    audited = []
    expected_v1 = {("no_memory", 0)} | {
        (method, budget) for method in ("full_reference", *SELECTORS) for budget in (2, 4, 8)
    }
    for question in sorted(questions, key=lambda row: row["query_id"]):
        query_id = question["query_id"]
        source_id = int(query_id.rsplit(":", 1)[1])
        upstream = next(row for row in annotation[question["video_id"]]["mc_question"] if row["id"] == source_id)
        for local, original in (("question", "question"), ("options", "options"), ("answer_index", "answer_id"), ("area", "area"), ("reasoning", "reasoning"), ("tags", "tag")):
            require(question[local] == upstream[original], f"Official annotation mismatch: {query_id}/{local}")
        require(question["answer_text"] == question["options"][question["answer_index"]], "Official answer text mismatch")
        pilot_rows = groups["pilot_records"][query_id]
        v1_rows = groups["v1_records"][query_id]
        require(len(pilot_rows) == 4 and {row["condition"] for row in pilot_rows} == set(PILOT_CONDITIONS), "Pilot condition coverage mismatch")
        require(len(v1_rows) == 28 and {(row["method"], row["budget_pairs"]) for row in v1_rows} == expected_v1, "V1 condition coverage mismatch")
        pilot = sorted((compact_pilot(row, question) for row in pilot_rows), key=lambda row: PILOT_CONDITIONS.index(row["condition"]))
        v1 = sorted((compact_readout(row, question, "v1_development") for row in v1_rows), key=lambda row: (row["budget_pairs"], row["method"]))
        tail = sorted((compact_readout(row, question, "matched_tail") for row in groups["tail_records"].get(query_id, [])), key=lambda row: row["method"])
        audited.append({
            "query_id": query_id, "video_id": question["video_id"],
            "official": {key: question[key] for key in ("question", "options", "answer_index", "answer_text", "area", "reasoning", "tags", "benchmark_split", "label_source")},
            "official_annotation_pointer": f"/{question['video_id']}/mc_question/id={source_id}",
            "audit_category": audit_category(question),
            "audit_category_is_official_label": False,
            "pilot": pilot, "v1_readout": v1, "matched_tail": tail,
            "recoverability": recovery_status(v1, pilot, tail),
            "counterfactuals": counterfactual_plan(question, endpoints),
            "qa_grounded_evidence_interval": None,
            "action_coverage_used_as_qa_causal_label": False,
        })
    def correct(method, budget):
        return sum(entry["reward"] for row in audited for entry in row["v1_readout"] if entry["method"] == method and entry["budget_pairs"] == budget)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_cpu_audit", "gpu_used": False,
        "scope": "All original 8 development videos / 36 questions; previously observed results; no independent confirmation.",
        "sources": sources,
        "summary": {
            "videos": len(EXPECTED_VIDEOS), "questions": 36, "pilot_records": 144,
            "v1_records": 1008, "tail_records": 66, "full64_correct": correct("full_reference", 4),
            "uniform4_correct": correct("uniform", 4), "spectral4_correct": correct("spectral_balanced", 4),
            "categories": dict(Counter(row["audit_category"] for row in audited)),
            "recovery_statuses": dict(Counter(row["recoverability"]["status"] for row in audited)),
            "official_memory_questions": sum(row["official"]["area"] == "memory" for row in audited),
            "causal_qa_evidence_loss_cases_established": 0,
        },
        "last_layer_visual_ffn_negative_control": {
            "layer": 27, "scope": "single forward; visual positions only; fresh inputs and cache",
            "expected_answer_logits": "exactly unchanged",
            "reason": "Last decoder self-attention precedes its FFN. No subsequent cross-token operation connects modified visual FFN outputs to the answer position.",
            "failure_action": "Reject hook/mask/cache implementation before interpreting middle-layer effects.",
        },
        "counterfactual_label_policy": {
            "all_pixel_edit_labels_pending_displayed_sequence_prompt_and_frame_verification": True,
            "candidate_labels_are_not_official_edited_video_annotations": True,
            "original_shuffling_presupposition_forbids_static_relabeling": True,
        },
        "questions": audited,
    }


def render_endpoints(root, output):
    from PIL import Image, ImageDraw
    manifests = read_jsonl(root / "data/perception_test_frames64/frames.manifest.jsonl")
    selected = sorted((row for row in manifests if row["video_id"] in {"video_1580", "video_8241", "video_9253"}), key=lambda row: row["video_id"])
    canvas = Image.new("RGB", (1024, len(selected) * 328), "white")
    drawing = ImageDraw.Draw(canvas)
    provenance = []
    for row_index, row in enumerate(selected):
        for column_index, frame_index in enumerate((0, 1, 62, 63)):
            path = Path(row["frames"][frame_index])
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail((256, 288))
            left, top = column_index * 256, row_index * 328
            canvas.paste(image, (left, top + 36))
            drawing.text((left + 4, top + 4), f"{row['video_id']} f={frame_index}", fill="black")
            drawing.text((left + 4, top + 18), f"PTS={row['source_frame_timestamps_s'][frame_index]:.3f}s", fill="black")
            provenance.append({"video_id": row["video_id"], "frame_index": frame_index, "path": str(path), "sha256": digest(path)})
    canvas.save(output / "endpoint_frames.png")
    (output / "endpoint_frames.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/causal_task_audit"))
    parser.add_argument("--document", type=Path, default=Path("docs/causal_task_audit.md"))
    parser.add_argument("--render-endpoints", action="store_true")
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    output = (root / arguments.output_dir).resolve()
    document = (root / arguments.document).resolve()
    output.mkdir(parents=True, exist_ok=True)
    document.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(root)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    document.write_text(render_report(report), encoding="utf-8")
    if arguments.render_endpoints:
        render_endpoints(root, output)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
