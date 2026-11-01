# V1：构造序列端点可读性与干扰诊断

写定日期：2026-11-01。本轮为单一开发样例的明确问题检测，不是原视频动作/洗牌计数QA，不是内部probe采集，也不预设存在长期遗忘。代码入口为`python -m spectral_memory.endpoint_control --output-dir outputs/<新的目录名>`。只接受尚不存在的新目录，不恢复或覆盖旧运行。

## 1. 固定输入与端点金标

调用`causal_intervention.load_frames(Path("outputs"))["video_1580"]`，按原校验读取16张开发采样图。A为`frames[0]`，B为`frames[-1]`。根代理已于2026-11-01实际视觉检查原图：A的三张卡从左到右为`men`，B为`enm`。这一说明只确认源图可见内容，不保证小模型在当前缩放下可读；卡片在全景中很小，识别失败是重要的候选结果。

- A图像SHA256：`4a8328d36754adf7fc5ce8b3be2a3034535f2ac93daf9c712fa50a617e2b678d`。
- B图像SHA256：`52dd2635cbdec968559da7788fb4325404555942255087717ec97f508b278040`。
- 原16帧manifest SHA256：`d141bfb6876efc155697f4673cb2aa43be46208bddcc5e984d35af6021c77de1`。

代码拒绝这些值发生变化的输入。A/B只表示构造源帧，不出现在提示中充当正确答案线索。构造时原封不动使用原16个`timestamps_s`展示槽位，另记录源图的原PTS/解码帧索引，明确二者不同。不得把重复图像的展示时间戳当作原事件真实发生时间。

## 2. 写定的8种序列与32道视觉QA

| 构造序列 | 16个展示槽位 | 首展示帧金标 | 末展示帧金标 |
| --- | --- | --- | --- |
| A16 | A重复16次 | men | men |
| B16 | B重复16次 | enm | enm |
| A8B8 | A重复8次，B重复8次 | men | enm |
| B8A8 | B重复8次，A重复8次 | enm | men |
| A2B14 | A重复2次，B重复14次 | men | enm |
| B14A2 | B重复14次，A重复2次 | enm | men |
| B2A14 | B重复2次，A重复14次 | enm | men |
| A14B2 | A重复14次，B重复2次 | men | enm |

每个序列提两个问题：`first displayed frame`与`last displayed frame`的三张字母卡从左到右拼什么。每题有`A. men / B. enm`及`A. enm / B. men`两种选项顺序，共`8×2×2=32`道视觉QA。金标从构造序列的指定端点计算，不加载或沿用原始shuffling/count题目、选项或答案。

所有提示采用相同模板：

    This is a constructed displayed sequence, not the original recording. Answer only about the frames displayed in this constructed sequence.
    Question: In the {first|last} displayed frame, what word do the three letter cards spell from left to right?
    A. {option_1}
    B. {option_2}
    Answer with the option letter only.

仅替换问题端点与预声明选项顺序，不因结果换措辞。每个构造边界落在相邻帧对之间，保持16帧/8时间格；不得添加裁剪、提高像素预算、另抽帧或补充原始视频。

## 3. 四个无视频提示先验

对同样的两个问题和两种选项顺序，各执行一次`prepare_text`，完全不提供图像，共4条。保留相同文字模板，测量缺少视觉证据时的提示/选项先验。无视频条件没有构造序列事实可供判断，`gold_word`、`gold_answer_index`、`correct`及`gold_margin`全部为null，不报无视频准确率。

先执行全部32条视觉条件，再执行4条无视频条件，共36次正常前向；不挂hook、不调用`passive_probe.capture`、不回写激活、不生成长答案。可复用`passive_probe.forward_logits`执行标准末位置单步前向。

## 4. 评分、主门槛与解释规则

使用固定Qwen3-VL-2B、CUDA/BF16、原backend的SDPA。A/B选项标记必须各自是唯一单token；以两者末位置logit的argmax选答案。保存完整词表logits，另列两个选项分数、二选一归一化概率、相等标记和金标margin。这是限制为A/B的选择读出，不等同无约束自由生成准确率。

**预声明主门槛：A16与B16的全部8道静态视觉QA均正确，才讨论混合序列的失败。** 门槛未通过仍运行并报告全部32道题，不提前截断或选择有利prompt。门槛未通过意味着当前输入预算、缩放、问题表述与读出方式下尚未建立静态端点可读性；它不区分视觉识别、指令遵循或选项偏好等原因。

即使静态门槛通过，混合序列失败也只能先称端点读取/干扰候选，不能自动称长期遗忘。全部4次/序列结果、首末端点、选项反转一致性及4个无视频先验均报告。32题是一个源样例的重复控制，不是32个独立视频，不以此形成总体显著性或普适机制结论。若后续考虑oracle crop，必须另立协议；本轮不执行。

## 5. 资源边界与失败规则

单模型、单进程，固定`ResourceGuard(..., "cuda", timeout_seconds=300)`；保持原2 CPU线程、1 interop线程、0.65 CUDA allocator及全部RAM/温度/空闲显存阈值。每批最多1024总token和512视觉token；视频始终16帧，使用原`DEFAULT_MAX_PIXELS`。BF16不可用、来源变化、任一预算越界、非有限logits、监控失败、超时或进程异常均失败退出，不放宽重试。

ResourceGuard硬停止可能直接以73退出并留下不完整config；只有进程成功、36条齐全、`config.status=complete`、无`resource_stop.json`且资源汇总成功时才是完成。任何部分结果只能标未完成，不据其通过静态门槛或筛选混合结论。

## 6. 审计产物

- `source_snapshot/`及config：运行器、backend、frame loader、标准forward helper、ResourceGuard、协议和实际Qwen3-VL modeling源码快照/哈希；固定模型/processor revision及运行库版本。
- `cases.json`：全部36条精确提示、选项、构造索引与金标；`inputs.json`：源帧、manifest、展示时间戳、源PTS、解码帧索引及人工视觉核验说明。
- 每条件NPZ：完整末位置词表`baseline_logits`、完整输入token IDs、A/B token IDs，无有损压缩；JSONL另存处理后每个输入tensor的形状、dtype和原始字节SHA256。
- `records.jsonl`：全部条件、原始分数、限制选项预测、构造金标/无视频null、batch元数据、文件哈希及前向耗时。
- `summary.json`与`report.md`：静态8题门槛、完整32题及4个先验，不挑选条件；`resources.jsonl`、资源限制/汇总和CUDA峰值用于核对开销。

本协议和代码落盘不表示GPU运行已完成；CPU构造检查与真正模型输出必须分别报告。
