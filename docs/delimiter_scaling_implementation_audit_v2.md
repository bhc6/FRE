# Delimiter Token Scaling：作者实现审计 v2

核查日期：2026-11-01。论文为 ICLR 2026《Enhancing Multi-Image Understanding through Delimiter Token Scaling》。本审计只读取论文和作者代码；没有下载新模型权重、运行GPU或读取本地测试答案/模型分数。

## 1. 固定来源

- 正式论文：`https://proceedings.iclr.cc/paper_files/paper/2026/file/aebf6284fe85a8f44b4785d41bc8249a-Paper-Conference.pdf`。
- 作者仓库：`https://github.com/MYMY-young/DelimScaling`。
- 本次固定 commit：`a2880738e3ed7284953f3611e7c06a3c0ede9334`。
- 各代码文件的来源、大小、SHA256：`outputs/literature_internal_mechanism_v2/delimiter_code_sources.json`。
- Qwen2.5主体代码首次缓存来自main；随后从以上固定commit另存为 `04_delimiter_qwen2_5_pinned_code.py`，两份SHA256均为 `54c33e4ca39e77e8e972e9010c1c82df2014a5b62f80e2a49fd90a049aedf389`，已验证字节一致。

## 2. 真实操作位置

论文 Eq2 对属于真实delimiter集合的token hidden乘正数λ>1。作者实现是在完整 decoder block返回后，读取 `layer_outputs[0]`，再在选定层号原地修改对应token的residual hidden。它不在attention softmax之后加偏置，不直接修改Q/K/V投影输出，也不是只缩放attention branch。

| 模型路径 | token定位 | residual缩放 | 执行条件 |
|---|---|---|---|
| Qwen2.5-VL | `04_delimiter_qwen2_5_code.py` 1231–1235，从config读vision_start/end ID，再对temp_input_ids作nonzero | 1272–1283，完整block输出后 | 层i在select_layer中，delim_scaling为真；start/end位置数各自>1 |
| InternVL3的HF实现 | `04_delimiter_internvl_model.py` 740–744传原input_ids与model_name到Qwen2；`04_delimiter_qwen2_model.py` 457–461识别ID | Qwen2文件498–507，完整block输出后 | 同上，且hidden_states.shape[1]>1 |

上述行号对应本地缓存文件，不是网页动态行号。Qwen2.5 block内input_layernorm在1095行；下一层先归一化再作attention投影。这一顺序对机制解释至关重要。

## 3. token、层和λ：哪些已公开

| 项目 | 可确认内容 | 证据 |
|---|---|---|
| Qwen2.5 delimiter | `<|vision_start|>`、`<|vision_end|>`，ID从对应config读取 | 论文Appendix A.3，作者Qwen2.5代码 |
| InternVL3 delimiter | `<img>`=151665，`</img>`=151666 | 作者Qwen2代码458–459；论文中`<\img>`拼写不能替代tokenizer定义 |
| Qwen2.5-VL-3B / Mantis示例 | λ=8，0-based层0,1,2,3；batch_size1 | README 58–65；论文Table A5层选择也为0–3 |
| Qwen3B其他benchmark | 同模型固定选层，λ可能另调；不能推定λ8处处最优 | Appendix A.3与Layer Selection |
| InternVL3-1B最优层/λ | **未找到已核实的公开值** | 已查主文/附录、README、CLI、InternVL wrapper及实现；wrapper没有预置最佳值 |
| CLI真正默认 | `delim_scaling=False`，`scale=None`，`select_layer=None` | `04_delimiter_cli.py`285–287 |

README中的公开运行示例不是所有模型共享的默认。代码接受任意用户传入层集合和浮点scale；模型内部可接受某值不等于论文使用过或验证过该值。

## 4. 复现边界和局限

1. 多图不是无条件可迁移到单video。作者§7明确指出缺少frame separator tokens的视频不适用。Qwen作者代码要求start/end各至少两个位置；若整个video只有一对包围token，该分支不执行。不能插入新token后仍宣称同输入的内部干预。
2. `len(nonzero_positions)>1`对整个batch计数，不是逐样本统计。README的batch1避开了跨样本凑足两对的情况；移植时应明确batch行为，不能悄悄混合两个定义。
3. InternVL代码用长度>1阻止单token解码步缩放；Qwen分支不能仅凭形似代码假定完全相同generation行为，应检查实际prefill/input_ids/cache路径。
4. `scale=1`应为identity，需验证原生logits/答案与无hook路径的数值等价；代码允许该值不是已有identity实验的证据。
5. 固定中间层、λ=.5/2/4、Smol的`fake_token_around_image`，均属于机制移植/消融。λ<1在论文附录有反向消融，但不等于主方法设置；Smol没有本篇的官方模型结果。
6. Appendix A.3明确多图任务用10% test调层/λ，多文档/多表直接test调参。本地独立test必须保持封存，公开论文最佳值也不能支持本地test后挑参数。

## 5. 机制解释需要实际张量检验

对正λ，零ε的RMSNorm满足 `RMSNorm(λh)=RMSNorm(h)`。因此post-block residual缩放不能直接等同于下一层归一化后Q/K/V同时乘λ。非零ε、数值精度、附加归一化和具体模型结构会带来差异；这些均应实测。

相对旧residual，后续attention/FFN更新可能被抑制，这是待检验解释。主任务的 `docs/delimiter_scaling_mathematics_v2.md`单独讨论严格等价条件；相关先行工作与新颖性边界另行核查。论文Appendix A.2也做了投影后的Q/K/V单独缩放实验，这些是不同算子，不能用其结果证明residual缩放立即放大Q/K/V。

建议记录同一次forward的delimiter处：block输出、next input norm、Q/K/V、attention output及FFN output，并验证no-extra-forward、cache位置和mask。任何最终结论须以模型实际张量和冻结样本的native准确率为依据。
