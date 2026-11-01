# V1：端点内部路由充分性对照

2026-11-01，在 endpoint_control_v1 及其独立审计完成后写定。该开发试验的静态端点8/8正确，混合12/24正确，12组同序列、同选项顺序的首末提问给出相同预测词；连续logit margin并不相同。本轮检验一个内部路由干预是否足以恢复端点选择，不将已有结果称为遗忘，不假设频谱有效。

## 固定输入与条件

完整复用 endpoint_control_v1 的32条视觉QA、源帧哈希、时间槽、提示、选项及模型revision。所有输入tensor逐项核对旧记录哈希。保留16帧、8时间格、480视觉token、618总token，batch=1、无cache、BF16；不读取独立确认集，不换提示或像素。

每题依次执行五种条件，共160次前向：

1. native：标准前向，完整词表logits必须与旧基线逐位相等。
2. passthrough：仅包装同一SDPA dispatch，原参数原样传入；完整logits必须与native逐位相等。
3. causal_sham：显式构建原等价因果mask，不删任何合法边；它是门控的主对照，记录与native的全部差异，不事后放宽阈值或忽略不同答案。
4. first_gate：所有28个decoder层中，仅对最后一个visual token之后的所有query，屏蔽除首时间格以外的visual keys。
5. last_gate：同上，改为保留末时间格的visual keys。

首末时间格各含两张相同源帧、60个空间token。所有非视觉keys（含时间戳与其它文本）仍按原因果mask开放；视觉prefix query完全不改。门控本身只接收时间格索引，不接收答案词、正确选项或样本成败。两个gate均对全部32条运行；问题first/last仅在离线评分中用于指定target gate与opposite gate。

若原mask为None，则按该模型标准SDPA规则建立下三角布尔mask；原调用显式指定is_causal=False时拒绝执行。若原mask为4D布尔或浮点mask，先核验未来key原已屏蔽，再保留其余值并叠加因果约束；布尔False、浮点负无穷或该dtype的最小有限值视为屏蔽。NaN、正无穷、原mask开放未来key均失败退出。显式mask传入时is_causal=False，防止含义不明确的重复规则。batch、序列长、无cache、dropout=0、decoder层数及mask形状不匹配则失败退出。视觉编码器的attention dispatch不改。

## 数值与结构对照

显式mask可能改变SDPA内核/GQA的实现路径，因此主要因果比较是first/last gate对causal_sham，不能直接忽略数值路径差异。native与passthrough的完整词表必须严格相同；native与旧结果严格相同。causal_sham与native的最大绝对差、预测变化全部报告。

在固定0/7/13/20/27层，保存visual token的decoder输出BF16原始字节SHA。每个gate的这些哈希必须与同题causal_sham相等；否则停止解释视觉prefix保持不变。此对照不能声称所有Q/K/V保持不变：门控会改变后续文本状态及后层文本Q/K/V。每次条件前向的所有挂钩与dispatch均在finally中恢复；仅在最后一题的五种条件完成后，额外执行一次无hook标准前向，核验恢复至该题native完整logits（总计161次前向）。

保存每种显式mask的完整CPU数组、源mask描述、目标query与允许visual key索引、每层访问次数；验证没有新增边、没有开放未来key、suffix query的非视觉mask不变、prefix mask不变、目标visual keys保留。所有处理后输入tensor、源码/协议/实现源码、每条件完整词表logits、资源记录可复核。

## 预设读出与解释

完整报告32题每个条件。主要指标为8静态与24混合QA的限制A/B准确率、固定语义men−enm margin、target/opposite gate的预测方向、首末问题的成对内容变化。不把32题当独立视频，不作总体显著性检验。

仅当causal_sham、first_gate和last_gate各自全部8道静态题均正确（两个gate的静态题合计16/16），才把混合恢复作为候选证据；不能仅挑与题目端点匹配的target gate静态8题。若任一门槛失败，所有结果照报，但不能宣称选择性时序修复。即使target gate成功，opposite gate也需报告：若它读出相反端点，支持指定访问路径能控制答案内容；若它也变好，考虑广泛去干扰/提示或位置偏差。

正结果仅支持：在这个已看源样例及固定任务中，改变后视频文本到视觉prefix的直接访问边足以影响/恢复端点回答。它不证明原始attention权重过小、FFN破坏、频谱衰减、长期遗忘或普适记忆改善。由于保留文本中转与视觉之间的路径，负结果也不能排除路由机制。词的读出仍是受限选项评分，不是自由生成。

该干预不训练参数，不做频率滤波，是频谱方法必须面对的简单非频谱机制对照。输入和模型参数存储不变；额外mask、GQA复制、内核/运行耗时须实测，不声称总计算或显存严格相同，也不称为可泛化的新方法。

## 资源与完整性

继续使用未修改的ResourceGuard：单GPU、两CPU线程、0.65 allocator、原全部RAM/温度阈值、300秒上限，无放宽重试。新目录不覆盖、不恢复。要求160条件齐全、恢复检查通过、config与资源summary complete、无resource_stop记录，才允许完整解释。若数值或结构核验失败，则保留失败产物、停止运行，不选择已完成的有利题目。
