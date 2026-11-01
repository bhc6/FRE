# 学习式控制器：独立CPU分析说明

日期：2026-11-01。本文件说明已冻结协议的分析实现，不代表真实模型评估已经完成。协议为 `docs/learned_controller_protocol_v1.md`，SHA256 `3954767ef2662e15eb4934f90eea08ca6d48eaaef39fd2d297e3d9a3df590748`。

## 输入与运行

分析器为 `spectral_memory/report_learned_controller.py`，只依赖Python标准库和NumPy，不导入torch，不加载模型，不执行GPU运算。它独立复算评分，不调用训练或评估程序的评分函数。

完整评估分片通过 `--evaluation-dirs` 一次传入；已传入的每个split必须覆盖全部预声明条件。test为74真实题+32无金标静态条件，validation为40+16，endpoint为32已有金标的seen构造条件。每个条件均必须恰有identity一条及4方法×3seed十二条。未完成分片、缺失方法、重复case、跨分片checkpoint或源码差异均会拒绝生成主表。

```powershell
$evaluationRuns = Get-ChildItem -LiteralPath 'D:\FRE\outputs\controller_evaluation_v1' -Directory
$trainingRuns = Get-ChildItem -LiteralPath 'D:\FRE\outputs\controller_training_v1' -Directory
& 'D:\FRE\.venv-cuda\Scripts\python.exe' -m spectral_memory.report_learned_controller `
  --evaluation-dirs $evaluationRuns.FullName `
  --training-dirs $trainingRuns.FullName `
  --output-dir 'D:\FRE\outputs\controller_analysis_v1'
```

示例评估目录名应替换为实际执行清单列出的完整分片目录；不要将尚在写入的分片加入。输出必须是新目录。可用 `--resource-dirs` 额外传入缓存、smoke、失败proof或重跑的运行目录，以保留真实成本；未提供的历史运行不会被推定已经计入。训练审计要求三个seed目录一起传入。独立数值自检命令如下：

```powershell
& 'D:\FRE\.venv-cuda\Scripts\python.exe' -m spectral_memory.report_learned_controller --self-check
```

## 校验范围

- 冻结协议、源题/帧manifest、每个评估分片的cases/records与源码快照SHA；同一12个checkpoint的文件SHA；模型ID与revision。
- 每个方法共享相同question、gold、选项token映射、帧hash及输入tensor hash。静态gold必须null；真实题元数据逐字段匹配冻结manifest。
- 逐case读取全词表NPZ，验证native、零控制器与恢复forward逐位相等，全部方法logits有限；重新取全部原选项并执行确定性argmax和精确平局记录。
- 独立FP64重算baseline→controller选项KL、预测、正确数、答案改变、gold margin及分布漂移；核验layer27、每head最大扰动与增益上限。这里只验证归档的head范数诊断；归档未保存全部中间激活，因此不宣称从NPZ独立重建整个控制器。
- 训练文件/optimizer/initial/checkpoint/source快照SHA、200步覆盖、所有loss有限及固定权重合成；同seed的四方法200×8训练case序列逐项相同。不会把不同seed间相同配对次数当作额外数据样本。

## 主要估计量与抽样

唯一primary为all74真实题上三seed平均的 `spectral_mlp - time_mlp` video-macro accuracy差。设第s个seed、第v个视频、给定子集的正确数差为d_sv，该视频子集题数为n_v：micro差为 `sum_v d_sv / sum_v n_v`；macro差为 `mean_{v:n_v>0}(d_sv/n_v)`。先形成各seed差，再取三seed平均。

bootstrap固定10,000次、RNG20261101，按两选源层各有放回抽8个视频。每个抽中的视频保留其全部题。所有比较和子集共享相同视频权重。第一种区间固定三个模型seed；第二种区间在每次重复中全局有放回抽三个seed并与同次视频权重交叉，不能在视频内独立重抽seed。identity只有一个模型，计算中按成对需要复用，未伪造三个独立模型。

子集在某次抽样中无题时记为undefined并报告有效重复数，不能填0。95%区间为NumPy percentile的默认线性分位数。16个源cluster及3个训练seed均较少，区间是小样本描述；不报告逐题独立t检验或从多个子集选择显著结果。主表同时公开micro、macro、每个seed及其均值/样本标准差。

## 产物

| 文件 | 内容 |
| --- | --- |
| report.json / report.md | 完整审计摘要、主要差值、所有方法/seed结果、限制与文件SHA |
| main_table.md / main_table.tex | test的identity+4×3seed共13行主表；Markdown另含seed均值/SD |
| validation_main.md / validation_main.tex | 全部validation方法/seed表，与test分开 |
| per_seed_metrics.jsonl | 各split/子集/方法/seed的源数、题数、正确数与指标 |
| paired_comparisons.jsonl / paired_comparisons.tex | 4子集×7比较×2指标，共56行完整成对结果 |
| bootstrap_draws.npz | 同步使用的视频cluster重复权重、全局seed权重、视频/选源层顺序 |
| audited_records.jsonl | 逐case原始记录与独立复算的KL、margin、tie及全词表变化 |
| audit_runs.json | 原始执行配置与其文件hash |
| training_audit.json | 十二次训练、配对采样计划SHA、loss及完整训练配置 |
| resource_ledger.json | 显式传入运行的资源与失败状态、replay差异、目录大小 |

静态重复帧只有KL和top1漂移，不输出准确率；seen端点8题有金标静态门槛和24题mixed结果在独立附录中报告，不纳入test bootstrap。若频谱相对时域无稳定增益，应保留这一结果，不把对identity的收益改称频谱创新已获支持。
