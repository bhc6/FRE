# V3 频谱反事实与适配器评估协议

这是一个 CPU-only 的记录审计与统计协议，入口为 `spectral_memory/evaluate_spectral_v3.py`。它不加载模型、不读取视频、不改变历史结果，也默认拒绝新确认集记录。

## 输入记录

每行至少包含：

- `method` 或 `tag`：`native` 必须存在，其余为候选方法；
- `source_id`/`video_id`、`query_id`/`case_id`；
- `rotation`：0、1、2；
- `supervised`；有监督记录必须有 `gold_answer_index` 或 `original_gold_answer_index`；
- `predicted_answer_index`，或直接提供 `predicted_original_option_index`；
- 可选 `option_logits` 和 `baseline_option_logits`，用于三选项 NLL；
- 可选 `baseline_predicted_original_option_index`。

`semantic_original_option_indices` 用于把 ABC/BCA/CAB 映射回原始答案语义。每个有监督 query 必须恰有三种 rotation；static 记录不带 gold，且只统计预测翻转。

## 聚合与统计

先在 query 内平均三次 rotation，再按 source 平均；presentation 数量不作为独立样本。source-cluster paired bootstrap 默认 10,000 次，seed `20261101`。报告 micro accuracy、source-macro accuracy、三选项 NLL、repair/regression、语义一致性和 static flips。

## 开发门槛

`--role development` 使用 source-macro accuracy 相对 native 至少 `+5 pp`、micro 不下降、static 不翻转的门槛。`--role confirmation` 使用至少 `+3 pp` 的确认门槛。工具只记录门槛结果，不自动选择 checkpoint 或修改数据。

## 数据封存

默认拒绝 `data_role`/`split` 为 `fresh_confirmation`、`confirmation` 或 `internal_routing_v2` 的记录。确认评估必须在方法、checkpoint、统计脚本、主指标和哈希冻结后显式使用 `--role confirmation`。

## 示例

```powershell
.\.venv-cuda\Scripts\python.exe -m spectral_memory.evaluate_spectral_v3 `
  --records outputs/spectral_adapter_v3/dev_records.jsonl `
  --output outputs/spectral_adapter_v3/dev_report.json `
  --baseline-method native `
  --role development
```

本工具不把单个 repair、单个 rotation 或静态无 gold 的预测稳定性解释为准确率收益。若训练器未提供 seed 字段，三 seed 门槛必须由上游训练审计单独核验。
