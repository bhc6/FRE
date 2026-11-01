# 跨模型只读内部诊断V2

日期：2026-11-01。此协议只描述开发诊断；没有干预、训练或未见确认评分。比较Qwen3-VL-2B与SmolVLM2-500M，分别单进程串行运行，沿用ResourceGuard原阈值与300秒上限，不同时占GPU。

## 数据与布局

仅使用已看V1 validation的8个源、40题冻结manifest。每源按 `(video_id, query_id)` 排序取第一题，共8题；每题同时运行真实16帧、第一帧重复16次、末帧重复16次，共24条件/模型。重复帧条件保留原展示时间戳，去掉原答案金标，不报告静态准确率。

问题manifest SHA为 `76e60bb8b3cc522dfd3aa6323782d753ef58caa78b7dbcbe864224f34d7d3504`，16帧manifest SHA为 `82d93ca7f9acb6fc8062160676a2f7e20885298218a0fd8430bdda11aac6b9b9`。不读取新32源。允许按video-ids分片，但分片完整不等于24条件全矩阵完整。

Qwen使用原backend的相邻双帧分组T=8，最多512视觉token/1024输入token；Smol使用官方512×512逐帧处理，T=16、每帧64视觉token，最多1024视觉token/2048输入token。两个模型预算不同，完整记录处理像素、token数、时间戳格式、frame groups及实际token位置，不把它们称作等预算模型对比。频点对应展示时间网格；原图像采样没有时间低通，不能直接解释成原视频运动频率。

## 原生接口与控制

Transformers固定4.57.1。Qwen捕获 `model.language_model.layers` 中28个decoder层，16个query heads、8个KV heads、head dim128。Smol真实语言架构为Llama，捕获 `model.text_model.layers` 中32层，15个query heads、5个KV heads、head dim64。分别包装其语言模型模块的SDPA registry，不改变视觉encoder接口，不切换eager attention。

每个条件依次执行native完整forward、被动捕获forward、移除全部hook后的恢复forward。完整词表必须逐位相等，输入tensor hash必须不变。记录每层残差加法的原生dtype恒等式。attention重建仅作旁路统计，始终返回原SDPA结果；不使用重建context替换原生context。

## 统计定义

捕获实际SDPA收到的最后文本query、全部keys及values；Q/K已含该模型原生RoPE，Qwen还含head norm，Smol Llama不人为补q/k norm。按真实GQA映射重建FP32的全key softmax与reference context，保留真实mask。原生context从o_proj输入取得，独立报告两者差异。

每层每query head输出visual attention mass、全key entropy及按可见key数归一的entropy、视觉内部条件entropy、按时间组汇总的条件entropy、native/reference/visual context范数、视觉与native context范数比及夹角、原生重建误差和逐时间组attention质量。注意力质量与范数不能直接作为因果贡献。

V频谱只在**真实KV head**的 `[H_kv,T,P,D]` 上沿T做orthonormal rFFT，不先空间平均，不沿channel变换，不复制GQA头充当额外样本。每个空间位置/通道分别变换后累加平方幅值；DC与Nyquist权重1，其余实频点权重2。保存每KV head的DC绝对能量、非DC绝对能量、总能量、非DC/总能量、逐频点能量、每元素能量与Parseval误差。零总能量的比例为null。

另外输出实际post-RoPE K的相同频谱，其静态非DC能量可能来自位置/因果上下文，不能当运动证据。weighted visual context以 `sum_p a[h,t,p] V[mapped_h,t,p,:]` 构成时间序列，并按query head单独输出时间能量；此量明确不同于真实KV head的V频谱。

最后文本位置额外记录decoder输入、attention输出、pre-FFN残差、FFN输出及post-FFN范数，输出FFN/attention、FFN/pre-FFN等比例。FFN大于attention并不证明FFN压制视觉，后续必须通过具体干预检验。head与层都不是独立数据样本，后续汇总以source video为单位；先比较同源real与两个static条件，再讨论模型差异。

## 运行与产物

```powershell
& 'D:/FRE/.venv-cuda/Scripts/python.exe' -m spectral_memory.cross_model_probe_v2 --self-check
& 'D:/FRE/.venv-cuda/Scripts/python.exe' -m spectral_memory.cross_model_probe_v2 --model qwen --output-dir outputs/cross_model_probe_v2/qwen
& 'D:/FRE/.venv-cuda/Scripts/python.exe' -m spectral_memory.cross_model_probe_v2 --model smol --output-dir outputs/cross_model_probe_v2/smol
```

GPU运行仅由根代理执行。输出 `config.json`、`cases.json`、`records.jsonl`、`layers.jsonl`、逐case完整词表控制NPZ、源码快照和资源账目。只保存每层/head统计及频能量，不保存大量原始Q/K/V；因此后续可以复核评分与恒等控制，但不能从归档重新计算所有QKV内部统计。CPU self-check使用随机小型真实Llama/Qwen文本模块核验hook、GQA、残差和whole-vocab恒等，不代表真实GPU模型已经通过。
