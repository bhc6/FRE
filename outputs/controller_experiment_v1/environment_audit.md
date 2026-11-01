# 学习式内部控制器：环境与参数审计

审计 UTC 时间：2026-11-01T14:24:46.395952+00:00。

- GPU 只读查询：`0, NVIDIA GeForce RTX 5070 Laptop GPU, 591.74, 8151`；字段为 index、型号、driver、显存 MiB。
- 实验 Python：`D:\FRE\.venv-cuda\Scripts\python.exe`；`3.13.7 (tags/v3.13.7:bcee1c3, Aug 14 2025, 14:15:11) [MSC v.1944 64 bit (AMD64)]`。
- Torch：`2.10.0+cu128`，来自 `D:\FRE\.venv-cuda\Lib\site-packages`。
- Transformers：`4.57.1`，来自 `C:\Users\123\AppData\Roaming\Python\Python313\site-packages`。
- 实验 venv 允许系统包，Transformers 的用户 site-packages 来源已记录；全局 Python/Torch 与实验 venv 分开列于 JSON。
- 固定模型：`Qwen/Qwen3-VL-2B-Instruct`，revision `89644892e4d85e24eaac8bacfd4f463576704203`。
- 全基座去重冻结参数：**2,127,532,032**。按本地固定 config 实例化 meta 模型计数，未读取权重值；共享 embedding/lm_head 仅计一次。
- 各 MLP 的训练参数为 **18,584**（1152×16+16+16×8+8）；constant 为 **8**；identity 为 **0**。

审计子进程在导入 Torch 前屏蔽 CUDA 并启用离线模式，CPU 数学库、Torch intra-op/inter-op 均限 2 线程。参数及 buffer 设备与 CUDA 初始化状态写入 JSON。没有加载模型权重、执行 forward、读取部分测试预测或修改冻结 runner/protocol。

identity 记录中的 forward_seconds 只计原生 forward_logits，不包括 probe 特征提取。每个 case 另做 zero-controller 与恢复后的 forward 检查，它们计入总 wall time，但不计入 identity 的该次 forward timer。12 个已训练控制器的方法/seed 评估共用 time / signed spectral / phase-blind 全部 bases 提取，time 和 constant 方法也承担 FFT 开销。因此控制器记录表示共同研究实现的运行开销，不能解释为各方法独立优化部署后的速度比较。本次不新增延迟、GPU 峰值显存或吞吐基准。

复现：在项目根目录运行 `.venv-cuda/Scripts/python.exe outputs/controller_experiment_v1/environment_audit_cpu.py`。详细结构化记录见同目录 `environment_audit.json`。
