# Feature-level spectral counterfactual protocol V3

日期：2026-11-02。该协议用于旧开发数据上的机制定位，不把 feature-level 变换解释为重新编码倒放视频，也不读取新确认集。

## 输入与阶段

输入是视觉探针保存的逻辑特征 `X[T,H,W,D]`。允许的阶段是 vision encoder 末端、merger/connector 输出和 LLM 第 0 层视觉输入。特征必须是有限实数，保留实际 Qwen/Smol 的时间、空间布局；不沿通道 `D` 做 FFT。

`feature_counterfactual_v3.py` 的 CLI 可以直接读取 probe NPZ 中的 dense 或 exact-repeat 特征，并生成 CPU 反事实摘要。当前实现不执行模型 forward；模型 hook、完整词表 logits 和后续 Q/K/V 捕获应由单独的 GPU runner 调用这些纯 NumPy 变换。

## 条件

1. **identity**：独立复制原特征，验证 round-trip 和哈希。
2. **temporal_reverse**：沿实际采样时间组反转 `X[::-1]`。对每个 `(h,w,d)` 的完整 temporal power 保持不变，但它不是像素级倒放视频。
3. **3d_allpass_phase**：对 `FFT(T,H,W)` 乘共轭对称单位模 phase mask，再逆变换。联合 3D power 保持，特征 phase/order 改变；零频谱位置使用单位 phase。
4. **spatial_mean_control**：只计算 `mean(H,W)` 后的 temporal spectrum，不替换模型输入，用于量化空间相消。

## 预注册输出

每个阶段和条件保存：

- logical shape、dtype、原始和反事实 SHA256；
- temporal power preservation error；
- joint 3D power preservation error；
- feature RMS、cosine、最大绝对差；
- model-side complete-vocabulary logits、答案 token rank、gold margin；
- 下一层 Q/K/V、attention output、FFN output 和最终 logits；
- model/processor revision、数据 manifest SHA、协议 SHA 和 ResourceGuard 记录。

Identity 的完整词表误差必须不超过 `1e-6`。Temporal reverse 的 temporal power 相对误差和 all-pass phase 的 joint 3D power 相对误差必须不超过 `1e-5`；失败时停止解释，不进入训练。

## 决策规则

- power 保持但 logits 改变：功率不是充分统计量，优先检验 phase/order-aware adapter。
- 变化只在 merger/connector 后出现：优先检验 patch-preserving connector adapter。
- encoder 改变而 LLM0 恢复：候选瓶颈位于中间投影或 token 重排。
- 所有阶段 logits 稳定：停止频谱干预主线，转为负结果或通用 FFN/attention 路径研究。

任何 feature-level 反事实都不能直接用于声称动作方向理解；若要报告任务准确率，必须另定义方向正确的反事实标签，并在独立确认数据上统一运行 native、反事实和所有预注册对照。

## CPU 检查

运行：

```powershell
.\.venv-cuda\Scripts\python.exe -m spectral_memory.feature_counterfactual_v3 --check
```

检查不加载 Torch、模型或 GPU，验证 identity、时间反转功率保持、联合 3D all-pass 功率保持、phase 确实改变和空间平均统计有限性。
