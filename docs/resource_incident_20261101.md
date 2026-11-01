# 2026-11-01 重启事件与实验资源记录

本报告区分已保存事实、可核实的系统错误定义及尚未确定的原因。SmolVLM2 迁移运行与多次系统重启相关，当前停止且不完整；现有材料不足以断言“显存不足导致重启”或锁定具体硬件、驱动、虚拟化组件。

## 1. 范围、时间与操作边界

事件取证于 **2026-11-01 19:29:27 +08:00** 进行，窗口为当日 **09:29:27–19:29:27 +08:00**，对应 **01:29:27–11:29:27 UTC**。实际 Windows 时区为 China Standard Time；应用环境显示的 America/Los_Angeles 不用于解释本机日志。本报告的未另注明时间均为 UTC+08:00。

仅保存 System 日志的四个 provider：Microsoft-Windows-Kernel-Power、Microsoft-Windows-WER-SystemErrorReporting、Microsoft-Windows-WHEA-Logger、Display。共 76 条，查询无截断、无报告错误。事件的完整消息、XML、EventData、原始记录号和双时区时间均保存在 outputs/resource_incident/events_10h.json。

本次取证只读取已落盘记录及系统配置，不读取或复制 MEMORY.DMP，不安装调试器，不调整驱动、电源、VBS、Hyper-V 或 BIOS，不请求重启，不启动推理。CPU 处理采用单进程，数值库线程上限设为 2。

用户随后重新明确授权“使用 GPU”。根代理安排单进程、16 帧 Qwen FFN pilot，2 个 CPU 线程、空闲 RAM/VRAM 预检与 0.65 的 CUDA allocator fraction；**这项授权不等于重启原因已经解决，本轮未安排恢复 Smol 长任务**。allocator 限额只约束相应 PyTorch 进程的分配器，不能保证避免系统、驱动、虚拟化或硬件故障。实际探针完成状态另见各次 `probe.json`，不根据计划推定结果。

## 2. 已确认的事件时间线

| 时间（+08:00） | 事件及记录号 | 已保存信息 |
| --- | --- | --- |
| 09:55:49.481–49.649 | WHEA 17，267525/267526/267528/267529/267530 | 5 条“已更正的硬件错误”；NVIDIA PCIe 设备 3 条，NVMe 控制器 2 条 |
| 18:12:07.800 | Kernel-Power 41，267741 | 非正常关机后启动；BugcheckCode 十进制 257，即 0x101 |
| 18:12:22.491–22.574 | WHEA 17，267842–267846 | 同类 5 条已更正 PCIe 错误 |
| 18:12:48.729 | WER 1001，267863 | 0x00000101；消息报告转储路径 C:\Windows\MEMORY.DMP |
| 18:57:50.036 | Kernel-Power 41，267959 | BugcheckCode 十进制 131073，即 0x20001 |
| 18:58:04.807–04.933 | WHEA 17，268062–268066 | 同类 5 条已更正 PCIe 错误 |
| 18:58:06.746 | WER 1001，268069 | 0x00020001；消息报告转储路径 C:\Windows\MEMORY.DMP |
| 19:20:35（文件修改时间） | Smol records.jsonl | 已落盘记录最后修改；该时间不是单次推理或崩溃的精确时间 |
| 19:21:11.500 | Win32_OperatingSystem.LastBootUpTime | 采集时系统的最近启动时间 |
| 19:21:14.298 | Kernel-Power 41，268178 | BugcheckCode 十进制 131073，即 0x20001 |
| 19:21:27.598 | WER 1001，268249 | 0x00020001；消息报告 C:\Windows\Minidump\110126-16406-02.dmp |
| 19:21:28.796–28.866 | WHEA 17，268282–268286 | 同类 5 条已更正 PCIe 错误 |

因此窗口内有 **3 条 Kernel-Power 41 和 3 条对应的 WER 1001**，不是只有两次相同错误。WER 时间是事件写入/报告时间，不能直接当作实际故障发生时刻；系统启动后补报的事件也不能据排序推导触发因果。未验证消息中转储文件目前是否仍存在或对应何次故障，未分析转储。

三条 WER 的 bugcheck 参数原样保留：

- 0x00000101：0x000000000000000c、0x0000000000000000、0xffffad01325aa180、0x0000000000000009。
- 两条 0x00020001：0x0000000000000028、0x0000000000000001、0x0000000029b92701、0x00000000fc800000。

Microsoft 将 0x101 定义为 CLOCK_WATCHDOG_TIMEOUT：预期的次级处理器时钟中断未在分配时间内收到。这一定义本身不证明 CPU 硬件损坏。[S1] 0x20001 为 HYPERVISOR_ERROR；官方将四个参数列为 Reserved，本报告不擅自解码或据此指认某一驱动。[S2] Kernel-Power 41 说明系统未正常关机后重启，单独无法确定原因。[S3]

## 3. WHEA 与设备对应

20 条 WHEA 均为事件 17，消息均标记“已更正的硬件错误”，分成表中四组，每组 5 条。12 条记录的 PrimaryDeviceName 为 PCI\VEN_10DE&DEV_2D18&SUBSYS_207A1EE7&REV_A1，与本机 NVIDIA GeForce RTX 5070 Laptop GPU 的 PNP ID 前缀一致。另 8 条为 PCI\VEN_1E0F&DEV_001A&SUBSYS_001A1E0F&REV_01，Win32_PnPEntity 将其识别为“标准 NVM Express 控制器”，PNPClass 为 SCSIAdapter。两个设备在查询时 Status 均为 OK、ConfigManagerErrorCode 均为 0。

WHEA 对已更正与未更正错误有明确区别；不能把这些已更正事件直接等同于本次致命错误根因。[S4] 相同 PCIe 消息还出现在 09:55，早于 Smol 输出目录内文件的 18:10 创建时间；它们并非仅在最后一次推理后出现。采集窗口内 Display provider 事件数为 0，这不能排除 GPU 或驱动问题。设备查询时的 OK 状态也不能证明此前没有瞬时故障。

## 4. 采集到的环境

| 项目 | 只读快照 |
| --- | --- |
| 系统 | Microsoft Windows 11 家庭版 中文版，25H2，10.0.26200.9711，64 位 |
| 设备 | HONOR DRB-P |
| BIOS | HONOR 3.04；CIM ReleaseDate 为 2025-08-19T08:00:00+08:00 |
| 物理内存 | 33,820,106,752 bytes（Win32_ComputerSystem） |
| HypervisorPresent | true；不能据此归因或推断所有虚拟化组件配置 |
| NVIDIA 驱动 | RTX 5070 Laptop GPU，32.0.15.9174 |
| Intel 驱动 | Intel Arc 140T GPU (16GB)，32.0.101.6554 |
| 其他显示设备 | Honor Virtual Display Device 1.0.23.3；IddDesk Device 25.0.0.4 |
| 其他观测 | Honor Virtual Display Device 的 CIM Status 为 Error；未做进一步诊断，不能据此归因 |

Win32_VideoController.AdapterRAM 不作为实际显存容量依据；本报告采用运行时保存的 device_total_memory_bytes，并标明它只是当时的设备报告值。

在 PATH、当前用户 WinDbg Appx 包、8 个常见 Windows Kits/WindowsApps 可执行文件路径和三处卸载注册表位置中，均未发现 WinDbg/cdb。此结果只覆盖列出的搜索范围，不证明全机或其他用户目录必然不存在；没有下载或运行调试器。

## 5. Smol 未完成运行与资源证据

来源为 outputs/smolvlm_confirm_b4_readout/records.jsonl 的完整逐行读取，仅统计记录完整性、forward ID 与资源字段；**未计算部分准确率、方法排名或迁移统计结论**。

| 项目 | 已核实数值 |
| --- | ---: |
| JSON 可解析的已落盘记录 | 1,633 |
| 计划总记录 | 2,002 |
| 未落盘的剩余记录 | 369 |
| 不同 forward_id | 1,534 |
| 标记复用的行 | 99 |
| 保存的 peak_memory_bytes 最大值 | 5,051,585,536 |
| 保存的 peak_reserved_memory_bytes 最大值 | 9,604,956,160 |
| 保存的 device_total_memory_bytes | 8,546,484,224 |

PyTorch 的 allocated 统计张量占用，reserved 统计缓存分配器管理的内存，两者不是相同口径，也不是整机显存驻留量或系统 RAM。[S5] 此处 reserved 数值大于记录的物理设备容量，值得后续核对分配器/Windows 驱动语义及采样方式；**这个数值关系不构成 OOM、共享内存迁移或重启因果的证据**。本次未获得崩溃瞬间的 VRAM 驻留量、系统提交量、温度、功耗、驱动栈或虚拟化堆栈。

原始 status.json 仍写 running、completed_records=166；它是中断前的过时状态，不能代表现在仍在运行。原始 execution_notes.json 曾记录运行进程消失、未见 Python OOM traceback；结合后来取得的系统日志，应将整体经过表述为与系统重启相关的中断，不能继续淡化为普通进程退出。由于缺少每次推理的绝对时间戳，不能把每一条中间记录精确归配到某次重启，也不能证明哪个输入触发故障。

原始记录和状态文件均未改写。当前判断另存于 smol_incomplete_resource_summary.json；源文件大小、创建/修改时间、SHA-256 见 evidence_manifest.json。已完成的 Qwen 确认结果保持有效的原有范围；Smol 的缺失数据不支持“完整独立读者迁移验证已通过”这一声明。

## 6. 已知限制及后续条件

已确认的是三次系统级异常重启报告、两种 bugcheck、重复的已更正 PCIe 错误、运行中断与资源计数。尚未确定的是它们之间的因果关系，以及 CPU 调度/虚拟化、驱动、PCIe、存储、内存、电源或温度等因素的具体作用。

用户重新授权的受限 GPU pilot 仍应按独立新任务记录，不能据一次短运行成功宣告底层故障修复。Smol 长评估保持停止；后续恢复需要明确任务指令及资源/故障评估。当前不自动执行高负载压力测试、不调整系统配置。没有转储分析时，不给出具体故障模块诊断。

## 7. 证据文件与官方定义

本地证据均位于 outputs/resource_incident/：

- events_10h.json：限定 provider 的全部 76 条事件，含原始 XML。
- event_summary.json：按 UTC 排序的 41/1001/17 紧凑时间线。
- system_inventory.json：启动时间、系统、BIOS、内存与显示驱动快照。
- whea_device_inventory.json：两组 PCI vendor/device ID 对应的本机 PNP 设备。
- debugger_inventory.json：调试器搜索范围与空结果。
- smol_incomplete_resource_summary.json：不完整运行的行数及资源统计。
- evidence_manifest.json：证据与源运行文件的大小、时间及 SHA-256。
- summarize_incident.py：纯标准库、单进程的离线汇总脚本；不加载模型、不读取准确率字段、不调用 CUDA。

官方资料于 2026-11-01 核对：

- [S1] Microsoft, [Bug Check 0x101: CLOCK_WATCHDOG_TIMEOUT](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x101---clock-watchdog-timeout).
- [S2] Microsoft, [Bug Check 0x20001: HYPERVISOR_ERROR](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x20001--hypervisor-error).
- [S3] Microsoft, [Advanced troubleshooting for Event ID 41](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/event-id-41-restart).
- [S4] Microsoft, [WHEA Error Processing](https://learn.microsoft.com/en-us/windows-hardware/drivers/whea/error-processing).
- [S5] PyTorch, [CUDA semantics: Memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management).

