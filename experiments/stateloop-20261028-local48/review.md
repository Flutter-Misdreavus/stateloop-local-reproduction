# StateLoop 本机缩小规模复现：最终审查

审查决定：**接受本次执行结果；在冻结配置下，持久状态获得初步支持。** 六组训练、六次完整测试和 18 项状态干预均完成；独立数值审计通过。这里复现的是持久状态的核心对照与部分干预机制，不是论文完整规模的性能。没有追加实验、改变测试集或根据结果降低判定标准。

## 核心结果

主指标为相同种子下 `NLL(reset) − NLL(persistent)`，越大越好。所有模型使用固定 token 预算的最终检查点，在相同的 1,000,000 个测试 prediction tokens、753 篇文档上评价。下表的 PPL 为 `exp(token-weighted NLL)`，不是逐文档 PPL 的均值。

| 模型 | seed 17 NLL | seed 17 PPL | seed 42 NLL | seed 42 PPL |
|---|---:|---:|---:|---:|
| Vanilla Looped | 4.115756 | 61.2985 | 4.129443 | 62.1433 |
| Reset state | 4.074373 | 58.8136 | 4.081618 | 59.2412 |
| Persistent StateLoop | **4.061201** | **58.0440** | **4.064998** | **58.2648** |

| 种子 | 主指标 ΔNLL，nat/token | 文档 bootstrap 95% 区间 | 相对 reset 的 PPL 降幅 |
|---|---:|---:|---:|
| 17 | 0.01317217 | [0.01055559, 0.01570474] | 1.3086% |
| 42 | 0.01661951 | [0.01474005, 0.01848098] | 1.6482% |

两种子的平均 ΔNLL 为 **0.01489584 nat/token**；对应 PPL 比值的几何平均改善 **1.4785%**。预先冻结的条件是“两种子均为正，均值至少 0.005”，本次满足。每个区间由 2,000 次配对文档重采样得到，条件是这一对已经训练好的模型；它们**不包含训练种子不确定性**，不能把区间不跨零解释为跨训练随机性的强显著性。

![测试指标与配对效应](figures/primary-comparison.png)

**事实：** reset 与 persistent 参数量、初始化、数据顺序和训练 token 数相同；两对结果均显示 persistent 更好。Vanilla 与 persistent 的 ΔNLL 分别为 0.05455517、0.06444457，但 vanilla 少 591,360 个参数，故只作为辅助比较。

**解释：** 结果支持“跨 loop 保留状态在该配置下有小幅额外收益”。这一训练对照不能唯一判定收益来自信息存储、梯度传播还是优化过程的变化，也不能证明训练已收敛。

## 状态干预及其含义

下表均为相对同一种子、未干预 persistent 模型的 ΔNLL，单位 nat/token。所有干预复用同一个最终检查点。通道均值和标准差来自独立的 500,000-token calibration 划分。

| 干预 | seed 17 | seed 42 |
|---|---:|---:|
| 关闭全部中间 feedback reads | 2.277376 | 1.898683 |
| 关闭 terminal read | 4.431974 | 4.330472 |
| 关闭全部 reads | 5.113211 | 5.245178 |
| 单次替换：保存 S1，替换 S2 | 0.039442 | 0.035520 |
| 单次替换：保存 S2，替换 S3 | 0.091054 | 0.075409 |
| 单次替换：保存 S3，替换 S4 | 0.104890 | 0.093306 |
| 冻结复用 S1：替换此后所有状态 | 1.363024 | 1.081024 |
| 冻结复用 S2：替换此后所有状态 | 0.509437 | 0.433693 |
| 冻结复用 S3：替换此后所有状态 | 0.104890 | 0.093306 |

![状态干预](figures/state-interventions.png)

**事实：** 两个种子中，单次替换较晚状态的损害更大；在 S1/S2 冻结后持续复用，比仅替换一次损害更大。S3 的两种干预数值完全相同，因为四个 loop 下只剩一个后续边界。这也是实现的一项一致性检查。

**解释：** 这些模式与“继续更新状态对已训练模型的预测有帮助”一致。单次替换后仍继续更新，冻结复用也继续执行 backbone；它们没有跳过后续计算。校准仅匹配通道的一阶、二阶统计，不能消除所有分布偏移，因此不能把差值完全归因于某种具体语义信息。

关闭全部 reads 后的巨大退化，衡量的是**已依赖状态分支的模型受到扰动**，不能代替重新训练的 Vanilla 基线，更不能宣称持久状态贡献了 5 nat/token。三个 group masks 没有枚举每个 read 的全部组合，故没有复现精确 Shapley 归因，也不能据 terminal_off 大于 feedback_off 推翻原论文的反馈贡献占比。

## 与论文原文的对应

依据用户提供的 [StateLoop v1](https://arxiv.org/abs/2610.35544) 原文，重点对应第 5.2 节、公式 14、Table 4，以及第 5.3 节、公式 15、Figure 4；逐项实现对应见 [method-traceability.md](method-traceability.md)。

原文 Table 4 的 reset/persistent PPL 为 9.25/9.04，采用约 410M 参数、7 loops、25B training tokens。本文观察到相同的改善方向，但本机采用约 53M 参数、4 loops、每组约 100M tokens，因此**绝对 PPL 和效应大小不可直接对齐**。原文对状态复用的解释是早期状态在邻近边界仍部分可用，而持续更新有作用；本次干预呈现相容的定性模式。原文 Figure 4(a) 的 63.70%/36.30% 是精确 Shapley 份额，本次没有测量该指标。

## 执行与可靠性

冻结配置见 [configs/frozen.json](../../configs/frozen.json)，判定规则见 [configs/protocol.json](../../configs/protocol.json)。使用本机 Tesla V100-SXM2-32GB；8 层共享 backbone、width/state width 384、6 heads、FFN 1024、4 loops、512-token context、identity state mixer。Vanilla 52,795,776 参数；reset/persistent 各 53,387,136 参数。

每组严格完成 **99,991,552 prediction tokens、6,103 optimizer steps**；有效 batch 32，microbatch 8。六组合计 **599,949,312 tokens**。AdamW、3e-4 峰值学习率、2% warmup、cosine 到 10%、梯度裁剪 1；FP16 autocast，状态递推/归一化统计/损失为 FP32，完整跨 loop 反传。优化器和未明确的初始化细节均在执行前记录为本地选择。

训练使用固定版本的 deduplicated Pile 两个 parquet shard，配对种子使用相同的训练样本顺序。dev/calibration/test 按规范化文档哈希独立划分；测试未用于选择模型或检查点。输入审计检查划分、文档不重叠、数据重编码和完整模型初始权重配对；精确文档及对齐 512-token 块检查不能排除所有近重复。数据版本和哈希见 [data-manifest.json](data-manifest.json) 与 [input-audit.json](input-audit.json)。

验证包括 10 项自动测试、FP16 检查、三模型 tiny overfit、真实数据 smoke，以及全部正式训练的基本学习门槛。审查复算了六组 token-weighted NLL/PPL、四组配对 bootstrap 区间，检查了检查点 SHA256、内部运行元数据、全部参数有限性、完整 token 预算和训练日志，并核对了 18 项干预的数值关系。结果见 [completion-audit.json](completion-audit.json)。该审计独立复算统计，但没有重新训练，也不是独立团队的第二套模型实现。

![训练与开发集曲线](figures/learning-curves.png)

图中 dev 曲线使用固定 65,536-token 子集；最终报告使用完整 1M-token dev。原 `validation.jsonl` 的 `tokens` 字段保存的是评价 token 数，训练横坐标按 `step × 32 × 512` 恢复。这个日志字段歧义未改变训练或评价结果。曲线仍有下降趋势；basic learning gate 只验证能正常学习，不证明 100M tokens 足以充分训练。

## 资源、异常与版本

从 2026-10-28 11:38:32 UTC 开始准备，到全部评价结束为 **7.095 小时**；正式训练循环累计 **5.315 小时**，包含每组初末评价的正式训练进程累计 **5.364 小时**。全部测试及干预约 **5.38 分钟**。数值审计和图像检查也已在 48h 内完成。完整资源快照见 [resources.json](resources.json)。

正式训练 PyTorch peak allocated memory 最大 6,487,782,912 bytes，约 **6.04 GiB**；这不是包含驱动/缓存的整卡占用。仓库连同本地数据、检查点、Git 的磁盘分配约 **6.27 GiB**，低于 15 GiB；既有共享 Python 环境通过软链接复用、不计入实验新增文件，磁盘值是结束时快照而非连续峰值。

正式训练 **0 次失败、0 次科学重跑**。准备阶段曾遇 DNS、慢传输和 SSL 中断，修复下载恢复逻辑后完成哈希验证；一段早期下载进度日志被覆盖，未涉及训练结果。数据准备最终执行版本为 `683d2163425f248476374cab9f911fc5c0c57586`。预检原始记录见 [preflight-review.md](preflight-review.md)。正式训练过程中有一次 GitHub 元数据查询 EOF，保留本地提交后按既定流程重试成功，没有影响训练；见 [publication-incidents.json](publication-incidents.json)。目前无未解决的执行问题。

| 正式运行 | 实际执行 commit |
|---|---|
| vanilla seed 17 | `2fae18a4eee399613b0ff1ec916a3408a7bd1d25` |
| reset seed 17 | `11a017f5001679786467b855a47e24c0646e93b0` |
| persistent seed 17 | `cb2a371c35107c6fb154ef5476b72aa3bb44c234` |
| persistent seed 42 | `f88c7d9e96ea87dbf050970970b002123af145d4` |
| reset seed 42 | `c39a3fe476fa44ea8581ddf1bb186e9c467e4c72` |
| vanilla seed 42 | `9a3ae5958abefcb66b3f5cf4fd1da60410cb2dfb` |

六个执行版本的 `model.py/train.py/runtime.py` 内容完全一致，其组合 SHA256 为 `d21db53052f831875d863d734447c70df7ec7a94288265c5817800a17859ad66`。不同 commit 反映分阶段证据提交，不代表改变模型。测试/干预实际执行 commit 为 `cfc824567f722c6408f6d5c7d5416be889559f79`。正式训练开始后没有模型或训练实现修复。

独立审计、图表生成和撰写本报告时的 HEAD 为 `8607def61814d22592844d2ba560e7c953ca7eb0`，已推送 public `main`。它包含审计状态兼容和图表展示调整，不是原训练版本。最终报告会由后续独立提交发布；本文件所属提交可在 GitHub 历史核查，最终发布 HEAD 与远端一致性另在本地 `runtime/publication-verification.json` 记录并在会话交付。避免把报告提交冒充模型运行提交。

## 审查边界与交付

本次结论受两种子、约 1.87 training tokens/parameter、较短 context、较少 loops、两 shard 的数据代表性、FP16 和 CUDA 非完全确定性限制。测试与 dev 来自不同文档样本，绝对 NLL 的差异不应解释为训练期选模收益。没有复现完整 Pile 评价、论文全部规模、其他 state mixers、GRU/state-only 消融、loop-depth 扫描或精确 Shapley。预算内完成冻结范围后停止，没有用剩余时间追逐更好的结果。

审查决定是**接受此次缩小规模的机制复现，并把科学结论限定为初步支持**。本报告和 `review.json` 是最终审查；`assessment.json` 中的 `gpt_review: PENDING` 保留自动测量结束时的原始状态，其文件哈希已被审计绑定。无需把历史测量文件改写成已审查。

公开交付包括实现、冻结配置、数据与检查点哈希、标量指标、三套 PNG/PDF 图、输入/结果审计和本报告。原始文本、模型权重、环境和原始执行日志保留本机。原始执行命令和新机器重跑的限制见仓库 [README](../../README.md)；本机复核可运行 `python scripts/review_results.py` 与 `python scripts/plot_results.py`，依赖仍保留在 `runtime/` 的数据和逐文档统计。仓库克隆本身不包含这些本地产物，不能单凭公开标量重新计算文档 bootstrap。
