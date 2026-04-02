# TAAC 2026 路线图与待办

## 总目标

围绕“统一序列建模与特征交互”这一主题，构建一个满足时延约束、具备清晰技术叙事、并且能持续提升 AUC 的单模型系统。

当前项目只维护这一套统一实现，不再保留旧栈和过渡配置。

## 当前判断

1. 工作区当前只有 sample parquet，没有正式完整训练集；当前所有指标只能作为方向判断，不能视为最终结构排序。
2. 2026-04-03 已删除历史输出产物，并在当前实现下完成 15 个配置的从零重训和统一评估。
3. 当前 clean rerun 的 AUC leader 是 creatorwyx_grouped_din_adapter，AUC 0.7311、PR-AUC 0.2818。
4. 当前最强低延迟强对照仍是 creatorwyx_din_adapter，AUC 0.6864、PR-AUC 0.3623、平均时延 0.1632 毫秒/样本。
5. creatorwyx_grouped_din_adapter 的退化已定位并修复：分路注意力应直接读取各自的 sequence view，并且 gate 需要对空路由做 mask；修复后 grouped routing 的 AUC 收益重新显现。
6. retrieval 两个变体稳定处于第二梯队，说明 ranking bias 在当前统一底座下依旧有效。
7. hyformer 仍是 unified / multi-view 家族里的强结构，但时延显著高于 creatorwyx_din_adapter 和 retrieval 方案。
8. sample 验证集里 user 几乎全唯一，因此所有模型的 GAUC 仍退化为 0.5，只能视为评估接口已接通。
9. truncation sweep 已按当前口径重跑完成：unirec_din_readout 的 AUC 均值最佳长度是 256，而 hyformer 的 AUC 均值最佳长度是 384，但它的更优 Pareto 点仍更接近 128。

## 已完成基础设施

1. 训练入口已经支持全部 15 个配置的统一训练与评估。
2. 当前模型实现已扩展为 taac2026/models 下的多文件模型仓，并共享同一套 typed batch 和多视图 encoder。
3. train/evaluate 现已支持命令行覆写 output_dir 和 run_dir，不再需要复制配置文件来区分实验目录。
4. 评估会统一写出 AUC、PR-AUC、Brier、logloss、bootstrap CI 和分桶结果。
5. batch_evaluate 已完成本轮 clean rerun 的 evaluation.json 回填和总表生成。
6. 历史输出目录已清理，当前有效汇总以 outputs/reports/current_experiments/experiment_report.md 为准。

## 模型主线待办

按优先级排序：

1. 以 creatorwyx_grouped_din_adapter、creatorwyx_din_adapter、unirec_din_readout 为三锚点，拆开比较 grouped routing、single-route DIN 和 unified readout 的收益来源。
2. 以 unirec_din_readout、creatorwyx_din_adapter、hyformer 为三锚点，比较 pre-transformer interest token、post-transformer readout 和 stacked both 的收益来源。
3. 对 hyformer 做 num_queries、segment_count、fusion_layers、query decoder depth 消融，确认性能来自 query booster 还是多序列融合。
4. 基于已重跑的 truncation 结果，在 unirec_din_readout 上默认采用 256 作为后续统一主线长度，在 hyformer 上分别保留 128 的实用口径和 384 的冲分口径。
5. 给高基数实体补多哈希 typed embedding，优先覆盖 user、target item、history item。
6. 试验 dense feature 注入方式，比较 dense token、candidate fusion 和低秩投影。
7. 对 zcyeee 与 O_o 做真正的 scheme-specific 适配，而不是继续共享同一 retrieval 核心。
8. 等正式数据到位后，重做 GAUC、热度分桶和稳定性判断。

## 特征工程主线待办

按优先级排序：

1. 扩展 feature_dictionary.json，增加字段分组建议、疑似时间字段、长数组字段和高频字段清单。
2. 输出 user_feature、item_feature、sequence 三部分的字段覆盖率报告，识别稀有字段和稳定字段。
3. 针对 sequence 中不同 group 的 feature_id 做共现统计，为 typed embedding 和 token 保留策略提供证据。
4. 设计“字段进入模型前检查表”，新模型改输入表示前必须先完成一次分析。
5. 当正式数据到位后，补齐更大样本上的字段分布与时间漂移分析。

## 实验方法约束

1. 尽量把“输入表示变化”和“骨干结构变化”拆开验证。
2. 每次实验都记录配置、指标、时延、结论和是否可直接比较。
3. 如果小样本结果和结构直觉冲突，先回到字段分析确认，不要直接过拟合小样本。
4. 如果缺乏新想法，优先补数据分析和查阅相关公开工作，而不是盲目堆模块。

## 近期执行顺序

1. 以 grouped DIN 为当前 accuracy leader、creatorwyx_din_adapter 为 latency-aware 强对照、unirec_din_readout 为 unified 主线代表，做少量结构消融定位收益来源。
2. 以 hyformer 作为 multi-view 代表，继续判断 query booster 相对 target-aware readout 的边际价值。
3. 在统一主线上把默认长度先固定为 256，在 hyformer 上同时保留 128 和 384 两条长度口径做后续比较。
4. 在同口径 leaderboard 稳定后，再进入 typed embedding 与 dense 注入方式消融。

## 结论

当前最可靠的推进方式不是继续扩大模型复杂度，而是让“字段理解 → 输入表示 → 主干建模”三者形成稳定闭环。当前 clean rerun 已经把问题从“谁能跑通”转成“哪种 target-aware 耦合方式最值得保留”。
