# 实验记录

## 用途

本文件只保留当前这一套实现下、当前工作区内仍然存在产物的实验结果。

2026-04-03 已按当前实现从零清理并重跑全部训练输出；更早的历史产物和过渡目录已经删除，不再作为当前结论来源。

## 记录规则

1. 每个配置运行单独占一行。
2. 至少记录模型名、核心结构改动、最佳验证 AUC 与时延。
3. 每组实验后补一条简短结论。
4. 当前工作区只有 sample parquet，没有正式完整训练集，因此所有结果都只能做方向判断。

## 当前主线

当前主线配置：configs/unirec_din_readout.yaml

当前 AUC leader：creatorwyx_grouped_din_adapter

当前强对照：creatorwyx_din_adapter

说明：以下结果全部来自 2026-04-03 的 clean rerun，统一使用当前代码、同一份 sample_data.parquet、相同时间切分、统一训练入口与统一评估入口。其间已修复 grouped DIN 的一个实现问题：分路注意力不再从全局截断后的 flat history 里硬拆，而是直接读取各自的分组序列视图，并对空路由做 gate mask。

## 当前活跃实验表

| 编号 | 配置 | 模型 | 核心改动 | 验证 AUC | 验证 PR-AUC | Brier | 平均时延（毫秒/样本） | P95 时延（毫秒/样本） | 结论 |
| ---- | ---- | ---- | ---- | -------: | ----------: | ----: | --------------------: | --------------------: | ---- |
| E001 | configs/baseline.yaml | grok_baseline | unified transformer + candidate isolation mask | 0.6483 | 0.2238 | 0.2407 | 0.4130 | 0.6716 | 当前统一 baseline 可训练，但仍弱于更直接的 target-aware 偏置 |
| E002 | configs/creatorwyx_din_adapter.yaml | creatorwyx_din_adapter | 单路 DIN 目标注意力读 history | 0.6864 | 0.3623 | 0.1560 | 0.1632 | 0.5268 | 当前最强低延迟强对照，PR-AUC 和校准也最稳 |
| E003 | configs/creatorwyx_grouped_din_adapter.yaml | creatorwyx_grouped_din_adapter | action/content/item 三路分组 DIN + gating | 0.7311 | 0.2818 | 0.2480 | 0.1754 | 0.5448 | 修复分路读法后成为当前 AUC leader，说明 grouped route 对全局排序确有收益 |
| E004 | configs/tencent_sasrec_adapter.yaml | tencent_sasrec_adapter | SASRec causal history encoder + candidate pooling | 0.6498 | 0.2501 | 0.2089 | 0.2239 | 0.4834 | 纯序列强基线合格，但仍不如最强 target-aware 方案 |
| E005 | configs/zcyeee_retrieval_adapter.yaml | zcyeee_retrieval_adapter | retrieval-style 多摘要读出 + BCE/pairwise | 0.6712 | 0.3519 | 0.1835 | 0.3124 | 0.8121 | retrieval bias 仍强，且在当前 clean rerun 下稳定进入第二梯队 |
| E006 | configs/oo_retrieval_adapter.yaml | o_o_retrieval_adapter | retrieval-style 多摘要读出 + BCE/pairwise | 0.6712 | 0.3519 | 0.1834 | 0.3010 | 0.8096 | 与 E005 仍几乎同核同分，后续要做 scheme-specific 区分 |
| E007 | configs/omnigenrec_adapter.yaml | omnigenrec_adapter | retrieval-style + Muon/AdamW + combined AUC loss | 0.6427 | 0.3393 | 0.3113 | 0.2775 | 0.8012 | PR-AUC 尚可，但概率质量明显弱于更简单的 retrieval 变体 |
| E008 | configs/deep_context_net.yaml | deep_context_net | CLS/global context unified stack | 0.6009 | 0.1973 | 0.2406 | 0.3823 | 0.6512 | 当前最弱家族之一，暂不适合作为主线 |
| E009 | configs/unirec.yaml | unirec | feature cross + interest token + unified stack | 0.6769 | 0.2358 | 0.2776 | 0.4569 | 0.8492 | AUC 已进入第一梯队边缘，但 PR-AUC 和校准偏弱 |
| E010 | configs/uniscaleformer.yaml | uniscaleformer | memory-compressed history + candidate cross-attention | 0.6277 | 0.2325 | 0.2682 | 0.2798 | 0.7591 | 压缩带来时延优势，但排序收益仍不足 |
| E011 | configs/grok_din_readout.yaml | grok_din_readout | unified backbone + post-transformer DIN readout | 0.6631 | 0.2369 | 0.2715 | 0.4432 | 0.8061 | 比 baseline 更强，但还不是当前 unified 线最优解 |
| E012 | configs/unirec_din_readout.yaml | unirec_din_readout | UniRec + post-transformer DIN readout | 0.6924 | 0.3166 | 0.2630 | 0.4772 | 0.8980 | 当前 unified 路线最强，说明 readout 与 interest token 的组合值得继续深挖 |
| E013 | configs/interformer.yaml | interformer | heterogeneous interaction 多视图交互 | 0.6339 | 0.1932 | 0.2718 | 0.5784 | 1.5435 | 异构交互有信号，但当前效率和精度都不占优 |
| E014 | configs/onetrans.yaml | onetrans | 单流 unified transformer | 0.5832 | 0.1752 | 0.2376 | 0.3129 | 0.6309 | 更像轻量校准型下界，不是当前冲分结构 |
| E015 | configs/hyformer.yaml | hyformer | query decoder + query booster + grouped sequence fusion | 0.6769 | 0.2996 | 0.2485 | 0.5820 | 1.3662 | unified / multi-view 家族里仍然很强，但部署成本明显更高 |

## 当前结论

1. 当前 clean rerun 下，AUC 最强的是 creatorwyx_grouped_din_adapter；修复分路输入后，它已经明显高于 single-route DIN 和 unified 家族。
2. creatorwyx_din_adapter 仍然是最重要的现实强对照，因为它同时具备高 AUC、高 PR-AUC、最好 Brier 和最低时延。
3. grouped DIN 的本轮修复说明，分组路由模型不能从全局截断后的 flat history 上再二次分流，否则 content/item 路由会被严重饿死。
4. unirec_din_readout 仍是当前 unified 路线最强配置，但与 grouped DIN 相比，仍需回答额外结构复杂度是否值得。
5. retrieval 两个变体在当前实现下依旧稳定，说明 ranking bias 没有因为统一底座而失效。
6. sample 验证集里 user 几乎全唯一，GAUC 全部退化为 0.5，因此当前所有 user-level 结论都不可靠。

## 当前截断结论

1. unirec_din_readout 的 truncation sweep 已重跑完成，当前按 AUC 均值看 256 最优，AUC 0.6717 ± 0.0204，平均时延 0.5549 ms/样本；128 的 PR-AUC 均值更高，但 AUC 方差过大。[outputs/truncation_sweep/unirec_din_readout/report.md](outputs/truncation_sweep/unirec_din_readout/report.md)
2. hyformer 的 truncation sweep 也已重跑完成，384 的 AUC 均值最高，AUC 0.7016 ± 0.0404，但平均时延升到 0.8840 ms/样本；如果要更稳的 Pareto 点，128 更合理。[outputs/truncation_sweep/hyformer/report.md](outputs/truncation_sweep/hyformer/report.md)
3. 因此当前可以把 unirec_din_readout 的默认长度判断更新为 256，把 hyformer 的长度判断更新为“384 冲分、128 实用”。

## 下一步实验

1. 以 creatorwyx_grouped_din_adapter、creatorwyx_din_adapter、unirec_din_readout 为三锚点，拆开比较 grouped routing、single-route DIN 和 unified readout 的收益来源。
2. 以 unirec_din_readout、creatorwyx_din_adapter、hyformer 为三锚点，比较 pre-transformer interest token、post-transformer readout、stacked both 的真实收益。
3. 对 HyFormer 做 num_queries、segment_count、fusion_layers、query decoder depth 消融，判断收益来源。
4. 对 zcyeee 与 O_o 做真正的 scheme-specific 适配，而不是继续共享同一核心结构。

## 模板

后续实验请按以下格式追加：

### EXXX

- 配置：
- 输出目录：
- 核心改动：
- 最佳验证 AUC：
- 平均时延（毫秒/样本）：
- P95 时延（毫秒/样本）：
- 观察：
- 结论：
