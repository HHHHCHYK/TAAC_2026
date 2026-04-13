---
title: 实验包与验证记录
icon: material/flask-outline
---

# 实验包与验证记录

## 用途

这页只记录两类信息：

1. 当前分支里真实存在的独立实验包。
2. 当前工作区里可以直接复核的验证证据。

旧版文档里出现过但当前仓库里已经不存在的 `taac2026/experiments/*` 目录、旧模型家族和历史排行榜结果，都不再保留。

## 当前独立实验包

| 实验包 ID        | 目录                        | 说明页                                       | 模型名               | 默认输出目录                 | 主要来源                                                                                                                                      |
| ---------------- | --------------------------- | -------------------------------------------- | -------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `baseline`       | `config/gen/baseline`       | [baseline](packages/baseline.md)             | `baseline_reference` | `outputs/gen/baseline`       | 本仓库维护的 starter/reference package                                                                                                        |
| `grok`           | `config/gen/grok`           | [grok](packages/grok.md)                     | `grok`               | `outputs/gen/grok`           | 从旧 `baseline` 中拆分出来的本地 grok 方案                                                                                                    |
| `ctr_baseline`   | `config/gen/ctr_baseline`   | [ctr_baseline](packages/ctr_baseline.md)     | `ctr_baseline_din`   | `outputs/gen/ctr_baseline`   | [creatorwyx/TAAC2026-CTR-Baseline](https://github.com/creatorwyx/TAAC2026-CTR-Baseline)                                                       |
| `deepcontextnet` | `config/gen/deepcontextnet` | [deepcontextnet](packages/deepcontextnet.md) | `deepcontextnet`     | `outputs/gen/deepcontextnet` | [suyanli220/TAAC-2026-Baseline-Tencent-Advertisement-Contest](https://github.com/suyanli220/TAAC-2026-Baseline-Tencent-Advertisement-Contest) |
| `interformer`    | `config/gen/interformer`    | [interformer](packages/interformer.md)       | `interformer`        | `outputs/gen/interformer`    | [InterFormer paper](https://arxiv.org/abs/2411.09852)                                                                                         |
| `onetrans`       | `config/gen/onetrans`       | [onetrans](packages/onetrans.md)             | `onetrans`           | `outputs/gen/onetrans`       | [OneTrans paper](https://arxiv.org/abs/2510.26104)                                                                                            |
| `hyformer`       | `config/gen/hyformer`       | [hyformer](packages/hyformer.md)             | `hyformer`           | `outputs/gen/hyformer`       | [HyFormer paper](https://arxiv.org/abs/2601.12681)                                                                                            |
| `unirec`         | `config/gen/unirec`         | [unirec](packages/unirec.md)                 | `unirec`             | `outputs/gen/unirec`         | [hojiahao/TAAC2026](https://github.com/hojiahao/TAAC2026)                                                                                     |
| `uniscaleformer` | `config/gen/uniscaleformer` | [uniscaleformer](packages/uniscaleformer.md) | `uniscaleformer`     | `outputs/gen/uniscaleformer` | [twx145/Unirec](https://github.com/twx145/Unirec)                                                                                             |
| `oo`             | `config/gen/oo`             | [oo](packages/oo.md)                         | `oo`                 | `outputs/gen/oo`             | [salmon1802/O_o](https://github.com/salmon1802/O_o)                                                                                           |

!!! info "未纳入可执行列表的目录"
    `config/gen/symbiosis` 当前只是概念说明，不是可执行实验包，因此不写进本页的正式实验清单；方案草稿见 [symbiosis](packages/symbiosis.md)。

## 当前回归基线

当前最重要的统一回归命令是：

```bash
uv run pytest tests -q
```

它覆盖目录式实验包加载、数据管线、前向构建、`train` / `evaluate` 闭环和 checkpoint 兼容性。其中 `baseline`、`grok`、`ctr_baseline`、`deepcontextnet`、`interformer`、`onetrans`、`hyformer`、`unirec`、`uniscaleformer`、`oo` 都应该纳入同一套目录式实验包回归视角。

## 当前可复核 smoke 结果

下面的指标全部来自 sample parquet 上当前工作区里可打开的训练产物，只用于说明“链路可跑”和“当前工作区里已有可复核产物”，不能拿来当作正式赛题结论。

其中：

1. `baseline` 这一行仍然是旧目录结构时期留下来的历史产物，实际对应的是现在 `grok` 这条 lineage 的 legacy baseline 记录。
2. 这次角色拆分后，新的 starter `baseline` 与重命名后的 `grok` 还没有在当前工作区里重新生成新路径下的 smoke 证据，因此本表不会伪造新的 `outputs/gen/grok` 或 `outputs/gen/baseline` 记录。
3. 其余实验包仍是统一默认配置下的 10-epoch sample smoke 结果。

| 实验包 ID        | 证据路径                                              | 最佳 epoch |    AUC | PR-AUC |  Brier | 平均时延（毫秒/样本） | P95 时延（毫秒/样本） | TFLOPs/批次 | 完整训练总 TFLOPs |                   参数量（MB） | 说明                                                |
| ---------------- | ----------------------------------------------------- | ---------: | -----: | -----: | -----: | --------------------: | --------------------: | ----------: | ----------------: | -----------------------------: | --------------------------------------------------- |
| `baseline`       | `outputs/gen/baseline_optuna/trial_0019/summary.json` |          9 | 0.7499 | 0.4086 | 0.1113 |                0.4465 |                0.7531 |    0.007692 |          3.148948 |                        67.1362 | 20-trial Optuna 最佳 trial，当前效果 / 算力折中最好 |
| `ctr_baseline`   | `outputs/smoke/ctr_baseline/summary.json`             |          8 | 0.6478 | 0.3168 | 0.1227 |                0.3493 |                0.5939 |    0.000224 |          0.091811 |                        49.0020 | 当前最轻，前向与完整训练总算力都最低                |
| `deepcontextnet` | `outputs/smoke/deepcontextnet/summary.json`           |          5 | 0.6206 | 0.1747 | 0.2126 |                0.1470 |                0.3082 |    0.001852 |         10.573857 |                        67.5103 | 推理最轻，但因 batch size 32 总算力不低             |
| `interformer`    | `outputs/smoke/interformer/summary.json`              |          5 | 0.6243 | 0.2309 | 0.1580 |               19.2306 |               19.5413 |    0.033679 |         13.356647 |                       128.3087 | 单批次前向最重，参数量也最大                        |
| `onetrans`       | `outputs/smoke/onetrans/summary.json`                 |          1 | 0.7088 | 0.2581 | 0.2054 |               10.1906 |               10.6896 |    0.016967 |          6.944763 |                        96.7710 | 高 AUC 组里算力中等偏上                             |
| `hyformer`       | `outputs/smoke/hyformer/summary.json`                 |          6 | 0.6487 | 0.3394 | 0.1410 |               19.6514 |               20.2111 |    0.017638 |          7.067170 |                        82.3516 | 与 `onetrans` 总算力接近                            |
| `unirec`         | `outputs/smoke/unirec/summary.json`                   |          6 | 0.7292 | 0.4199 | 0.2045 |               10.1691 |               10.4839 |    0.022693 |         18.975869 |                        75.5584 | 当前 AUC 最高，但完整训练总算力也最高               |
| `uniscaleformer` | `outputs/smoke/uniscaleformer/summary.json`           |          3 | 0.7125 | 0.3325 | 0.2160 |                0.8987 |                1.2936 |    0.008117 |          3.323026 |                        72.9640 | 高 AUC 组里总算力最省                               |
| `oo`             | `outputs/smoke/oo/summary.json`                       |          2 | 0.6483 | 0.3735 | 0.1869 |                0.4345 |                0.7286 |    0.012981 |          5.308285 | 仍属低延迟组，总算力也相对温和 |

其中 `TFLOPs/批次` 与参数量来自各自 `summary.json` 里的 `model_profile`，口径仍是单次验证前向 profile。`完整训练总 TFLOPs` 来自 `compute_profile.estimated_end_to_end_tflops_total`，按“单次训练步 profile × 实际 train sample 数 + 单次验证前向 profile × 实际 val sample 数 + 训练结束后的 latency probe sample 数”估算。`deepcontextnet` 的 profile batch size 为 32，其余实验默认是 64。

## 当前结论

1. 当前仓库里真正存在、可直接运行的独立实验包已经变成 `baseline`、`grok`、`ctr_baseline`、`deepcontextnet`、`interformer`、`onetrans`、`hyformer`、`unirec`、`uniscaleformer`、`oo` 十个。
2. 当前可复核的最佳历史记录仍然是 legacy baseline 路径下那条 20-trial Optuna 最佳 trial；它现在应被理解为 grok lineage 的历史最佳点，而不是新的 starter baseline 成绩。
3. 如果只看其余未搜索的默认 smoke 配置，AUC 最高的仍是 `unirec`。
4. 这组结果仍然不适合做正式赛题结论，因为数据仍然是 sample parquet，GAUC 覆盖率仍为 0，而且 baseline / grok 已经发生角色拆分，当前表里保留的是历史可复核证据而不是新角色的最新产物。

## 后续更新规则

如果后续要继续维护本页，请遵守下面几点：

1. 只记录当前分支真实存在的实验包。
2. 指标只写能直接打开文件复核的产物。
3. 产物优先写 `summary.json` 或 `evaluation.json` 的文件路径。
4. 如果某个实验还没有 smoke 结果，就写“仅 forward regression”，不要补猜测值。
