---
icon: lucide/search
---

# 超参数搜索

## 概述

框架集成了基于 [Optuna](https://optuna.org/) 的超参数搜索，支持自动 GPU 调度、参数量约束，以及可选的模型单样本前向计算量约束和多 trial 并发。

## 基本用法

```bash
# 对 baseline 搜索 20 个 trial
uv run taac-search --experiment config/baseline --trials 20

# 指定调度模式
uv run taac-search --experiment config/baseline --trials 20 --scheduler auto
```

## 搜索约束

| 约束项         | 默认值 | 说明                                            |
| -------------- | ------ | ----------------------------------------------- |
| 模型参数量上限 | 3 GiB  | `SearchConfig.max_parameter_bytes`              |
| 模型单样本前向计算量上限 | 未启用 | `SearchConfig.max_model_tflops_per_sample` |
| Trial 数量     | 20     | `SearchConfig.n_trials`                         |
| 超时           | 无     | `SearchConfig.timeout_seconds`                  |

如需启用模型计算量预算，可通过 CLI 传入 `--max-model-tflops-per-sample`，或在实验包里显式设置 `SearchConfig.max_model_tflops_per_sample`。搜索阶段不会复用真实验证样本，而是用固定 synthetic batch 对模型做一次单样本前向 profiling，再读取其中的 `flops_per_sample` 作为预算口径，因此不受训练 epoch、验证集规模、样本排序或 latency probe 样本量影响。

## GPU 自动调度

搜索 CLI 会自动检测当前可见 GPU 的空闲显存，按配置的最小空闲显存要求和每卡最大并发数进行 trial 派发：

- 检测可用 GPU 及其空闲显存
- 根据 `min_free_memory_gb` 和 `max_jobs_per_gpu` 分配 trial
- 如果 CUDA 不可用，自动退化为 CPU 顺序执行

## SearchConfig

```python
@dataclass(slots=True)
class SearchConfig:
    n_trials: int = 20
    timeout_seconds: int | None = None
    metric_name: str = "best_val_auc"            # 优化目标
    direction: str = "maximize"                   # 优化方向
    sampler_seed: int | None = None
    max_parameter_bytes: int = 3 * 1024**3        # 3 GiB
    max_model_tflops_per_sample: float | None = None
```

## 自定义搜索空间

实验包可以通过 `build_search_experiment` 回调定义自己的搜索空间：

```python
def build_search_experiment(
    spec: ExperimentSpec,
    trial: optuna.Trial,
) -> ExperimentSpec:
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    layers = trial.suggest_int("num_layers", 2, 6)
    return spec.derive(
        train=dataclasses.replace(spec.train, learning_rate=lr),
        model=dataclasses.replace(spec.model, num_layers=layers),
    )
```

将此函数赋值给 `EXPERIMENT.build_search_experiment` 即可。

## 搜索产物

搜索完成后，产物默认保存在实验输出目录的同级 `_optuna` 目录下：

```
outputs/config/<name>_optuna/
├── best_experiment.json   # 最优 trial 对应的实验配置
├── study_summary.json     # 搜索摘要
├── trial_<N>/             # 每个 trial 的训练产物
│   ├── best.pt
│   ├── summary.json
│   └── ...
```

## Trial 状态

每个 trial 可能处于以下状态：

- **Success**：正常完成，有有效指标
- **Pruned**：被 Optuna 剪枝（违反约束或指标不佳）
- **Failed**：运行时错误
