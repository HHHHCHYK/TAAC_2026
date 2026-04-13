---
title: 快速开始
icon: material/rocket-launch-outline
---

# 快速开始

这页解决五件事：把环境拉起来、把样例数据下到仓库默认路径、把 starter baseline 跑起来、看懂输出目录、知道如何切到自己的数据路径。

## 1. 环境准备

仓库基于 `uv` 管理环境，依赖事实来源是 `pyproject.toml` 与 `uv.lock`。

```bash
uv python install 3.13
uv sync --locked
```

!!! info "国内镜像与锁文件一致性"
    仓库在 `pyproject.toml` 里固定了 `uv` 的 canonical 默认索引，用来保证 `uv.lock` 在本机和 CI 间保持一致。
    如果你的机器全局把 `uv` 换到了国内镜像，直接执行 `uv sync --locked` 仍会按项目配置工作；不要再额外传 `--default-index` 或 `--index-url` 指向镜像，否则 `uv` 会认为 `uv.lock` 需要更新。
    如果只是想加速下载，优先使用系统代理、透明代理或企业缓存代理；如果你确实临时用镜像执行过 `uv lock`，提交前请运行 `uv lock --default-index https://pypi.org/simple --python 3.13` 把锁文件归一回仓库基线。

!!! info "PyTorch 轮子来源"
    Linux 环境下，`torch` 通过 `pyproject.toml` 的 uv source 固定到 PyTorch 官方 `cu128` 索引。也就是说，这里的环境同步会安装 CUDA 12.8 轮子，而不是继续跟随最新版默认索引。

## 2. 下载样例数据

内置实验包默认读取 Hugging Face 数据集 `TAAC2026/data_sample_1000` 在仓库 `data/` 缓存下的本地副本，并会自动解析当前缓存里可用的 snapshot。推荐直接用 `huggingface_hub.snapshot_download`，并把缓存根目录指到仓库下的 `data/`：

```bash
uv run --with huggingface_hub python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='TAAC2026/data_sample_1000', repo_type='dataset', cache_dir='data'))"
```

命令完成后，Hugging Face 会在下面这个缓存目录下创建 `refs/`、`snapshots/` 等结构；默认实验会从这里自动找到实际的 parquet 文件：

```text
data/datasets--TAAC2026--data_sample_1000/
```

该样例数据是公开数据集，默认不需要登录；如果这个路径已经存在，可以直接跳到下一节。

## 3. 第一次跑通

=== "训练"

    ```bash
    uv run taac-train --experiment config/gen/baseline
    ```

    可选运行时优化示例：

    ```bash
    uv run taac-train --experiment config/gen/baseline --compile --amp --amp-dtype bfloat16
    ```

=== "评估"

    ```bash
    uv run taac-evaluate single --experiment config/gen/baseline
    ```

    如果要按同一套运行时优化路径复测：

    ```bash
    uv run taac-evaluate single --experiment config/gen/baseline --compile --amp --amp-dtype bfloat16
    ```

=== "搜索"

    ```bash
    uv run taac-search --experiment config/gen/baseline --trials 20
    ```

=== "回归"

    ```bash
    uv run pytest tests -q
    ```

!!! info "baseline 和 grok 的区别"
    `config/gen/baseline` 现在是面向扩展的 starter/reference package；如果你想跑原来那条更激进的本地 unified backbone，请改用 `config/gen/grok`。

## 4. 当前仓库默认怎么找数据

当前实验包已经在各自的 `__init__.py` 中把默认数据目录指到 Hugging Face 缓存根目录；运行时会优先跟随 `refs/main`，再回退到 `snapshots/` 下可用的 parquet：

```text
data/datasets--TAAC2026--data_sample_1000/
```

当前 CLI 没有 `--dataset-path` 覆写参数。如果你要切到别的数据集，推荐做法是写一个本地 wrapper 实验包：

```python
from config.gen.oo import EXPERIMENT

EXPERIMENT = EXPERIMENT.clone()
EXPERIMENT.data.dataset_path = "/path/to/your.parquet"
EXPERIMENT.train.output_dir = "outputs/custom/oo"
```

然后把这个本地目录传给 `--experiment`。

## 5. 输出目录里会出现什么

每次训练当前会在输出目录写入四类主要产物：

| 文件                   | 作用                                                                             |
| ---------------------- | -------------------------------------------------------------------------------- |
| `best.pt`              | 保存当前最佳 epoch 的模型参数和指标。                                            |
| `summary.json`         | 保存最佳指标、latency、`model_profile`、`inference_profile`、`compute_profile`。 |
| `training_curves.json` | 保存逐 epoch 的 train loss、val loss 和 val AUC。                                |
| `training_curves.png`  | 训练过程中持续覆盖刷新的曲线图。                                                 |

如果启用了 `torch.compile` 或 AMP，`summary.json` 和评估输出里还会包含 `runtime_optimization` 字段，用来记录请求状态、实际是否生效，以及可能的降级原因。

## 6. 初次浏览建议

1. 如果你只是想跑通链路，继续看[CLI 指南](cli.md)即可。
2. 如果你要挑模型或看 smoke 记录，直接跳到[实验包与验证记录](experiments.md)。
3. 如果你要新增实验包、改测试、改图表或本地预览文档站，看[开发文档](dev.md)。
