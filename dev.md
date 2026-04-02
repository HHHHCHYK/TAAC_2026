# 开发文档

## 线下测试环境
基于 uv 管理环境。依赖的唯一事实来源是 `pyproject.toml` 与 `uv.lock`，新增/删除依赖请使用 `uv add` / `uv remove`，不要再手写 `uv pip install -r requirements.txt`。
CUDA 12.8 H800 * 8

```bash
uv python install 3.14
uv sync --locked
```

## 常用命令

```bash
# 训练 baseline
uv run taac-train --config configs/baseline.yaml
uv run taac-train --config configs/grok_din_readout.yaml

# 训练其它公开方案复现版本
uv run taac-train --config configs/creatorwyx_din_adapter.yaml
uv run taac-train --config configs/tencent_sasrec_adapter.yaml
uv run taac-train --config configs/omnigenrec_adapter.yaml
uv run taac-train --config configs/deep_context_net.yaml
uv run taac-train --config configs/unirec.yaml
uv run taac-train --config configs/unirec_din_readout.yaml
uv run taac-train --config configs/uniscaleformer.yaml
uv run taac-train --config configs/interformer.yaml
uv run taac-train --config configs/onetrans.yaml
uv run taac-train --config configs/hyformer.yaml

# 注意：项目现在只维护当前这一套统一训练实现，旧版本代码产出的 best.pt 需要按当前代码重训后再评估。
# taac-evaluate 会对不兼容 checkpoint 给出明确提示，taac-batch-evaluate 会自动跳过。

# 如果需要保留独立实验目录，不要再复制一份配置文件，直接覆写输出目录。
uv run taac-train --config configs/unirec.yaml --output-dir outputs/runs/unirec_ablation_001
uv run taac-evaluate --config configs/unirec.yaml --run-dir outputs/runs/unirec_ablation_001

# 对已有 best.pt 直接回填多指标与分桶评估
uv run taac-evaluate --config configs/creatorwyx_din_adapter.yaml
uv run taac-batch-evaluate

# 基于现有 JSON 产物生成 matplotlib 可视化
uv run taac-visualize evaluation --input outputs/creatorwyx_din_adapter/evaluation.json --formats png svg
uv run taac-visualize batch-report --formats png svg
uv run taac-visualize summary --input outputs/grok_din_readout/summary.json --formats png svg
uv run taac-visualize truncation-sweep --formats png svg
uv run taac-visualize dataset-profile --formats png svg

# 分析 parquet 的 schema、特征统计与数据分布
uv run taac-feature-schema
uv run taac-feature-profile
uv run taac-feature-cluster
uv run taac-feature-cluster --outlier-fraction 0.01 --min-cluster-share 0.03 --formats png svg

# history truncation 多 seed 消融
uv run taac-truncation-sweep --config configs/grok_din_readout.yaml --seq-lens 128 256 384 --seeds 42 43 44
```

聚类分析默认会输出：

```text
outputs/feature_engineering/clustering/cluster_report.json
outputs/feature_engineering/clustering/cluster_report.md
outputs/feature_engineering/clustering/cluster_assignments.csv
outputs/feature_engineering/clustering/plots/
```

## README 展示素材

```bash
# 为 README 准备一轮独立 SVG 资产
uv run taac-visualize batch-report --formats svg --output-dir docs/visualizations/round_001/batch
uv run taac-visualize summary --input outputs/grok_din_readout/summary.json --formats svg --output-dir docs/visualizations/round_001/grok_din_readout
uv run taac-visualize truncation-sweep --formats svg --output-dir docs/visualizations/round_001/truncation
uv run taac-visualize dataset-profile --formats svg --output-dir docs/visualizations/round_001/dataset

# 为 README 准备聚类分析 SVG 资产
uv run taac-feature-cluster --outlier-fraction 0.01 --min-cluster-share 0.03 --formats svg --output-dir docs/visualizations/round_002/clustering
```

## 线下测试数据
```bash
uv run hf download TAAC2026/data_sample_1000 --cache-dir ./data --type dataset
```

## 线上运行环境
TODO

## 线上训练数据
TODO