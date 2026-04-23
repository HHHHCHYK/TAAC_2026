# Testing

## 概览

测试套件位于 `tests/`，使用 [pytest](https://docs.pytest.org/) 运行。测试阶段由目录决定：`tests/unit/` 自动标记为 **unit**，`tests/integration/` 自动标记为 **integration**，`tests/gpu/` 自动标记为 **gpu**，`tests/benchmarks/cpu/` 自动标记为 **benchmark_cpu**，`tests/benchmarks/gpu/` 自动标记为 **benchmark_gpu**。这样自动 CI、手动 GPU 测试和手动性能基准各自有稳定入口，不再靠一个总目录绑定到同一个 marker。

快速 CI（`.github/workflows/ci.yml`）负责 CPU unit、CPU benchmark 与 CPU-safe coverage 门槛；GPU 测试与 GPU benchmark 则分别拆到手动触发的 `.github/workflows/gpu-tests.yml` 和 `.github/workflows/performance-benchmarks.yml`。这样做是因为仓库当前固定的是 CUDA 版 TorchRec 和 fbgemm-gpu，部分看似 CPU 断言的测试在导入阶段也会要求 `libcuda.so.1`，而文档部署不应被等待自托管 runner 的队列阻塞。

## 快速开始

```bash
# 同步环境（锁定版本）
uv sync --locked

# 全量回归
uv run pytest -q

# 仅单元测试
uv run pytest -m unit -q

# 仅集成测试（需要 CUDA/TorchRec 运行栈，手动 GPU workflow 执行）
uv run pytest -m integration -q

# GPU 测试（需要 CUDA 硬件）
uv run pytest tests/gpu/test_gpu_environment.py tests/gpu -q

# 自动 CI 上的 CPU benchmark
uv run pytest tests/benchmarks/cpu -m benchmark_cpu -v

# 手动 GPU benchmark
uv run pytest tests/benchmarks/gpu -m benchmark_gpu -v

# 共享 Transformer backend benchmark（torch / triton / te）
uv run pytest tests/benchmarks/gpu/bench_transformer_backends.py -v

# 通用代码风格检查
uv run --with ruff ruff check .
```

## 测试分层

| 标记            | 说明                                      | 目录 / 示例                                                                                          |
| --------------- | ----------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `unit`          | 纯逻辑、CPU 可直接运行、快速              | `tests/unit/`，如 `test_metrics.py`、`test_property_based.py`                                        |
| `integration`   | 跨模块闭环，依赖 TorchRec/CUDA 运行栈     | `tests/integration/`，如 `test_runtime_integration.py`、`test_search_worker.py`                      |
| `gpu`           | 需要真实 CUDA 设备或 Triton 内核的功能测试 | `tests/gpu/`，如 `test_triton_kernels.py`、`test_gpu.py`                                             |
| `benchmark_cpu` | CPU-only benchmark，自动在 CI 中运行      | `tests/benchmarks/cpu/`，如 `bench_ffn_forward.py`、`bench_inference_latency.py`                     |
| `benchmark_gpu` | GPU-only benchmark，仅在手动 workflow 运行 | `tests/benchmarks/gpu/`，如 `bench_attention_forward.py`、`bench_transformer_backends.py`            |

新增测试文件时，把它放进对应阶段目录即可；如果测试模块仍留在 `tests/` 根目录，pytest 收集阶段会直接报错，避免出现未分类测试悄悄混入默认路由。

## Style 与 Policy

通用代码风格检查走 Ruff，而不是新增一个 `tests/format/` 相位目录：

```bash
uv run --with ruff ruff check .
```

仓库特定的策略规则仍然保留在 pytest 单元测试里，目前入口是：

```bash
uv run pytest tests/unit/test_repo_policy.py -q
```

这样可以把“通用 style 工具”和“仓库特定 policy 断言”分开治理，同时不破坏现有 `tests/unit`、`tests/integration`、`tests/gpu`、`tests/benchmarks/cpu`、`tests/benchmarks/gpu` 的阶段语义。

## Coverage

```bash
uv run --with coverage coverage erase
uv run --with coverage coverage run -m pytest -m unit -q
uv run --with coverage coverage run --append -m pytest -m integration -q
uv run --with coverage coverage report
```

| 项目     | 值                                                                                            |
| -------- | --------------------------------------------------------------------------------------------- |
| 统计范围 | `src/taac2026/domain`、`src/taac2026/application/search`、`src/taac2026/application/training` |
| 分支覆盖 | 开启                                                                                          |
| 最低门槛 | **70 %**                                                                                      |

配置位于 `pyproject.toml` 的 `[tool.coverage.*]` 段。

快速 CI 不会直接拿 unit-only 数据去校验这组全量范围，而是单独对 CPU-safe 子集执行门槛：`src/taac2026/domain/*`、`src/taac2026/application/training/__init__.py`、`src/taac2026/application/training/cli.py`、`src/taac2026/application/training/runtime_optimization.py`。如果你要看完整仓库覆盖率，仍然要合并上面的 unit 与 integration/gpu 数据再执行 `uv run --with coverage coverage report`。

## 模块改动速查

改完代码后，按下表选择最小验证集，快速确认不回退：

| 改动范围                             | 建议运行                                                                                                                                                                    |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `domain/metrics.py`                  | `tests/unit/test_metrics.py` `tests/unit/test_property_based.py`                                                                                                            |
| `infrastructure/experiments/payload` | `tests/unit/test_payload.py`                                                                                                                                                |
| `application/training/`              | `tests/unit/test_runtime_optimization.py` `tests/integration/test_profiling_unit.py` `tests/integration/test_profiling.py` `tests/integration/test_training_recovery.py`    |
| `application/search/`                | `tests/integration/test_search_trial.py` `tests/integration/test_search_worker.py` `tests/integration/test_search_worker_integration.py` `tests/integration/test_search.py` |
| 数据读取 / batch 组装                | `tests/integration/test_data_pipeline.py` `tests/integration/test_runtime_integration.py`                                                                                   |
| 实验包 (`config/`)                   | `tests/integration/test_experiment_packages.py` `tests/integration/test_model_robustness.py`                                                                                |
| 共享 Transformer / TE backend        | `tests/unit/test_transformer_blocks.py` `uv run pytest tests/benchmarks/gpu/bench_transformer_backends.py -v`                                                               |

示例：

```bash
uv run pytest tests/unit/test_metrics.py tests/unit/test_property_based.py -q
```

## 编写新测试

1. 在 `tests/unit/`、`tests/integration/`、`tests/gpu/`、`tests/benchmarks/cpu/` 或 `tests/benchmarks/gpu/` 中新建对应的 `test_*.py` 或 `bench_*.py`。
2. 统一使用 `uv run pytest ...` 运行，不要直接调 `python -m pytest`。
3. Property-based 测试用 Hypothesis，控制 `max_examples` 保持速度。
4. 涉及训练产出物时，验证 `best.pt`、`summary.json`、`training_curves.json`、`profiling/` 的兼容性。
5. 涉及搜索流程时，覆盖 success、fail、pruned 三种 trial 状态。

## CI 流程

快速 CI 在 `ubuntu-latest` + Python 3.13 上运行 CPU 纯逻辑测试、CPU benchmark，并只对 CPU-safe 子集执行 coverage 门槛。GPU 测试与 GPU benchmark 改为独立的手动 workflow，在需要时再投递到自托管 GPU runner。文档部署只等待快速 CI 完成。

快速 CI：

1. `uv sync --locked` — 严格锁定环境
2. `uv run --with ruff ruff check .` — 通用代码风格检查
3. `uv run pytest tests/unit/test_repo_policy.py -q` — 仓库策略规则检查
4. `coverage run --data-file=.coverage.cpu -m pytest -m unit` — CPU 单元测试 + 覆盖率采集
5. `uv run pytest tests/benchmarks/cpu -m benchmark_cpu --benchmark-json=benchmark-result-cpu.json -v` — 自动执行 CPU benchmark 并上传结果 artifact
6. `coverage report --fail-under=70 --include="src/taac2026/domain/*,src/taac2026/application/training/__init__.py,src/taac2026/application/training/cli.py,src/taac2026/application/training/runtime_optimization.py"` — CPU-safe 子集门槛校验（< 70 % 失败）
7. `coverage xml --fail-under=0 --include="src/taac2026/domain/*,src/taac2026/application/training/__init__.py,src/taac2026/application/training/cli.py,src/taac2026/application/training/runtime_optimization.py" -o coverage.xml` — 导出 CPU-only coverage artifact

手动 GPU 测试：

1. `uv sync --locked` — 在自托管 GPU runner 上同步环境
2. `TAAC_GPU_ENV_REPORT_PATH=gpu-env-report.json uv run pytest tests/gpu/test_gpu_environment.py -q` — 记录 GPU compute capability、精度路由、TorchRec / fbgemm / Triton 以及可选 Transformer Engine 工具链证据
3. `coverage run --data-file=.coverage.gpu -m pytest -m "integration or gpu"` — 执行 integration + gpu 标记测试
4. 上传 `.coverage.gpu` artifact，供需要时与 `.coverage.cpu` 合并查看完整 coverage

手动性能测试：

1. `uv sync --locked` — 在自托管 GPU runner 上同步环境
2. `uv run pytest tests/benchmarks/gpu -m benchmark_gpu --benchmark-json=benchmark-result.json -v` — 只执行 GPU benchmark 并产出 benchmark 原始结果
3. `uv run taac-bench-report --input benchmark-result.json` — 生成 benchmark 图表缓存供文档站点复用

如果只在本地复现实验，可直接运行 `uv run pytest tests/benchmarks/cpu -m benchmark_cpu -v`。如果是在本地 CUDA 机器上复现 GPU benchmark，则运行 `uv run pytest tests/benchmarks/gpu -m benchmark_gpu -v`。
