# TESTING

这个文件只描述当前仓库已经实现并正在使用的测试约定，方便在改代码时快速判断该跑哪些回归。

## 测试分层

当前 CI 真正执行的分层只有两层：`unit` 和 `integration`。

- `Smoke`：默认指最小可运行闭环，通常体现在少量 `unit` / `integration` 回归里，而不是单独的 pytest marker。
- `Unit`：纯逻辑、轻依赖、单模块契约测试，例如 `metrics`、`payload`、`profiling` 边界、CLI 参数解析。
- `Fault`：失败路径和异常分支测试，当前大多归在 `unit` 或轻量 `integration`，例如坏 JSON、缺失 `EXPERIMENT`、worker 崩溃、checkpoint 不兼容。
- `Property`：基于 Hypothesis 的性质测试，当前入口在 `tests/test_property_based.py`，并计入 `unit`。
- `Integration`：训练、评估、搜索、数据管道、恢复等跨模块闭环测试。

分类落地方式见 `tests/conftest.py`。仓库目前通过文件名集合给测试动态打 `unit` / `integration` marker，而不是靠目录或每条用例手写 decorator。

## 常用命令

先同步环境：

```bash
uv sync --locked
```

跑完整回归：

```bash
uv run pytest tests -q
```

只跑快速单元测试：

```bash
uv run pytest -m unit -q
```

只跑集成测试：

```bash
uv run pytest -m integration -q
```

按 CI 口径收集 coverage：

```bash
uv run --with coverage coverage erase
uv run --with coverage coverage run -m pytest -m unit -q
uv run --with coverage coverage run --append -m pytest -m integration -q
uv run --with coverage coverage report
```

当前 coverage 统计范围与门槛由 `pyproject.toml` 控制：

- `src/taac2026/domain`
- `src/taac2026/application/search`
- `src/taac2026/application/training`
- `branch = true`
- `fail_under = 70`

## 模块改动后的最小复核

改 `src/taac2026/domain/metrics.py`：

```bash
uv run pytest tests/test_metrics.py tests/test_property_based.py -q
```

改 `src/taac2026/infrastructure/experiments/payload.py` 或 loader：

```bash
uv run pytest tests/test_payload.py -q
```

改 `src/taac2026/application/training/profiling.py`、`service.py`、`artifacts.py`、`runtime_optimization.py`：

```bash
uv run pytest tests/test_profiling_unit.py tests/test_profiling.py tests/test_runtime_optimization.py tests/test_training_recovery.py -q
```

改 `src/taac2026/application/search/trial.py`、`worker.py`、`service.py`：

```bash
uv run pytest tests/test_search_trial.py tests/test_search_worker.py tests/test_search_worker_integration.py tests/test_search.py -q
```

改数据读取、样本整理或 batch 组装：

```bash
uv run pytest tests/test_data_pipeline.py tests/test_runtime_integration.py -q
```

## Recovery / Fault / Property 当前覆盖点

- `tests/test_property_based.py`：Metrics 边界、稳定哈希、训练 CLI 运行时参数解析。
- `tests/test_payload.py`：缺 section、坏字段、缺 `EXPERIMENT`、导入失败。
- `tests/test_search_worker.py`：worker 丢结果、坏 JSON、trial 执行异常。
- `tests/test_search_worker_integration.py`：auto worker 成功 / 失败状态收敛。
- `tests/test_training_recovery.py`：训练中断后恢复、训练曲线部分产物清理、checkpoint 重跑覆盖一致性。

## 编写新测试时的约束

- 新 Python 测试命令统一写成 `uv run pytest ...`，不要写裸 `pytest`。
- 新测试文件加入后，要同步更新 `tests/conftest.py` 的文件名集合，否则 `-m unit` / `-m integration` 不会选中它。
- Property 测试默认控制样本数，优先保持稳定和快速，不把 CI 变成随机压力测试。
- 如果改动会影响训练输出物，至少检查 `best.pt`、`summary.json`、`training_curves.json`、`profiling/` 下计划文件的兼容性。
- 如果改动会影响搜索运行时，至少覆盖 success、fail、pruned 三类状态中的相关分支。