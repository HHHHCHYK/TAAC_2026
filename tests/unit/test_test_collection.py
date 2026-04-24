from __future__ import annotations

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from tests import conftest as tests_conftest


class _FakeItem:
    def __init__(self, relative_path: str) -> None:
        self.fspath = Path(relative_path)
        self.markers: list[str] = []

    def add_marker(self, marker) -> None:
        self.markers.append(marker.name)


def test_collection_marks_files_by_phase_directory() -> None:
    unit_item = _FakeItem("tests/unit/test_unit.py")
    integration_item = _FakeItem("tests/integration/test_integration.py")
    benchmark_cpu_item = _FakeItem("tests/benchmarks/cpu/bench_latency.py")
    benchmark_gpu_item = _FakeItem("tests/benchmarks/gpu/bench_latency.py")

    tests_conftest.pytest_collection_modifyitems(
        object(),
        [unit_item, integration_item, benchmark_cpu_item, benchmark_gpu_item],
    )

    assert unit_item.markers == ["unit"]
    assert integration_item.markers == ["integration"]
    assert benchmark_cpu_item.markers == ["benchmark_cpu"]
    assert benchmark_gpu_item.markers == ["benchmark_gpu"]


def test_classify_test_path_maps_supported_directories() -> None:
    assert tests_conftest._classify_test_path(Path("tests/unit/test_unit.py")) == "unit"
    assert tests_conftest._classify_test_path(Path("tests/integration/test_integration.py")) == "integration"
    assert tests_conftest._classify_test_path(Path("tests/gpu/test_gpu.py")) == "gpu"
    assert tests_conftest._classify_test_path(Path("tests/benchmarks/cpu/bench_latency.py")) == "benchmark_cpu"
    assert tests_conftest._classify_test_path(Path("tests/benchmarks/gpu/bench_latency.py")) == "benchmark_gpu"


def test_collection_rejects_unclassified_files() -> None:
    with pytest.raises(pytest.UsageError, match=r"tests/test_unknown\.py"):
        tests_conftest.pytest_collection_modifyitems(object(), [_FakeItem("tests/test_unknown.py")])


class _FakeConfig:
    def __init__(self, markexpr: str) -> None:
        self.option = SimpleNamespace(markexpr=markexpr)


def test_requested_collection_phases_parses_simple_marker_expressions() -> None:
    assert tests_conftest._requested_collection_phases(_FakeConfig("unit")) == {"unit"}
    assert tests_conftest._requested_collection_phases(_FakeConfig("integration or gpu")) == {"integration", "gpu"}
    assert tests_conftest._requested_collection_phases(_FakeConfig("unit or benchmark_cpu")) == {"unit", "benchmark_cpu"}
    assert tests_conftest._requested_collection_phases(_FakeConfig("unit and integration")) is None


def test_ignore_collect_skips_files_outside_requested_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _FakeConfig("unit")

    assert tests_conftest.pytest_ignore_collect(Path("tests/unit/test_unit.py"), config) is False
    assert tests_conftest.pytest_ignore_collect(Path("tests/integration/test_integration.py"), config) is True
    assert tests_conftest.pytest_ignore_collect(Path("tests/gpu/test_gpu.py"), config) is True
    assert tests_conftest.pytest_ignore_collect(Path("tests/benchmarks/cpu/bench_latency.py"), config) is True


def test_pytest_config_collects_benchmark_modules_by_default() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    config = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert config["tool"]["pytest"]["ini_options"]["python_files"] == ["test_*.py", "bench_*.py"]
    assert config["tool"]["pytest"]["ini_options"]["markers"][-2:] == [
        "benchmark_cpu: CPU-safe benchmark tests that run in automatic CI",
        "benchmark_gpu: GPU benchmark tests that run only via local CLI entry points",
    ]


def test_benchmark_module_import_does_not_build_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    import tests.benchmarks.transformer_backend_support as transformer_backend_support

    module_name = "tests.benchmarks.gpu.bench_transformer_backends"
    existing_module = sys.modules.pop(module_name, None)
    monkeypatch.setattr(
        transformer_backend_support,
        "build_profiles",
        lambda: (_ for _ in ()).throw(AssertionError("build_profiles should not run at module import time")),
    )

    try:
        imported_module = importlib.import_module(module_name)
        assert imported_module is not None
    finally:
        sys.modules.pop(module_name, None)
        if existing_module is not None:
            sys.modules[module_name] = existing_module


def test_shared_benchmark_conftest_does_not_import_workspace_support() -> None:
    module_name = "tests.benchmarks.conftest"
    existing_module = sys.modules.pop(module_name, None)
    existing_support = sys.modules.pop("tests.support", None)

    try:
        imported_module = importlib.import_module(module_name)
        assert imported_module is not None
        assert "tests.support" not in sys.modules
    finally:
        sys.modules.pop(module_name, None)
        if existing_module is not None:
            sys.modules[module_name] = existing_module
        if existing_support is not None:
            sys.modules["tests.support"] = existing_support


def test_count_classified_test_files_counts_phase_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tests_root = tmp_path / "tests"
    (tests_root / "unit").mkdir(parents=True)
    (tests_root / "integration").mkdir()
    (tests_root / "gpu").mkdir()
    (tests_root / "benchmarks" / "cpu").mkdir(parents=True)
    (tests_root / "benchmarks" / "gpu").mkdir()
    (tests_root / "unit" / "test_unit.py").write_text("", encoding="utf-8")
    (tests_root / "integration" / "test_integration.py").write_text("", encoding="utf-8")
    (tests_root / "gpu" / "test_gpu.py").write_text("", encoding="utf-8")
    (tests_root / "benchmarks" / "cpu" / "bench_latency.py").write_text("", encoding="utf-8")
    (tests_root / "benchmarks" / "gpu" / "bench_latency.py").write_text("", encoding="utf-8")

    monkeypatch.setattr(tests_conftest, "TESTS_ROOT", tests_root)

    assert tests_conftest.count_classified_test_files() == {
        "unit": 1,
        "integration": 1,
        "gpu": 1,
        "benchmark_cpu": 1,
        "benchmark_gpu": 1,
    }


def test_count_classified_test_files_rejects_unclassified_modules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tests_root = tmp_path / "tests"
    tests_root.mkdir(parents=True)
    (tests_root / "test_unknown.py").write_text("", encoding="utf-8")

    monkeypatch.setattr(tests_conftest, "TESTS_ROOT", tests_root)

    with pytest.raises(pytest.UsageError, match=r"tests/test_unknown\.py"):
        tests_conftest.count_classified_test_files()


def test_cpu_ci_safe_benchmark_files_excludes_torchrec_runtime_gated_modules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tests_root = tmp_path / "tests"
    benchmark_root = tests_root / "benchmarks" / "cpu"
    benchmark_root.mkdir(parents=True)
    (benchmark_root / "bench_safe.py").write_text("def test_safe():\n    pass\n", encoding="utf-8")
    (benchmark_root / "bench_runtime.py").write_text(
        "def test_runtime(require_torchrec_runtime):\n    del require_torchrec_runtime\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(tests_conftest, "TESTS_ROOT", tests_root)

    assert tests_conftest.cpu_ci_safe_benchmark_files() == ("tests/benchmarks/cpu/bench_safe.py",)