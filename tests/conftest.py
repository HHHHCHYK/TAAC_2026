from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


TESTS_ROOT = Path(__file__).resolve().parent
PHASE_DIRECTORIES: dict[str, frozenset[str]] = {
    "unit": frozenset({"unit"}),
    "integration": frozenset({"integration"}),
    "gpu": frozenset({"gpu"}),
}
BENCHMARK_PHASE_DIRECTORIES: dict[str, tuple[str, str]] = {
    "benchmark_cpu": ("benchmarks", "cpu"),
    "benchmark_gpu": ("benchmarks", "gpu"),
}


def cpu_ci_safe_benchmark_files() -> tuple[str, ...]:
    benchmark_root = TESTS_ROOT / "benchmarks" / "cpu"
    safe_files: list[str] = []

    for candidate in sorted(benchmark_root.glob("bench_*.py")):
        candidate_text = candidate.read_text(encoding="utf-8")
        if "require_torchrec_runtime" in candidate_text:
            continue
        safe_files.append(_render_test_path(candidate))

    return tuple(safe_files)


def _relative_test_parts(collection_path: Path) -> tuple[str, ...] | None:
    parts = collection_path.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "tests":
            return parts[index + 1 :]
    return None


def _is_test_module_path(collection_path: Path) -> bool:
    return collection_path.suffix == ".py" and collection_path.name.startswith(("test_", "bench_"))


def _classify_test_path(collection_path: Path) -> str | None:
    if not _is_test_module_path(collection_path):
        return None

    relative_parts = _relative_test_parts(collection_path)
    if not relative_parts:
        return None

    if relative_parts[0] == "benchmarks":
        if len(relative_parts) < 2:
            return None
        benchmark_directory = (relative_parts[0], relative_parts[1])
        for phase, phase_directory in BENCHMARK_PHASE_DIRECTORIES.items():
            if benchmark_directory == phase_directory:
                return phase
        return None

    phase_directory = relative_parts[0]
    for phase, directory_names in PHASE_DIRECTORIES.items():
        if phase_directory in directory_names:
            return phase
    return None


def _render_test_path(collection_path: Path) -> str:
    relative_parts = _relative_test_parts(collection_path)
    if not relative_parts:
        return collection_path.as_posix()
    return Path("tests", *relative_parts).as_posix()


def count_classified_test_files() -> dict[str, int]:
    counts = {
        **{phase: 0 for phase in PHASE_DIRECTORIES},
        **{phase: 0 for phase in BENCHMARK_PHASE_DIRECTORIES},
    }
    unclassified_files: list[str] = []

    for candidate in TESTS_ROOT.rglob("*.py"):
        if not _is_test_module_path(candidate):
            continue
        phase = _classify_test_path(candidate)
        if phase is None:
            unclassified_files.append(_render_test_path(candidate))
            continue
        counts[phase] += 1

    if unclassified_files:
        missing = ", ".join(sorted(unclassified_files))
        raise pytest.UsageError(
            "Collected test files must live under tests/unit, tests/integration, "
            f"tests/gpu, tests/benchmarks/cpu, or tests/benchmarks/gpu: {missing}"
        )

    return counts


def _requested_collection_phases(config: pytest.Config) -> set[str] | None:
    markexpr = str(getattr(getattr(config, "option", SimpleNamespace(markexpr="")), "markexpr", "") or "").strip()
    if not markexpr:
        return None
    normalized = markexpr.replace("(", " ").replace(")", " ").lower()
    tokens = [token for token in normalized.split() if token]
    if not tokens:
        return None
    allowed_tokens = {"unit", "integration", "gpu", "benchmark_cpu", "benchmark_gpu", "or"}
    if any(token not in allowed_tokens for token in tokens):
        return None
    phases = {token for token in tokens if token in {"unit", "integration", "gpu", "benchmark_cpu", "benchmark_gpu"}}
    return phases or None


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    if collection_path.suffix != ".py":
        return False

    requested_phases = _requested_collection_phases(config)
    if requested_phases is None:
        return False

    file_phase = _classify_test_path(collection_path)
    if file_phase is None:
        return False
    return file_phase not in requested_phases


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config
    unclassified_files: set[str] = set()
    for item in items:
        marker = _classify_test_path(Path(str(item.fspath)))
        if marker == "unit":
            item.add_marker(pytest.mark.unit)
        elif marker == "integration":
            item.add_marker(pytest.mark.integration)
        elif marker == "gpu":
            item.add_marker(pytest.mark.gpu)
        elif marker == "benchmark_cpu":
            item.add_marker(pytest.mark.benchmark_cpu)
        elif marker == "benchmark_gpu":
            item.add_marker(pytest.mark.benchmark_gpu)
        else:
            unclassified_files.add(_render_test_path(Path(str(item.fspath))))

    if unclassified_files:
        missing = ", ".join(sorted(unclassified_files))
        raise pytest.UsageError(
            "Collected test files must live under tests/unit, tests/integration, "
            f"tests/gpu, tests/benchmarks/cpu, or tests/benchmarks/gpu: {missing}"
        )
