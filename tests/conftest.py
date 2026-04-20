from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


UNIT_TEST_FILES = {
    "test_benchmark_charts.py",
    "test_clean_pycache.py",
    "test_github_cleanup.py",
    "test_dataset_eda.py",
    "test_metrics.py",
    "test_model_performance_plot.py",
    "test_norms.py",
    "test_package_training.py",
    "test_property_based.py",
    "test_payload.py",
    "test_pooling_heads.py",
    "test_runtime_optimization.py",
    "test_schema_contract.py",
    "test_tech_timeline.py",
    "test_test_collection.py",
    "test_transformer_blocks.py",
}

GPU_TEST_FILES = {
    "bench_attention_forward.py",
    "bench_collate.py",
    "bench_e2e_train_step.py",
    "bench_embedding_lookup.py",
    "bench_ffn_forward.py",
    "bench_inference_latency.py",
    "bench_rmsnorm.py",
    "test_gpu.py",
    "test_triton_kernels.py",
}

INTEGRATION_TEST_FILES = {
    "test_data_pipeline.py",
    "test_embedding_collection.py",
    "test_evaluate_cli.py",
    "test_experiment_packages.py",
    "test_model_robustness.py",
    "test_optimizers.py",
    "test_profiling.py",
    "test_profiling_unit.py",
    "test_quantization.py",
    "test_runtime_integration.py",
    "test_search.py",
    "test_search_trial.py",
    "test_search_worker.py",
    "test_search_worker_integration.py",
    "test_torchrec_embedding.py",
    "test_training_recovery.py",
}


def _build_test_file_classification() -> dict[str, str]:
    overlapping_files = (
        (UNIT_TEST_FILES & INTEGRATION_TEST_FILES)
        | (UNIT_TEST_FILES & GPU_TEST_FILES)
        | (INTEGRATION_TEST_FILES & GPU_TEST_FILES)
    )
    if overlapping_files:
        overlap = ", ".join(sorted(overlapping_files))
        raise pytest.UsageError(
            "Test files cannot be classified into multiple phases: "
            f"{overlap}"
        )

    classification = {filename: "unit" for filename in UNIT_TEST_FILES}
    classification.update({filename: "integration" for filename in INTEGRATION_TEST_FILES})
    classification.update({filename: "gpu" for filename in GPU_TEST_FILES})
    return classification


def _requested_collection_phases(config: pytest.Config) -> set[str] | None:
    markexpr = str(getattr(getattr(config, "option", SimpleNamespace(markexpr="")), "markexpr", "") or "").strip()
    if not markexpr:
        return None
    normalized = markexpr.replace("(", " ").replace(")", " ").lower()
    tokens = [token for token in normalized.split() if token]
    if not tokens:
        return None
    allowed_tokens = {"unit", "integration", "gpu", "or"}
    if any(token not in allowed_tokens for token in tokens):
        return None
    phases = {token for token in tokens if token in {"unit", "integration", "gpu"}}
    return phases or None


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    if collection_path.suffix != ".py":
        return False

    requested_phases = _requested_collection_phases(config)
    if requested_phases is None:
        return False

    filename = collection_path.name
    classification = _build_test_file_classification()
    file_phase = classification.get(filename)
    if file_phase is None:
        return False
    return file_phase not in requested_phases


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config
    classification = _build_test_file_classification()
    unclassified_files: set[str] = set()
    for item in items:
        filename = Path(str(item.fspath)).name
        marker = classification.get(filename)
        if marker == "unit":
            item.add_marker(pytest.mark.unit)
        elif marker == "integration":
            item.add_marker(pytest.mark.integration)
        elif marker == "gpu":
            item.add_marker(pytest.mark.gpu)
        else:
            unclassified_files.add(filename)

    if unclassified_files:
        missing = ", ".join(sorted(unclassified_files))
        raise pytest.UsageError(
            "Collected test files are not classified in UNIT_TEST_FILES, "
            f"INTEGRATION_TEST_FILES, or GPU_TEST_FILES: {missing}"
        )
