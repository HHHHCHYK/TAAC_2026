from __future__ import annotations

import json
from collections.abc import Generator

import pytest
import torch

from tests.benchmarks.transformer_backend_support import (
    DEFAULT_BACKENDS,
    DEFAULT_OUTPUT,
    DEFAULT_PROFILE_NAMES,
    DEFAULT_SCENARIOS,
    build_case,
    build_profiles,
    measure_callable,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for transformer backend benchmarks",
)

WARMUP_STEPS = 10
MEASURE_STEPS = 50
TE_PRECISION = "auto"
TE_RECIPE_MODE = "auto"


@pytest.fixture(scope="session")
def transformer_backend_records() -> Generator[list[dict[str, object]], None, None]:
    records: list[dict[str, object]] = []
    yield records
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(
        json.dumps({"records": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


@pytest.fixture(scope="session")
def transformer_backend_profiles() -> dict[str, object]:
    return build_profiles()


def _case_parameters() -> list[tuple[str, str, str]]:
    return [
        (profile_name, scenario, backend)
        for profile_name in DEFAULT_PROFILE_NAMES
        for scenario in DEFAULT_SCENARIOS
        for backend in DEFAULT_BACKENDS
    ]


@pytest.mark.parametrize(
    ("profile_name", "scenario", "backend"),
    _case_parameters(),
    ids=lambda value: str(value),
)
def test_transformer_backend_benchmarks(
    profile_name: str,
    scenario: str,
    backend: str,
    benchmark,
    benchmark_device: torch.device,
    cuda_timer,
    transformer_backend_records: list[dict[str, object]],
    transformer_backend_profiles: dict[str, object],
) -> None:
    profile = transformer_backend_profiles[profile_name]
    try:
        run = build_case(
            scenario=scenario,
            backend=backend,
            profile=profile,
            device=benchmark_device,
            te_precision=TE_PRECISION,
            te_recipe_mode=TE_RECIPE_MODE,
        )
    except Exception as exc:
        if backend == "te":
            pytest.skip(str(exc))
        raise

    benchmark.extra_info.update({
        "name": f"transformer_backend_{profile.name}_{scenario}_{backend}",
        "component": "transformer_backend",
        "phase": backend,
        "label": profile.name,
        "model": f"{profile.name}/{scenario}",
        "metric": "latency",
        "scenario": scenario,
        "backend": backend,
        "profile": profile.name,
    })
    benchmark(run)

    if benchmark_device.type == "cuda":
        stats = measure_callable(run, device=benchmark_device, warmup=WARMUP_STEPS, steps=MEASURE_STEPS)
    else:
        stats = cuda_timer(run)

    query_tokens = profile.batch_size * profile.query_length
    attention_pairs = profile.batch_size * profile.query_length * profile.context_length
    median_ms = float(stats["median_ms"])
    record = {
        "name": f"transformer_backend_{profile.name}_{scenario}_{backend}",
        "component": "transformer_backend",
        "phase": backend,
        "label": profile.name,
        "model": f"{profile.name}/{scenario}",
        "metric": "latency",
        "scenario": scenario,
        "backend": backend,
        "profile": profile_name,
        **stats,
        "throughput": float(query_tokens) / max(median_ms / 1e3, 1e-9),
        "query_tokens_per_second": float(query_tokens) / max(median_ms / 1e3, 1e-9),
        "attention_pairs_per_second": float(attention_pairs) / max(median_ms / 1e3, 1e-9),
    }

    transformer_backend_records.append(record)