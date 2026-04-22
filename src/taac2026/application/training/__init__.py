from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "build_evaluation_external_profiler_plan": (".external_profilers", "build_evaluation_external_profiler_plan"),
    "build_profiling_report": (".profiling", "build_profiling_report"),
    "build_training_external_profiler_plan": (".external_profilers", "build_training_external_profiler_plan"),
    "collect_compute_profile": (".profiling", "collect_compute_profile"),
    "collect_external_profiler_plan": (".external_profilers", "collect_external_profiler_plan"),
    "collect_inference_profile": (".profiling", "collect_inference_profile"),
    "collect_loader_outputs": (".profiling", "collect_loader_outputs"),
    "collect_synthetic_model_profile": (".profiling", "collect_synthetic_model_profile"),
    "main": (".cli", "main"),
    "measure_latency": (".profiling", "measure_latency"),
    "parse_train_args": (".cli", "parse_train_args"),
    "run_training": (".service", "run_training"),
    "select_device": (".profiling", "select_device"),
    "set_random_seed": (".profiling", "set_random_seed"),
    "write_external_profiler_plan_artifacts": (".external_profilers", "write_external_profiler_plan_artifacts"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

__all__ = [
    "build_evaluation_external_profiler_plan",
    "build_profiling_report",
    "build_training_external_profiler_plan",
    "collect_compute_profile",
    "collect_external_profiler_plan",
    "collect_inference_profile",
    "collect_loader_outputs",
    "collect_synthetic_model_profile",
    "main",
    "measure_latency",
    "parse_train_args",
    "run_training",
    "select_device",
    "set_random_seed",
    "write_external_profiler_plan_artifacts",
]
