from __future__ import annotations

from pathlib import Path
from typing import Any

from ...domain.config import SearchConfig
from ...domain.experiment import ExperimentSpec
from ...infrastructure.experiments.payload import apply_serialized_experiment, serialize_experiment
from ...infrastructure.nn.defaults import resolve_experiment_builders
from ..training.profiling import collect_experiment_model_profile, select_device
from ..training.runtime_optimization import prepare_runtime_execution
from ..training.service import run_training


def resolve_metric(summary: dict[str, Any], metric_name: str) -> float:
    current: Any = summary
    for part in metric_name.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Metric '{metric_name}' is not present in summary")
        current = current[part]
    return float(current)


def budget_status(
    model_profile: dict[str, Any],
    search_config: SearchConfig,
) -> dict[str, Any]:
    parameter_bytes = float(model_profile.get("parameter_size_mb", 0.0)) * 1024.0 * 1024.0
    model_flops_per_sample = float(model_profile.get("flops_per_sample", 0.0))
    model_tflops_per_sample = model_flops_per_sample / 1.0e12
    model_compute_profile_available = str(model_profile.get("flops_profile_status", "measured" if model_flops_per_sample > 0.0 else "unavailable")) == "measured"
    compute_budget_limit = search_config.max_model_tflops_per_sample
    parameter_budget_met = parameter_bytes <= float(search_config.max_parameter_bytes)
    model_compute_budget_met = True
    model_compute_budget_reason: str | None = None
    if compute_budget_limit is not None:
        if not model_compute_profile_available:
            model_compute_budget_met = False
            model_compute_budget_reason = "model FLOPs profile unavailable"
        else:
            model_compute_budget_met = model_tflops_per_sample <= float(compute_budget_limit)
            if not model_compute_budget_met:
                model_compute_budget_reason = "model FLOPs/sample exceeds configured limit"
    return {
        "parameter_budget_met": parameter_budget_met,
        "model_compute_budget_met": model_compute_budget_met,
        "constraints_met": parameter_budget_met and model_compute_budget_met,
        "parameter_bytes": parameter_bytes,
        "parameter_gib": parameter_bytes / float(1024**3),
        "max_parameter_bytes": int(search_config.max_parameter_bytes),
        "max_parameter_gib": float(search_config.max_parameter_bytes) / float(1024**3),
        "model_flops_per_sample": model_flops_per_sample,
        "model_tflops_per_sample": model_tflops_per_sample,
        "model_compute_profile_available": model_compute_profile_available,
        "model_compute_budget_reason": model_compute_budget_reason,
        "max_model_tflops_per_sample": None if compute_budget_limit is None else float(compute_budget_limit),
    }


def _prune_reason(phase: str, budget: dict[str, Any]) -> str:
    detail = budget.get("model_compute_budget_reason")
    if detail:
        return f"trial exceeds search budget {phase}: {detail}"
    return f"trial exceeds search budget {phase}"


def _parameter_only_model_profile(model) -> dict[str, float | int | str]:
    total_parameters = 0
    trainable_parameters = 0
    parameter_bytes = 0
    for parameter in model.parameters():
        parameter_count = int(parameter.numel())
        total_parameters += parameter_count
        if parameter.requires_grad:
            trainable_parameters += parameter_count
        parameter_bytes += parameter_count * int(parameter.element_size())
    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "parameter_size_bytes": parameter_bytes,
        "parameter_size_mb": parameter_bytes / (1024.0 * 1024.0),
        "flops_per_sample": 0.0,
        "flops_profile_status": "unavailable",
    }


def profile_trial_budget(experiment: ExperimentSpec) -> dict[str, Any]:
    device = select_device(experiment.train.device)
    experiment.refresh_feature_schema()
    compute_budget_enabled = experiment.search.max_model_tflops_per_sample is not None
    dense_dim = int(experiment.feature_schema.dense_dim if experiment.feature_schema is not None else experiment.data.dense_feature_dim)
    if compute_budget_enabled:
        builders = resolve_experiment_builders(experiment)
        _, _, data_stats = builders.build_data_pipeline(
            experiment.data,
            experiment.model,
            experiment.train,
        )
        dense_dim = int(data_stats.dense_dim)
    model = experiment.build_model_component(experiment.data, experiment.model, dense_dim)

    try:
        if compute_budget_enabled:
            model = model.to(device)
            runtime_execution = prepare_runtime_execution(model, experiment.train, device)
            model_profile = collect_experiment_model_profile(
                experiment,
                model,
                device,
                runtime_execution=runtime_execution,
                dense_dim=dense_dim,
            )
        else:
            model_profile = _parameter_only_model_profile(model)
        return {
            "model_profile": model_profile,
            "budget_status": budget_status(model_profile, experiment.search),
        }
    finally:
        del model
        if compute_budget_enabled and device.type == "cuda":
            import torch

            torch.cuda.empty_cache()


def execute_search_trial(experiment: ExperimentSpec) -> dict[str, Any]:
    budget_probe = profile_trial_budget(experiment)
    result: dict[str, Any] = {
        "status": "pruned",
        "budget_probe": budget_probe,
        "summary_path": None,
        "final_budget_status": None,
        "objective_value": None,
        "prune_reason": None,
    }
    if not budget_probe["budget_status"]["constraints_met"]:
        result["prune_reason"] = _prune_reason("before training", budget_probe["budget_status"])
        return result

    summary = run_training(experiment)
    summary_path = Path(experiment.train.output_dir) / "summary.json"
    if experiment.search.max_model_tflops_per_sample is None:
        final_budget = budget_status(summary["model_profile"], experiment.search)
    else:
        final_budget = dict(budget_probe["budget_status"])
    result["summary_path"] = str(summary_path)
    result["final_budget_status"] = final_budget

    result["status"] = "complete"
    result["objective_value"] = resolve_metric(summary, experiment.search.metric_name)
    return result


__all__ = [
    "apply_serialized_experiment",
    "budget_status",
    "execute_search_trial",
    "profile_trial_budget",
    "resolve_metric",
    "serialize_experiment",
]
