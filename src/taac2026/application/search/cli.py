from __future__ import annotations

import argparse
from collections import Counter
import sys
from typing import Any

from ...infrastructure.compute.device_scheduler import parse_gpu_indices
from ...infrastructure.experiments.loader import load_experiment_package
from ...infrastructure.io.console import configure_logging, print_json, print_panel
from .service import run_search


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search a TAAC 2026 experiment with Optuna")
    parser.add_argument("--experiment", required=True, help="Experiment package path or module path")
    parser.add_argument("--study-dir", help="Directory used to store the Optuna study artifacts")
    parser.add_argument("--trials", type=int, help="Override the default Optuna trial count")
    parser.add_argument("--timeout-seconds", type=int, help="Optional wall-clock limit for the search")
    parser.add_argument("--metric-name", help="Objective metric in summary.json, e.g. best_val_auc or metrics.auc")
    parser.add_argument("--direction", choices=["maximize", "minimize"], help="Optimization direction")
    parser.add_argument("--max-parameter-gb", type=float, help="Hard cap on parameter bytes in GiB units")
    parser.add_argument(
        "--max-model-tflops-per-sample",
        type=float,
        help="Hard cap on profiled model forward compute per sample, in TFLOPs",
    )
    parser.add_argument("--seed", type=int, help="Optuna sampler seed override")
    parser.add_argument(
        "--scheduler",
        choices=["auto", "sequential"],
        default="auto",
        help="Trial execution mode; auto dispatches trials across visible GPUs based on free memory",
    )
    parser.add_argument("--gpu-indices", help="Optional comma-separated physical GPU indices for auto scheduling")
    parser.add_argument(
        "--min-free-memory-gb",
        type=float,
        default=12.0,
        help="Minimum free GPU memory required to launch one additional worker in auto mode",
    )
    parser.add_argument(
        "--max-jobs-per-gpu",
        type=int,
        default=4,
        help="Maximum concurrently launched search workers per physical GPU in auto mode",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=5.0,
        help="Polling interval for worker completion and GPU availability in auto mode",
    )
    parser.add_argument("--json", action="store_true", help="Print the full report JSON instead of the compact summary")
    return parser.parse_args(argv)


def _summarize_pruned_trials(trials: list[dict[str, Any]]) -> dict[str, int]:
    reasons: Counter[str] = Counter()
    for trial in trials:
        if trial.get("state") != "PRUNED":
            continue
        user_attrs = trial.get("user_attrs", {})
        reason = user_attrs.get("prune_reason") or user_attrs.get("trial_error") or "pruned"
        reasons[str(reason)] += 1
    return dict(reasons)


def _format_search_report(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "Search complete",
        f"experiment: {report['experiment_name']}",
        f"study dir: {report['study_dir']}",
        f"metric: {report['search']['metric_name']} ({report['search']['direction']})",
    ]

    scheduler = report.get("scheduler")
    if scheduler is not None:
        used_mode = scheduler.get("used_mode")
        fallback_reason = scheduler.get("fallback_reason")
        if used_mode == "auto":
            gpu_indices = scheduler.get("gpu_indices")
            gpu_label = ",".join(str(index) for index in gpu_indices) if gpu_indices else "all-visible"
            lines.append(
                "scheduler: "
                f"auto (gpus={gpu_label}, min_free={scheduler['min_free_memory_gb']:.1f} GiB, "
                f"max_jobs_per_gpu={scheduler['max_jobs_per_gpu']})"
            )
        elif fallback_reason:
            lines.append(f"scheduler: sequential ({fallback_reason})")

    state_counts = report.get("trial_state_counts", {})
    state_summary = ", ".join(f"{state.lower()}={count}" for state, count in state_counts.items())
    lines.append(f"trials: {report['trial_count']} ({state_summary})")

    best_trial = report.get("best_trial")
    if best_trial is None:
        lines.append("best trial: none")
    else:
        final_budget_status = best_trial.get("final_budget_status") or {}
        parameter_mib = float(final_budget_status.get("parameter_bytes", 0.0)) / float(1024**2)
        max_parameter_gib = float(final_budget_status.get("max_parameter_gib", 0.0))
        model_tflops_per_sample = float(final_budget_status.get("model_tflops_per_sample", 0.0))
        model_compute_profile_available = bool(final_budget_status.get("model_compute_profile_available", model_tflops_per_sample > 0.0))
        model_compute_budget_reason = final_budget_status.get("model_compute_budget_reason")
        model_budget_tflops = final_budget_status.get("max_model_tflops_per_sample")
        model_compute_label = (
            f"{model_tflops_per_sample:.6f} TFLOPs/sample"
            if model_compute_profile_available
            else "unavailable"
        )
        if model_budget_tflops is None:
            compute_summary = f"model_compute={model_compute_label} / uncapped"
        else:
            compute_summary = (
                f"model_compute={model_compute_label} / "
                f"{float(model_budget_tflops):.6f} TFLOPs/sample"
            )
        if model_compute_budget_reason:
            compute_summary = f"{compute_summary} ({model_compute_budget_reason})"

        lines.extend(
            [
                "",
                f"best trial: #{best_trial['number']}",
                f"value: {best_trial['value']:.6f}",
                f"trial dir: {best_trial['trial_dir']}",
                f"summary: {best_trial['summary_path']}",
                f"budget: params={parameter_mib:.2f} MiB / {max_parameter_gib:.2f} GiB, {compute_summary}",
                "params:",
            ]
        )
        for key, value in sorted(best_trial.get("params", {}).items()):
            lines.append(f"  {key} = {value}")

    prune_reason_counts = _summarize_pruned_trials(report.get("trials", []))
    if prune_reason_counts:
        lines.append("")
        lines.append("pruned:")
        for reason, count in sorted(prune_reason_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"  {count} x {reason}")

    lines.extend(
        [
            "",
            f"study summary: {report['study_summary_path']}",
            f"best experiment: {report['best_experiment_path'] or '(not available)'}",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    experiment = load_experiment_package(args.experiment)
    if args.trials is not None:
        experiment.search.n_trials = args.trials
    if args.timeout_seconds is not None:
        experiment.search.timeout_seconds = args.timeout_seconds
    if args.metric_name is not None:
        experiment.search.metric_name = args.metric_name
    if args.direction is not None:
        experiment.search.direction = args.direction
    if args.seed is not None:
        experiment.search.sampler_seed = args.seed
    if args.max_parameter_gb is not None:
        experiment.search.max_parameter_bytes = int(args.max_parameter_gb * (1024**3))
    if args.max_model_tflops_per_sample is not None:
        experiment.search.max_model_tflops_per_sample = args.max_model_tflops_per_sample

    report = run_search(
        experiment,
        experiment_path=args.experiment,
        study_dir=args.study_dir,
        show_progress=bool(sys.stderr.isatty()) and not args.json,
        scheduler=args.scheduler,
        gpu_indices=parse_gpu_indices(args.gpu_indices),
        min_free_memory_gb=args.min_free_memory_gb,
        max_jobs_per_gpu=args.max_jobs_per_gpu,
        poll_interval_seconds=args.poll_interval_seconds,
    )
    if args.json:
        print_json(report)
    else:
        print_panel("taac-search", _format_search_report(report))
    return 0


__all__ = ["_format_search_report", "main", "parse_args"]
