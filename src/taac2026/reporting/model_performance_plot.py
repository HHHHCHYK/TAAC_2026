from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


DISPLAY_NAMES = {
    "baseline": "baseline",
    "ctr_baseline": "ctr_baseline",
    "deepcontextnet": "deepcontextnet",
    "interformer": "interformer",
    "onetrans": "onetrans",
    "hyformer": "hyformer",
    "unirec": "unirec",
    "uniscaleformer": "uniscaleformer",
    "oo": "o_o",
}

DOC_SLUGS = {
    "Baseline": "baseline",
    "CTR Baseline": "ctr_baseline",
    "DeepContextNet": "deepcontextnet",
    "InterFormer": "interformer",
    "OneTrans": "onetrans",
    "HyFormer": "hyformer",
    "UniRec": "unirec",
    "UniScaleFormer": "uniscaleformer",
    "O_o": "oo",
}

LABEL_OFFSETS = {
    "baseline": (10, 8),
    "ctr_baseline": (10, -18),
    "deepcontextnet": (10, -18),
    "interformer": (10, -8),
    "onetrans": (10, 8),
    "hyformer": (10, -18),
    "unirec": (10, 10),
    "uniscaleformer": (10, -18),
    "oo": (10, -18),
}

SEARCH_COLORS = ["#f59e0b", "#10b981", "#ef4444", "#a855f7", "#22d3ee", "#f97316"]
XMetric = Literal["size", "compute"]
PointXGetter = Callable[["ModelPoint"], float]


@dataclass(slots=True)
class ModelPoint:
    slug: str
    label: str
    auc: float
    params_million: float
    parameter_size_mb: float
    total_tflops: float
    source: str
    evidence_path: str | None = None


@dataclass(slots=True)
class SearchSeries:
    slug: str
    label: str
    study_dir: Path
    points: list[ModelPoint]


def _point_from_payload(slug: str, label: str, payload: dict, *, source: str, evidence_path: str | None) -> ModelPoint:
    model_profile = payload["model_profile"]
    compute_profile = payload.get("compute_profile", {})
    metrics = payload["metrics"]
    total_parameters = float(model_profile.get("total_parameters", 0.0))
    parameter_size_mb = float(model_profile["parameter_size_mb"])
    params_million = total_parameters / 1.0e6 if total_parameters > 0 else (parameter_size_mb * 1024.0 * 1024.0 / 4.0e6)
    return ModelPoint(
        slug=slug,
        label=label,
        auc=float(metrics["auc"]),
        params_million=params_million,
        parameter_size_mb=parameter_size_mb,
        total_tflops=float(compute_profile.get("estimated_end_to_end_tflops_total", 0.0)),
        source=source,
        evidence_path=evidence_path,
    )


def _parse_experiments_doc(doc_path: Path) -> dict[str, ModelPoint]:
    if not doc_path.exists():
        return {}

    lines = doc_path.read_text(encoding="utf-8").splitlines()
    table_lines: list[str] = []
    for line in lines:
        if line.startswith("|"):
            table_lines.append(line)
            continue
        if table_lines:
            points = _parse_experiments_doc_table(table_lines)
            if points:
                return points
            table_lines = []

    if table_lines:
        return _parse_experiments_doc_table(table_lines)
    return {}


def _parse_experiments_doc_table(table_lines: list[str]) -> dict[str, ModelPoint]:
    if len(table_lines) < 3:
        return {}

    headers = [cell.strip() for cell in table_lines[0].strip().strip("|").split("|")]
    try:
        experiment_index = headers.index("实验包")
        auc_index = headers.index("AUC")
        tflops_index = headers.index("TFLOPs")
        size_index = headers.index("模型大小(MB)")
    except ValueError:
        return {}

    points: dict[str, ModelPoint] = {}
    for row in table_lines[2:]:
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        required_index = max(experiment_index, auc_index, tflops_index, size_index)
        if len(cells) <= required_index:
            continue
        experiment_name = cells[experiment_index]
        slug = DOC_SLUGS.get(experiment_name)
        if slug is None:
            continue
        try:
            parameter_size_mb = float(cells[size_index])
            auc = float(cells[auc_index])
            total_tflops = float(cells[tflops_index])
        except ValueError:
            continue
        points[slug] = ModelPoint(
            slug=slug,
            label=DISPLAY_NAMES[slug],
            auc=auc,
            params_million=parameter_size_mb * 1024.0 * 1024.0 / 4.0e6,
            parameter_size_mb=parameter_size_mb,
            total_tflops=total_tflops,
            source="docs",
        )
    return points


def load_base_points(summary_root: Path, experiments_doc_path: Path) -> list[ModelPoint]:
    doc_points = _parse_experiments_doc(experiments_doc_path)
    points: list[ModelPoint] = []
    missing: list[str] = []
    for slug, label in DISPLAY_NAMES.items():
        summary_path = summary_root / slug / "summary.json"
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            points.append(
                _point_from_payload(
                    slug,
                    label,
                    payload,
                    source="summary",
                    evidence_path=str(summary_path),
                )
            )
            continue
        if slug in doc_points:
            points.append(doc_points[slug])
            continue
        missing.append(str(summary_path))
    if points:
        return points
    if missing:
        missing_block = "\n".join(missing)
        raise FileNotFoundError(
            "Missing summary files and no markdown performance snapshot was available:\n"
            f"{missing_block}"
        )
    raise FileNotFoundError("No summary.json artifacts or markdown performance snapshot were available")


def load_search_series(search_root: Path) -> dict[str, SearchSeries]:
    if not search_root.exists():
        return {}

    search_series: dict[str, SearchSeries] = {}
    for study_dir in sorted(search_root.glob("*_optuna")):
        slug = study_dir.name.removesuffix("_optuna")
        if slug not in DISPLAY_NAMES:
            continue

        points: list[ModelPoint] = []
        for summary_path in sorted(study_dir.glob("trial_*/summary.json")):
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            points.append(
                _point_from_payload(
                    slug,
                    DISPLAY_NAMES[slug],
                    payload,
                    source="search",
                    evidence_path=str(summary_path),
                )
            )
        if not points:
            continue
        search_series[slug] = SearchSeries(
            slug=slug,
            label=DISPLAY_NAMES[slug],
            study_dir=study_dir,
            points=points,
        )
    return search_series


def merge_best_search_points(base_points: list[ModelPoint], search_series: dict[str, SearchSeries]) -> list[ModelPoint]:
    merged = {point.slug: point for point in base_points}
    for slug, series in search_series.items():
        merged[slug] = max(series.points, key=lambda point: point.auc)
    return [merged[slug] for slug in DISPLAY_NAMES if slug in merged]


def pareto_frontier(points: list[ModelPoint], x_getter: PointXGetter) -> list[ModelPoint]:
    frontier: list[ModelPoint] = []
    best_auc = float("-inf")
    for point in sorted(points, key=x_getter):
        if point.auc > best_auc:
            frontier.append(point)
            best_auc = point.auc
    return frontier


def metric_config(x_metric: XMetric) -> dict[str, object]:
    if x_metric == "compute":
        return {
            "title": "Model Performance VS Compute",
            "xlabel": "Estimated End-to-End Training Compute (TFLOPs)",
            "subtitle": "sample parquet, baseline point overridden by optuna best when available",
            "x_getter": lambda point: point.total_tflops,
            "x_formatter": FuncFormatter(lambda value, _pos: f"{value:g}"),
            "x_scale": "log",
            "x_ticks": [0.1, 0.2, 0.5, 1, 2, 5, 10, 20],
            "highlight_note": "blue frontier = best model tradeoff; colored dots/curve = optuna search trials and quadratic fit",
        }
    return {
        "title": "Model Performance VS Size",
        "xlabel": "Total Model Size (Million Parameters)",
        "subtitle": "sample parquet, baseline point overridden by optuna best when available",
        "x_getter": lambda point: point.params_million,
        "x_formatter": FuncFormatter(lambda value, _pos: f"{value:.0f}"),
        "x_scale": "linear",
        "x_ticks": None,
        "highlight_note": "blue frontier = best model tradeoff; colored dots/curve = optuna search trials and quadratic fit",
    }


def fit_curve(points: list[ModelPoint], x_getter: PointXGetter) -> tuple[np.ndarray, np.ndarray] | None:
    x_values = np.array([float(x_getter(point)) for point in points], dtype=float)
    y_values = np.array([float(point.auc) for point in points], dtype=float)
    valid_mask = x_values > 0
    x_values = x_values[valid_mask]
    y_values = y_values[valid_mask]
    if x_values.size < 2:
        return None

    log_x = np.log10(x_values)
    unique_x = np.unique(np.round(log_x, 10))
    if unique_x.size < 2:
        return None

    degree = 2 if unique_x.size >= 3 else 1
    coefficients = np.polyfit(log_x, y_values, degree)
    dense_log_x = np.linspace(log_x.min(), log_x.max(), 200)
    dense_y = np.polyval(coefficients, dense_log_x)
    dense_x = np.power(10.0, dense_log_x)
    return dense_x, dense_y


def render(points: list[ModelPoint], search_series: dict[str, SearchSeries], output_path: Path, x_metric: XMetric) -> None:
    plt.style.use("dark_background")
    figure, axis = plt.subplots(figsize=(13, 9), facecolor="#111418")
    axis.set_facecolor("#111418")
    config = metric_config(x_metric)
    x_getter = config["x_getter"]

    frontier = pareto_frontier(points, x_getter)
    frontier_slugs = {point.slug for point in frontier}
    frontier = sorted(frontier, key=x_getter)

    other_points = [point for point in points if point.slug not in frontier_slugs]
    if other_points:
        axis.scatter(
            [x_getter(point) for point in other_points],
            [point.auc for point in other_points],
            s=110,
            color="#c7d0dc",
            edgecolors="none",
            alpha=0.95,
            zorder=4,
        )

    for color_index, slug in enumerate(sorted(search_series)):
        series = search_series[slug]
        color = SEARCH_COLORS[color_index % len(SEARCH_COLORS)]
        axis.scatter(
            [x_getter(point) for point in series.points],
            [point.auc for point in series.points],
            s=42,
            color=color,
            edgecolors="none",
            alpha=0.34,
            zorder=2,
        )
        fitted = fit_curve(series.points, x_getter)
        if fitted is not None:
            fitted_x, fitted_y = fitted
            axis.plot(
                fitted_x,
                fitted_y,
                color=color,
                linewidth=2.0,
                linestyle=(0, (5, 3)),
                alpha=0.9,
                zorder=3,
            )

    axis.scatter(
        [x_getter(point) for point in frontier],
        [point.auc for point in frontier],
        s=130,
        color="#3b82f6",
        edgecolors="#93c5fd",
        linewidths=1.5,
        alpha=1.0,
        zorder=5,
    )

    frontier_x = [x_getter(point) for point in frontier]
    frontier_y = [point.auc for point in frontier]
    fill_floor = min(point.auc for point in points) - 0.012
    axis.fill_between(frontier_x, frontier_y, [fill_floor] * len(frontier_x), color="#2563eb", alpha=0.22, zorder=1)
    axis.plot(frontier_x, frontier_y, color="#3b82f6", linewidth=2.5, alpha=0.95, zorder=3)

    for point in sorted(points, key=x_getter):
        dx, dy = LABEL_OFFSETS.get(point.slug, (10, 8))
        color = "#4f9cff" if point.slug in frontier_slugs else "#c7d0dc"
        axis.annotate(
            point.label,
            (x_getter(point), point.auc),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=12,
            color=color,
            alpha=0.98,
        )

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#c7d0dc", markersize=10, label="model point"),
        Line2D([0], [0], marker="o", color="#3b82f6", markerfacecolor="#3b82f6", markersize=10, linewidth=2.5, label="pareto frontier"),
    ]
    for color_index, slug in enumerate(sorted(search_series)):
        series = search_series[slug]
        color = SEARCH_COLORS[color_index % len(SEARCH_COLORS)]
        legend_handles.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor=color, alpha=0.5, markersize=8, label=f"{series.label} search")
        )
        legend_handles.append(
            Line2D([0], [0], color=color, linestyle=(0, (5, 3)), linewidth=2.0, label=f"{series.label} fit")
        )
    axis.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=10.5, labelcolor="#cbd5e1")

    axis.set_title(str(config["subtitle"]), fontsize=13, color="#94a3b8", pad=12)
    figure.suptitle(str(config["title"]), fontsize=24, fontweight="bold", color="#f8fafc", y=0.965)
    axis.set_xlabel(str(config["xlabel"]), fontsize=15, color="#e5e7eb", labelpad=16)
    axis.set_ylabel("Validation AUC", fontsize=15, color="#e5e7eb", labelpad=14)

    axis.grid(True, color="#2b313c", linewidth=1.0, alpha=0.85)
    for spine in axis.spines.values():
        spine.set_color("#111418")

    axis.tick_params(axis="both", colors="#cbd5e1", labelsize=12)
    axis.set_xscale(str(config["x_scale"]))
    x_ticks = config["x_ticks"]
    if x_ticks is not None:
        tick_values = [
            tick
            for tick in x_ticks
            if min(x_getter(point) for point in points) * 0.8 <= tick <= max(x_getter(point) for point in points) * 1.2
        ]
        axis.set_xticks(tick_values)
    axis.xaxis.set_major_formatter(config["x_formatter"])
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.3f}"))

    x_values = [x_getter(point) for point in points]
    if str(config["x_scale"]) == "log":
        axis.set_xlim(min(x_values) * 0.8, max(x_values) * 1.18)
    else:
        axis.set_xlim(min(x_values) * 0.92, max(x_values) * 1.12)
    axis.set_ylim(fill_floor, max(point.auc for point in points) + 0.02)

    figure.text(
        0.995,
        0.018,
        str(config["highlight_note"]),
        ha="right",
        va="bottom",
        fontsize=10.5,
        color="#94a3b8",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0.02, 0.04, 0.98, 0.94))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def plot_model_performance(
    *,
    summary_root: Path,
    search_root: Path,
    experiments_doc_path: Path,
    output_path: Path,
    x_metric: XMetric,
) -> None:
    base_points = load_base_points(summary_root, experiments_doc_path)
    search_series = load_search_series(search_root)
    render(merge_best_search_points(base_points, search_series), search_series, output_path, x_metric)


__all__ = [
    "DISPLAY_NAMES",
    "ModelPoint",
    "SearchSeries",
    "fit_curve",
    "load_base_points",
    "load_search_series",
    "merge_best_search_points",
    "metric_config",
    "pareto_frontier",
    "plot_model_performance",
    "render",
]
