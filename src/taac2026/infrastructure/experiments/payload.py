from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ...domain.config import DataConfig, ModelConfig, SearchConfig, TrainConfig
from ...domain.experiment import ExperimentSpec


LEGACY_SEARCH_KEYS = (
    "max_end_to_end_inference_seconds",
    "max_end_to_end_tflops_total",
)


def _normalize_search_payload(search_payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(search_payload)
    for key in LEGACY_SEARCH_KEYS:
        normalized.pop(key, None)
    return normalized


def serialize_experiment(experiment: ExperimentSpec) -> dict[str, Any]:
    return {
        "name": experiment.name,
        "data": asdict(experiment.data),
        "model": asdict(experiment.model),
        "train": asdict(experiment.train),
        "search": asdict(experiment.search),
        "switches": dict(experiment.switches),
    }


def apply_serialized_experiment(
    base_experiment: ExperimentSpec,
    payload: dict[str, Any],
) -> ExperimentSpec:
    data_payload = dict(payload["data"])
    if "sequence_names" in data_payload:
        data_payload["sequence_names"] = tuple(data_payload["sequence_names"])

    train_payload = dict(payload["train"])
    train_payload["switches"] = dict(train_payload.get("switches") or {})

    experiment = base_experiment.clone()
    experiment.name = str(payload["name"])
    experiment.data = DataConfig(**data_payload)
    experiment.model = ModelConfig(**dict(payload["model"]))
    experiment.train = TrainConfig(**train_payload)
    experiment.search = SearchConfig(**_normalize_search_payload(dict(payload["search"])))
    experiment.switches = dict(payload.get("switches") or {})
    experiment.refresh_feature_schema()
    return experiment


__all__ = ["apply_serialized_experiment", "serialize_experiment"]
