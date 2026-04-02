from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SEQUENCE_NAMES = ("action_seq", "content_seq", "item_seq")


@dataclass(slots=True)
class DataConfig:
    dataset_path: str
    max_seq_len: int = 256
    max_feature_tokens: int = 64
    max_event_features: int = 12
    val_ratio: float = 0.2
    label_action_type: int = 2
    sequence_names: tuple[str, ...] = DEFAULT_SEQUENCE_NAMES

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DataConfig:
        normalized = dict(payload)
        if "sequence_names" in normalized:
            normalized["sequence_names"] = tuple(str(name) for name in normalized["sequence_names"])
        return cls(**normalized)


@dataclass(slots=True)
class ModelConfig:
    name: str = "grok_baseline"
    vocab_size: int = 200_003
    embedding_dim: int = 96
    hidden_dim: int = 192
    dropout: float = 0.1
    num_layers: int = 3
    num_heads: int = 4
    recent_seq_len: int = 32
    memory_slots: int = 12
    ffn_multiplier: int = 4
    feature_cross_layers: int = 1
    sequence_layers: int = 1
    static_layers: int = 1
    query_decoder_layers: int = 2
    fusion_layers: int = 1
    num_queries: int = 12
    head_hidden_dim: int = 0
    segment_count: int = 4


@dataclass(slots=True)
class TrainConfig:
    seed: int = 42
    epochs: int = 5
    batch_size: int = 64
    eval_batch_size: int = 0
    learning_rate: float = 1e-3
    muon_learning_rate: float = 2e-2
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "auto"
    optimizer_name: str = "adamw"
    loss_name: str = "bce"
    pairwise_weight: float = 0.5
    grad_clip_norm: float = 0.0
    output_dir: str = "outputs/baseline"
    use_amp: bool = False

    @property
    def resolved_eval_batch_size(self) -> int:
        return self.eval_batch_size or self.batch_size


@dataclass(slots=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    defaults = {
        "data": asdict(DataConfig(dataset_path="")),
        "model": asdict(ModelConfig()),
        "train": asdict(TrainConfig()),
    }
    merged = _merge_dict(defaults, raw)

    return ExperimentConfig(
        data=DataConfig.from_dict(merged["data"]),
        model=ModelConfig(**merged["model"]),
        train=TrainConfig(**merged["train"]),
    )


__all__ = [
    "DEFAULT_SEQUENCE_NAMES",
    "DataConfig",
    "ExperimentConfig",
    "ModelConfig",
    "TrainConfig",
    "load_config",
]