from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import torch


@dataclass(slots=True)
class DatasetStats:
    dense_dim: int
    pos_weight: float
    train_size: int
    val_size: int
    train_positive_rate: float
    sequence_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(slots=True)
class BatchTensors:
    candidate_tokens: torch.Tensor
    candidate_mask: torch.Tensor
    context_tokens: torch.Tensor
    context_mask: torch.Tensor
    history_tokens: torch.Tensor
    history_mask: torch.Tensor
    history_component_tokens: torch.Tensor
    history_component_mask: torch.Tensor
    history_group_ids: torch.Tensor
    history_time_gaps: torch.Tensor
    history_positions: torch.Tensor
    sequence_tokens: torch.Tensor
    sequence_mask: torch.Tensor
    sequence_component_tokens: torch.Tensor
    sequence_component_mask: torch.Tensor
    sequence_time_gaps: torch.Tensor
    sequence_positions: torch.Tensor
    dense_features: torch.Tensor
    labels: torch.Tensor
    timestamps: torch.Tensor
    user_indices: torch.Tensor
    item_indices: torch.Tensor

    def to(self, device: torch.device) -> BatchTensors:
        moved: dict[str, torch.Tensor] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            moved[field.name] = value.to(device)
        return BatchTensors(**moved)

    def __getitem__(self, key: str) -> torch.Tensor:
        return getattr(self, key)

    @property
    def batch_size(self) -> int:
        return int(self.labels.size(0))


__all__ = ["BatchTensors", "DatasetStats"]