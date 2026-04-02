from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig
from .types import BatchTensors, DatasetStats
from .utils import stable_hash


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def _normalize_feature_collection(features: Any) -> list[dict[str, Any]]:
    if features is None:
        return []
    if isinstance(features, float) and np.isnan(features):
        return []
    return list(features)


def _feature_tokens_and_stats(
    sources: list[tuple[str, Any]],
    vocab_size: int,
    max_feature_tokens: int,
) -> tuple[list[int], list[float]]:
    tokens: list[int] = []
    float_values: list[float] = []
    feature_count = 0
    sparse_count = 0

    for prefix, raw_features in sources:
        feature_list = _normalize_feature_collection(raw_features)
        feature_count += len(feature_list)

        for feature in feature_list:
            if len(tokens) >= max_feature_tokens:
                break

            feature_id = int(feature["feature_id"])
            int_value = feature.get("int_value")
            if not _is_missing(int_value) and len(tokens) < max_feature_tokens:
                tokens.append(stable_hash(f"{prefix}|{feature_id}|iv|{int(int_value)}", vocab_size))
                sparse_count += 1

            int_array = feature.get("int_array")
            if int_array is not None and len(tokens) < max_feature_tokens:
                remaining = max_feature_tokens - len(tokens)
                for element in np.asarray(int_array).tolist()[:remaining]:
                    tokens.append(stable_hash(f"{prefix}|{feature_id}|ia|{int(element)}", vocab_size))
                    sparse_count += 1

            float_value = feature.get("float_value")
            if not _is_missing(float_value):
                float_values.append(float(float_value))

            float_array = feature.get("float_array")
            if float_array is not None:
                float_values.extend(float(value) for value in np.asarray(float_array).tolist())

    if float_values:
        float_array_np = np.asarray(float_values, dtype=np.float32)
        float_mean = float(float_array_np.mean())
        float_std = float(float_array_np.std())
        float_max_abs = float(np.abs(float_array_np).max())
        float_count = float(float_array_np.size)
    else:
        float_mean = 0.0
        float_std = 0.0
        float_max_abs = 0.0
        float_count = 0.0

    dense_stats = [
        float(feature_count),
        float(sparse_count),
        float_count,
        float_mean,
        float_std,
        float_max_abs,
    ]
    return tokens[:max_feature_tokens], dense_stats


def _timestamp_feature_id(arrays: list[tuple[int, np.ndarray]]) -> int | None:
    for feature_id, values in arrays:
        if values.size == 0:
            continue
        if float(np.median(values)) > 1_000_000_000:
            return feature_id
    return None


def _build_group_events(
    feature_group: Any,
    group_name: str,
    group_id: int,
    row_timestamp: int,
    vocab_size: int,
    max_event_features: int,
    max_seq_len: int,
) -> tuple[list[tuple[float, int, list[int], int]], list[float]]:
    arrays: list[tuple[int, np.ndarray]] = []
    for feature in _normalize_feature_collection(feature_group):
        values = feature.get("int_array")
        if values is None:
            continue
        arrays.append((int(feature["feature_id"]), np.asarray(values)))

    if not arrays:
        return [], []

    timestamp_feature_id = _timestamp_feature_id(arrays)
    timestamp_values = next((values for feature_id, values in arrays if feature_id == timestamp_feature_id), None)
    non_timestamp_arrays = [
        (feature_id, values)
        for feature_id, values in arrays
        if feature_id != timestamp_feature_id
    ]

    event_count = min(min(len(values) for _, values in arrays), max_seq_len)
    events: list[tuple[float, int, list[int], int]] = []
    gap_stats: list[float] = []

    for index in range(event_count):
        signature_parts = [group_name]
        component_tokens: list[int] = []
        for feature_id, values in non_timestamp_arrays:
            feature_value = int(values[index])
            signature_parts.append(f"{feature_id}:{feature_value}")
            component_tokens.append(stable_hash(f"{group_name}|{feature_id}|{feature_value}", vocab_size))

        if not component_tokens:
            component_tokens.append(stable_hash(f"{group_name}|empty_event", vocab_size))

        token = stable_hash("|".join(signature_parts), vocab_size)
        raw_gap = max(row_timestamp - int(timestamp_values[index]), 0) if timestamp_values is not None else index
        events.append((float(raw_gap), token, component_tokens[:max_event_features], group_id))
        gap_stats.append(float(np.log1p(raw_gap)))

    events.sort(key=lambda item: item[0])
    return events[:max_seq_len], gap_stats


def _build_sequence_features(
    row: pd.Series,
    config: DataConfig,
    vocab_size: int,
) -> tuple[
    list[int],
    list[list[int]],
    list[int],
    list[float],
    list[list[int]],
    list[list[list[int]]],
    list[list[float]],
    list[float],
]:
    row_timestamp = int(row["timestamp"])
    raw_sequence_map = row["seq_feature"] if row["seq_feature"] is not None else {}

    grouped_events: list[list[tuple[float, int, list[int], int]]] = []
    merged_events: list[tuple[float, int, list[int], int]] = []
    gap_stats: list[float] = []

    for group_index, group_name in enumerate(config.sequence_names, start=1):
        group_events, local_gap_stats = _build_group_events(
            feature_group=raw_sequence_map.get(group_name, []),
            group_name=group_name,
            group_id=group_index,
            row_timestamp=row_timestamp,
            vocab_size=vocab_size,
            max_event_features=config.max_event_features,
            max_seq_len=config.max_seq_len,
        )
        grouped_events.append(group_events)
        merged_events.extend(group_events)
        gap_stats.extend(local_gap_stats)

    merged_events.sort(key=lambda item: item[0])
    selected = merged_events[: config.max_seq_len]

    history_tokens = [token for _, token, _, _ in selected]
    history_component_tokens = [component_tokens for _, _, component_tokens, _ in selected]
    history_group_ids = [group_id for _, _, _, group_id in selected]
    history_time_gaps = [float(np.log1p(raw_gap)) for raw_gap, _, _, _ in selected]

    sequence_tokens = [[token for _, token, _, _ in group_events] for group_events in grouped_events]
    sequence_component_tokens = [
        [component_tokens for _, _, component_tokens, _ in group_events]
        for group_events in grouped_events
    ]
    sequence_time_gaps = [
        [float(np.log1p(raw_gap)) for raw_gap, _, _, _ in group_events]
        for group_events in grouped_events
    ]

    if gap_stats:
        gap_array = np.asarray(gap_stats, dtype=np.float32)
        dense_stats = [
            float(len(merged_events)),
            float(gap_array.mean()),
            float(gap_array.std()),
            float(gap_array.min()),
            float(gap_array.max()),
            float(len(selected)),
        ]
    else:
        dense_stats = [0.0] * 6

    return (
        history_tokens,
        history_component_tokens,
        history_group_ids,
        history_time_gaps,
        sequence_tokens,
        sequence_component_tokens,
        sequence_time_gaps,
        dense_stats,
    )


@dataclass(slots=True)
class EncodedSample:
    candidate_tokens: list[int]
    context_tokens: list[int]
    history_tokens: list[int]
    history_component_tokens: list[list[int]]
    history_group_ids: list[int]
    history_time_gaps: list[float]
    sequence_tokens: list[list[int]]
    sequence_component_tokens: list[list[list[int]]]
    sequence_time_gaps: list[list[float]]
    dense_features: list[float]
    label: float
    timestamp: int
    user_index: int
    item_index: int


def encode_row(row: pd.Series, config: DataConfig, vocab_size: int) -> EncodedSample:
    candidate_tokens, candidate_dense = _feature_tokens_and_stats(
        [("item", row.get("item_feature"))],
        vocab_size=vocab_size,
        max_feature_tokens=config.max_feature_tokens,
    )
    candidate_tokens.insert(0, stable_hash(f"target_item|{int(row['item_id'])}", vocab_size))

    context_tokens, context_dense = _feature_tokens_and_stats(
        [("user", row.get("user_feature")), ("context", row.get("context_feature"))],
        vocab_size=vocab_size,
        max_feature_tokens=config.max_feature_tokens,
    )
    time_bucket = int(row["timestamp"]) // 3600
    context_tokens.append(stable_hash(f"hour_bucket|{time_bucket}", vocab_size))
    context_tokens.append(stable_hash(f"user_id|{row['user_id']}", vocab_size))

    (
        history_tokens,
        history_component_tokens,
        history_group_ids,
        history_time_gaps,
        sequence_tokens,
        sequence_component_tokens,
        sequence_time_gaps,
        history_dense,
    ) = _build_sequence_features(row=row, config=config, vocab_size=vocab_size)

    label_entries = row["label"] if row["label"] is not None else []
    label = 1.0 if any(int(entry["action_type"]) == config.label_action_type for entry in label_entries) else 0.0
    dense_features = candidate_dense + context_dense + history_dense + [float(np.log1p(int(row["timestamp"])))]

    return EncodedSample(
        candidate_tokens=candidate_tokens[: config.max_feature_tokens],
        context_tokens=context_tokens[: config.max_feature_tokens],
        history_tokens=history_tokens,
        history_component_tokens=history_component_tokens,
        history_group_ids=history_group_ids,
        history_time_gaps=history_time_gaps,
        sequence_tokens=sequence_tokens,
        sequence_component_tokens=sequence_component_tokens,
        sequence_time_gaps=sequence_time_gaps,
        dense_features=dense_features,
        label=label,
        timestamp=int(row["timestamp"]),
        user_index=int(row["user_index"]),
        item_index=int(row["item_index"]),
    )


class TaacDataset(Dataset[EncodedSample]):
    def __init__(self, samples: list[EncodedSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> EncodedSample:
        return self.samples[index]


def collate_samples(batch: list[EncodedSample]) -> BatchTensors:
    batch_size = len(batch)
    group_count = len(batch[0].sequence_tokens)
    dense_dim = len(batch[0].dense_features)

    max_candidate_len = max(len(sample.candidate_tokens) for sample in batch)
    max_context_len = max(len(sample.context_tokens) for sample in batch)
    max_history_len = max(max(len(sample.history_tokens), 1) for sample in batch)
    max_history_components = max(
        max((len(component_tokens) for component_tokens in sample.history_component_tokens), default=1)
        for sample in batch
    )
    max_group_len = max(
        max((len(group_tokens) for group_tokens in sample.sequence_tokens), default=0)
        for sample in batch
    )
    max_group_len = max(max_group_len, 1)
    max_group_components = max(
        max(
            (len(component_tokens) for group in sample.sequence_component_tokens for component_tokens in group),
            default=1,
        )
        for sample in batch
    )

    candidate_tokens = torch.zeros((batch_size, max_candidate_len), dtype=torch.long)
    candidate_mask = torch.zeros((batch_size, max_candidate_len), dtype=torch.bool)
    context_tokens = torch.zeros((batch_size, max_context_len), dtype=torch.long)
    context_mask = torch.zeros((batch_size, max_context_len), dtype=torch.bool)

    history_tokens = torch.zeros((batch_size, max_history_len), dtype=torch.long)
    history_mask = torch.zeros((batch_size, max_history_len), dtype=torch.bool)
    history_component_tokens = torch.zeros((batch_size, max_history_len, max_history_components), dtype=torch.long)
    history_component_mask = torch.zeros((batch_size, max_history_len, max_history_components), dtype=torch.bool)
    history_group_ids = torch.zeros((batch_size, max_history_len), dtype=torch.long)
    history_time_gaps = torch.zeros((batch_size, max_history_len), dtype=torch.float32)
    history_positions = torch.zeros((batch_size, max_history_len), dtype=torch.long)

    sequence_tokens = torch.zeros((batch_size, group_count, max_group_len), dtype=torch.long)
    sequence_mask = torch.zeros((batch_size, group_count, max_group_len), dtype=torch.bool)
    sequence_component_tokens = torch.zeros(
        (batch_size, group_count, max_group_len, max_group_components),
        dtype=torch.long,
    )
    sequence_component_mask = torch.zeros(
        (batch_size, group_count, max_group_len, max_group_components),
        dtype=torch.bool,
    )
    sequence_time_gaps = torch.zeros((batch_size, group_count, max_group_len), dtype=torch.float32)
    sequence_positions = torch.zeros((batch_size, group_count, max_group_len), dtype=torch.long)

    dense_features = torch.zeros((batch_size, dense_dim), dtype=torch.float32)
    labels = torch.zeros(batch_size, dtype=torch.float32)
    timestamps = torch.zeros(batch_size, dtype=torch.long)
    user_indices = torch.zeros(batch_size, dtype=torch.long)
    item_indices = torch.zeros(batch_size, dtype=torch.long)

    for row_index, sample in enumerate(batch):
        candidate_length = len(sample.candidate_tokens)
        context_length = len(sample.context_tokens)
        history_length = len(sample.history_tokens)

        candidate_tokens[row_index, :candidate_length] = torch.tensor(sample.candidate_tokens, dtype=torch.long)
        candidate_mask[row_index, :candidate_length] = True

        context_tokens[row_index, :context_length] = torch.tensor(sample.context_tokens, dtype=torch.long)
        context_mask[row_index, :context_length] = True

        if history_length > 0:
            history_tokens[row_index, :history_length] = torch.tensor(sample.history_tokens, dtype=torch.long)
            history_mask[row_index, :history_length] = True
            history_group_ids[row_index, :history_length] = torch.tensor(sample.history_group_ids, dtype=torch.long)
            history_time_gaps[row_index, :history_length] = torch.tensor(sample.history_time_gaps, dtype=torch.float32)
            history_positions[row_index, :history_length] = torch.arange(history_length, dtype=torch.long)

            for event_index, component_tokens_per_event in enumerate(sample.history_component_tokens):
                component_length = len(component_tokens_per_event)
                history_component_tokens[row_index, event_index, :component_length] = torch.tensor(
                    component_tokens_per_event,
                    dtype=torch.long,
                )
                history_component_mask[row_index, event_index, :component_length] = True

        for group_index, group_tokens in enumerate(sample.sequence_tokens):
            group_length = len(group_tokens)
            if group_length == 0:
                continue

            sequence_tokens[row_index, group_index, :group_length] = torch.tensor(group_tokens, dtype=torch.long)
            sequence_mask[row_index, group_index, :group_length] = True
            sequence_time_gaps[row_index, group_index, :group_length] = torch.tensor(
                sample.sequence_time_gaps[group_index],
                dtype=torch.float32,
            )
            sequence_positions[row_index, group_index, :group_length] = torch.arange(group_length, dtype=torch.long)

            for event_index, component_tokens_per_event in enumerate(sample.sequence_component_tokens[group_index]):
                component_length = len(component_tokens_per_event)
                sequence_component_tokens[row_index, group_index, event_index, :component_length] = torch.tensor(
                    component_tokens_per_event,
                    dtype=torch.long,
                )
                sequence_component_mask[row_index, group_index, event_index, :component_length] = True

        dense_features[row_index] = torch.tensor(sample.dense_features, dtype=torch.float32)
        labels[row_index] = sample.label
        timestamps[row_index] = sample.timestamp
        user_indices[row_index] = sample.user_index
        item_indices[row_index] = sample.item_index

    return BatchTensors(
        candidate_tokens=candidate_tokens,
        candidate_mask=candidate_mask,
        context_tokens=context_tokens,
        context_mask=context_mask,
        history_tokens=history_tokens,
        history_mask=history_mask,
        history_component_tokens=history_component_tokens,
        history_component_mask=history_component_mask,
        history_group_ids=history_group_ids,
        history_time_gaps=history_time_gaps,
        history_positions=history_positions,
        sequence_tokens=sequence_tokens,
        sequence_mask=sequence_mask,
        sequence_component_tokens=sequence_component_tokens,
        sequence_component_mask=sequence_component_mask,
        sequence_time_gaps=sequence_time_gaps,
        sequence_positions=sequence_positions,
        dense_features=dense_features,
        labels=labels,
        timestamps=timestamps,
        user_indices=user_indices,
        item_indices=item_indices,
    )


def load_dataloaders(
    config: DataConfig,
    vocab_size: int,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DatasetStats]:
    dataframe = pd.read_parquet(Path(config.dataset_path))
    dataframe = dataframe.sort_values("timestamp").reset_index(drop=True)
    dataframe["user_index"] = pd.factorize(dataframe["user_id"].astype(str), sort=False)[0].astype(np.int64)
    dataframe["item_index"] = pd.factorize(dataframe["item_id"].astype(str), sort=False)[0].astype(np.int64)

    samples = [encode_row(row, config=config, vocab_size=vocab_size) for _, row in dataframe.iterrows()]
    if len(samples) < 2:
        raise ValueError("Dataset must contain at least 2 rows to form train/validation splits.")

    split_index = max(1, int(len(samples) * (1.0 - config.val_ratio)))
    split_index = min(split_index, len(samples) - 1)
    train_samples = samples[:split_index]
    val_samples = samples[split_index:]

    train_labels = np.asarray([sample.label for sample in train_samples], dtype=np.float32)
    positive_count = float(train_labels.sum())
    negative_count = float(train_labels.shape[0] - positive_count)

    train_loader = DataLoader(
        TaacDataset(train_samples),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_samples,
    )
    val_loader = DataLoader(
        TaacDataset(val_samples),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_samples,
    )

    return train_loader, val_loader, DatasetStats(
        dense_dim=len(samples[0].dense_features),
        pos_weight=negative_count / max(positive_count, 1.0),
        train_size=len(train_samples),
        val_size=len(val_samples),
        train_positive_rate=float(train_labels.mean()) if train_labels.size else 0.0,
        sequence_count=len(config.sequence_names),
    )


__all__ = ["EncodedSample", "collate_samples", "encode_row", "load_dataloaders"]