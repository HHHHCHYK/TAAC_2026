from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from taac2026.application.training.profiling import (
    build_synthetic_profile_batch,
    collect_loader_outputs,
    collect_synthetic_model_profile,
    measure_latency,
)
from taac2026.domain.config import ModelConfig, TrainConfig
from taac2026.domain.features import FeatureSchema, FeatureTableSpec, build_default_feature_schema
from tests.support import (
    TestWorkspace,
    build_local_data_pipeline,
    build_local_model_component,
    create_test_workspace,
)


@pytest.fixture
def test_workspace(tmp_path: Path) -> TestWorkspace:
    return create_test_workspace(tmp_path)


def _build_profile_model_and_batch(test_workspace: TestWorkspace):
    train_loader, _, data_stats = build_local_data_pipeline(
        test_workspace.data_config,
        ModelConfig(name="profile_test", **test_workspace.model_kwargs),
        TrainConfig(batch_size=2, eval_batch_size=2, num_workers=0),
    )
    model = build_local_model_component(
        test_workspace.data_config,
        ModelConfig(name="profile_test", **test_workspace.model_kwargs),
        data_stats.dense_dim,
    )
    return model, next(iter(train_loader))


def test_collect_loader_outputs_returns_empty_arrays_for_empty_loader(test_workspace: TestWorkspace) -> None:
    model, _ = _build_profile_model_and_batch(test_workspace)

    logits, labels, groups, loss = collect_loader_outputs(model, [], "cpu")

    assert logits.size == 0
    assert labels.size == 0
    assert groups.size == 0
    assert loss == pytest.approx(0.0)


def test_measure_latency_tracks_warmup_only_runs(test_workspace: TestWorkspace) -> None:
    model, batch = _build_profile_model_and_batch(test_workspace)

    latency = measure_latency(model, [batch], "cpu", warmup_steps=3, measure_steps=1)

    assert latency["warmup_batches"] == 1
    assert latency["warmup_samples"] == batch.batch_size
    assert latency["measured_batches"] == 0
    assert latency["measured_samples"] == 0
    assert latency["profiled_batches"] == 1


def test_measure_latency_zero_measure_steps_profiles_all_remaining_batches(
    test_workspace: TestWorkspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, batch = _build_profile_model_and_batch(test_workspace)
    ticks = iter([1.0, 1.004, 2.0, 2.008])
    monkeypatch.setattr("taac2026.application.training.profiling.time.perf_counter", lambda: next(ticks))

    latency = measure_latency(model, [batch, batch, batch], "cpu", warmup_steps=1, measure_steps=0)

    assert latency["warmup_batches"] == 1
    assert latency["measured_batches"] == 2
    assert latency["mean_latency_ms_per_sample"] == pytest.approx(3.0)
    assert latency["profiled_batches"] == 3


def test_build_synthetic_profile_batch_uses_fixed_shapes(test_workspace: TestWorkspace) -> None:
    model_config = ModelConfig(name="profile_test", **test_workspace.model_kwargs)

    batch = build_synthetic_profile_batch(test_workspace.data_config, model_config, batch_size=1)

    assert batch.batch_size == 1
    assert tuple(batch.dense_features.shape) == (1, test_workspace.data_config.dense_feature_dim)
    assert batch.sparse_features is not None
    assert batch.sequence_features is not None


def test_build_synthetic_profile_batch_supports_empty_sequence_names(test_workspace: TestWorkspace) -> None:
    model_config = ModelConfig(name="profile_test", **test_workspace.model_kwargs)
    data_config = test_workspace.data_config
    data_config.sequence_names = ()

    batch = build_synthetic_profile_batch(data_config, model_config, batch_size=1)

    assert batch.batch_size == 1
    assert batch.sparse_features is not None
    assert batch.sequence_features is not None


def test_build_synthetic_profile_batch_respects_explicit_empty_feature_schema_sequences(
    test_workspace: TestWorkspace,
) -> None:
    model_config = ModelConfig(name="profile_test", **test_workspace.model_kwargs)
    empty_sequence_data_config = replace(test_workspace.data_config, sequence_names=())
    empty_sequence_schema = build_default_feature_schema(empty_sequence_data_config, model_config)
    feature_schema = FeatureSchema(
        tables=empty_sequence_schema.tables,
        dense_dim=empty_sequence_schema.dense_dim,
        sequence_names=(),
        variant=empty_sequence_schema.variant,
        auto_sync=False,
    )

    batch = build_synthetic_profile_batch(
        test_workspace.data_config,
        model_config,
        feature_schema=feature_schema,
        batch_size=1,
    )

    assert batch.batch_size == 1
    assert batch.sparse_features is not None
    assert batch.sequence_features is not None


def test_build_synthetic_profile_batch_rejects_custom_feature_schema(test_workspace: TestWorkspace) -> None:
    model_config = ModelConfig(name="profile_test", **test_workspace.model_kwargs)
    feature_schema = FeatureSchema(
        tables=(
            FeatureTableSpec(
                name="custom_tokens",
                family="custom",
                num_embeddings=8,
                embedding_dim=test_workspace.model_kwargs["embedding_dim"],
            ),
        ),
        dense_dim=test_workspace.data_config.dense_feature_dim,
        sequence_names=(),
        variant="custom_profile_v1",
        auto_sync=False,
    )

    with pytest.raises(ValueError, match="Default data pipeline only supports the canonical TorchRec feature schema"):
        build_synthetic_profile_batch(
            test_workspace.data_config,
            model_config,
            feature_schema=feature_schema,
            batch_size=1,
        )


def test_collect_synthetic_model_profile_uses_fixed_fake_batch(test_workspace: TestWorkspace) -> None:
    model_config = ModelConfig(name="profile_test", **test_workspace.model_kwargs)
    _, _, data_stats = build_local_data_pipeline(
        test_workspace.data_config,
        model_config,
        TrainConfig(batch_size=2, eval_batch_size=2, num_workers=0),
    )
    model = build_local_model_component(
        test_workspace.data_config,
        model_config,
        data_stats.dense_dim,
    )

    profile = collect_synthetic_model_profile(model, test_workspace.data_config, model_config, "cpu")

    assert profile["profile_scope"] == "synthetic_fixed_forward"
    assert profile["profile_input_kind"] == "synthetic_fixed_batch"
    assert profile["profiled_batches"] == 1
    assert profile["selected_batch_index"] == 0
    assert profile["profile_batch_size"] == 1
    assert profile["flops_profile_status"] == "measured"
