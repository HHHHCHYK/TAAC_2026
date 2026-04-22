from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from taac2026.application.search.trial import budget_status, execute_search_trial, profile_trial_budget, resolve_metric
from taac2026.domain.config import DataConfig, ModelConfig, TrainConfig
from taac2026.domain.experiment import ExperimentSpec
from taac2026.infrastructure.experiments.loader import load_experiment_package
from tests.support import TestWorkspace, build_local_data_pipeline, create_test_workspace


@pytest.fixture
def test_workspace(tmp_path: Path) -> TestWorkspace:
    return create_test_workspace(tmp_path)


def test_resolve_metric_reads_nested_paths() -> None:
    summary = {"metrics": {"gauc": {"value": 0.78}}}

    assert resolve_metric(summary, "metrics.gauc.value") == pytest.approx(0.78)


def test_resolve_metric_raises_for_missing_paths() -> None:
    with pytest.raises(KeyError, match="not present"):
        resolve_metric({"metrics": {"auc": 0.9}}, "metrics.gauc.value")


def test_budget_status_tracks_parameter_and_compute_limits() -> None:
    status = budget_status(
        {"parameter_size_mb": 128.0},
        load_experiment_package("config/baseline").search,
    )

    assert status["constraints_met"] is True
    assert status["model_compute_budget_met"] is True
    assert status["parameter_bytes"] == pytest.approx(128.0 * 1024.0 * 1024.0)
    assert status["model_flops_per_sample"] == pytest.approx(0.0)
    assert status["model_tflops_per_sample"] == pytest.approx(0.0)
    assert status["model_compute_profile_available"] is False
    assert status["max_model_tflops_per_sample"] is None


def test_budget_status_applies_explicit_compute_cap() -> None:
    experiment = load_experiment_package("config/baseline")
    experiment.search.max_model_tflops_per_sample = 10.0

    status = budget_status(
        {"parameter_size_mb": 128.0, "flops_per_sample": 12.5e12},
        experiment.search,
    )

    assert status["constraints_met"] is False
    assert status["model_compute_budget_met"] is False
    assert status["model_tflops_per_sample"] == pytest.approx(12.5)
    assert status["model_compute_profile_available"] is True
    assert status["model_compute_budget_reason"] == "model FLOPs/sample exceeds configured limit"
    assert status["max_model_tflops_per_sample"] == pytest.approx(10.0)


def test_budget_status_rejects_capped_search_when_model_flops_are_unavailable() -> None:
    experiment = load_experiment_package("config/baseline")
    experiment.search.max_model_tflops_per_sample = 10.0

    status = budget_status(
        {"parameter_size_mb": 128.0, "flops_per_sample": 0.0, "flops_profile_status": "unavailable"},
        experiment.search,
    )

    assert status["constraints_met"] is False
    assert status["model_compute_budget_met"] is False
    assert status["model_compute_profile_available"] is False
    assert status["model_compute_budget_reason"] == "model FLOPs profile unavailable"


def test_profile_trial_budget_skips_compute_profile_when_cap_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ProfileModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, dtype=torch.float32))

        def forward(self, batch) -> torch.Tensor:
            del batch
            raise AssertionError("compute profile should not run when the compute cap is disabled")

    captured: dict[str, object] = {}

    def build_data_pipeline(data_config, model_config, train_config):
        del data_config
        del model_config
        del train_config
        raise AssertionError("search budget probe should not build the real data pipeline when the compute cap is disabled")

    def build_model_component(data_config, model_config, dense_dim):
        captured["dense_dim"] = dense_dim
        captured["dataset_path"] = data_config.dataset_path
        captured["model_name"] = model_config.name
        return _ProfileModel()

    experiment = ExperimentSpec(
        name="custom_profile_batch",
        data=DataConfig(dataset_path="synthetic://profile"),
        model=ModelConfig(name="profile_model", vocab_size=8, embedding_dim=4, hidden_dim=4),
        train=TrainConfig(batch_size=2, eval_batch_size=2, num_workers=0),
        build_data_pipeline=build_data_pipeline,
        build_model_component=build_model_component,
    )

    monkeypatch.setattr(
        "taac2026.application.search.trial.collect_experiment_model_profile",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("compute profile should not be collected when the cap is disabled")),
    )

    result = profile_trial_budget(experiment)

    assert result["budget_status"]["constraints_met"] is True
    assert captured["dataset_path"] == "synthetic://profile"
    assert captured["model_name"] == "profile_model"
    assert captured["dense_dim"] == experiment.data.dense_feature_dim
    assert result["model_profile"]["flops_profile_status"] == "unavailable"
    assert result["budget_status"]["model_compute_profile_available"] is False


def test_profile_trial_budget_uses_pipeline_dense_dim_when_cap_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ProfileModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, dtype=torch.float32))

        def forward(self, batch) -> torch.Tensor:
            del batch
            return self.weight

    captured: dict[str, object] = {}

    def build_data_pipeline(data_config, model_config, train_config):
        del data_config
        del model_config
        del train_config
        return [], [], type("_Stats", (), {"dense_dim": 37})()

    def build_model_component(data_config, model_config, dense_dim):
        captured["dense_dim"] = dense_dim
        return _ProfileModel()

    experiment = ExperimentSpec(
        name="profile_pipeline_dense_dim",
        data=DataConfig(dataset_path="synthetic://profile", dense_feature_dim=16),
        model=ModelConfig(name="profile_model", vocab_size=8, embedding_dim=4, hidden_dim=4),
        train=TrainConfig(batch_size=2, eval_batch_size=2, num_workers=0),
        build_data_pipeline=build_data_pipeline,
        build_model_component=build_model_component,
    )
    experiment.search.max_model_tflops_per_sample = 1.0

    monkeypatch.setattr(
        "taac2026.application.search.trial.collect_experiment_model_profile",
        lambda *args, **kwargs: {"parameter_size_mb": 1.0, "flops_per_sample": 1.0e12, "flops_profile_status": "measured"},
    )

    result = profile_trial_budget(experiment)

    assert captured["dense_dim"] == 37
    assert result["budget_status"]["model_compute_profile_available"] is True


def test_profile_trial_budget_does_not_mask_forward_errors(test_workspace: TestWorkspace) -> None:
    class _GuardedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, dtype=torch.float32))

        def forward(self, batch) -> torch.Tensor:
            del batch
            raise RuntimeError("synthetic profile forward failed")
            return self.weight.repeat(batch.batch_size)

    experiment = ExperimentSpec(
        name="profile_forward_error",
        data=test_workspace.data_config,
        model=ModelConfig(name="profile_test", **test_workspace.model_kwargs),
        train=TrainConfig(batch_size=2, eval_batch_size=2, num_workers=0),
        build_data_pipeline=build_local_data_pipeline,
        build_model_component=lambda data_config, model_config, dense_dim: _GuardedModel(),
    )
    experiment.search.max_model_tflops_per_sample = 10.0

    with pytest.raises(RuntimeError, match="synthetic profile forward failed"):
        profile_trial_budget(experiment)


def test_execute_search_trial_prunes_before_training(
    test_workspace: TestWorkspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = load_experiment_package(test_workspace.write_experiment_package())
    called = {"training": False}

    monkeypatch.setattr(
        "taac2026.application.search.trial.profile_trial_budget",
        lambda experiment: {
            "model_profile": {},
            "latency": {},
            "inference_profile": {},
            "budget_status": {"constraints_met": False},
        },
    )
    monkeypatch.setattr(
        "taac2026.application.search.trial.run_training",
        lambda experiment: called.__setitem__("training", True),
    )

    result = execute_search_trial(experiment)

    assert result["status"] == "pruned"
    assert "before training" in str(result["prune_reason"])
    assert called["training"] is False


def test_execute_search_trial_reports_unavailable_model_flops_in_prune_reason(
    test_workspace: TestWorkspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = load_experiment_package(test_workspace.write_experiment_package())

    monkeypatch.setattr(
        "taac2026.application.search.trial.profile_trial_budget",
        lambda experiment: {
            "model_profile": {"flops_per_sample": 0.0, "flops_profile_status": "unavailable"},
            "budget_status": {
                "constraints_met": False,
                "model_compute_budget_reason": "model FLOPs profile unavailable",
            },
        },
    )

    result = execute_search_trial(experiment)

    assert result["status"] == "pruned"
    assert "model FLOPs profile unavailable" in str(result["prune_reason"])


def test_execute_search_trial_reuses_pretraining_budget_after_training(
    test_workspace: TestWorkspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = load_experiment_package(test_workspace.write_experiment_package())
    experiment.train.output_dir = str(test_workspace.root / "trial_budget")
    experiment.search.max_model_tflops_per_sample = 20.0

    monkeypatch.setattr(
        "taac2026.application.search.trial.profile_trial_budget",
        lambda experiment: {
            "model_profile": {},
            "budget_status": {
                "constraints_met": True,
                "model_tflops_per_sample": 12.5,
                "model_compute_profile_available": True,
            },
        },
    )
    monkeypatch.setattr(
        "taac2026.application.search.trial.run_training",
        lambda experiment: {
            "best_val_auc": 0.81,
            "model_profile": {"parameter_size_mb": 1.0, "flops_per_sample": 999.0e12},
            "metrics": {"auc": 0.81},
        },
    )

    result = execute_search_trial(experiment)

    assert result["status"] == "complete"
    assert result["prune_reason"] is None
    assert result["final_budget_status"]["constraints_met"] is True
    assert result["final_budget_status"]["model_tflops_per_sample"] == pytest.approx(12.5)
    assert result["summary_path"] == str(Path(experiment.train.output_dir) / "summary.json")


def test_execute_search_trial_uses_training_profile_for_uncapped_search(
    test_workspace: TestWorkspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = load_experiment_package(test_workspace.write_experiment_package())
    experiment.train.output_dir = str(test_workspace.root / "trial_budget_uncapped")

    monkeypatch.setattr(
        "taac2026.application.search.trial.profile_trial_budget",
        lambda experiment: {
            "model_profile": {"parameter_size_mb": 1.0, "flops_per_sample": 0.0, "flops_profile_status": "unavailable"},
            "budget_status": {
                "constraints_met": True,
                "model_tflops_per_sample": 0.0,
                "model_compute_profile_available": False,
                "max_model_tflops_per_sample": None,
            },
        },
    )
    monkeypatch.setattr(
        "taac2026.application.search.trial.run_training",
        lambda experiment: {
            "best_val_auc": 0.81,
            "model_profile": {"parameter_size_mb": 1.0, "flops_per_sample": 3.0e12, "flops_profile_status": "measured"},
            "metrics": {"auc": 0.81},
        },
    )

    result = execute_search_trial(experiment)

    assert result["status"] == "complete"
    assert result["final_budget_status"]["model_compute_profile_available"] is True
    assert result["final_budget_status"]["model_tflops_per_sample"] == pytest.approx(3.0)
    assert result["final_budget_status"]["max_model_tflops_per_sample"] is None


def test_execute_search_trial_surfaces_missing_metric_paths(
    test_workspace: TestWorkspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = load_experiment_package(test_workspace.write_experiment_package())
    experiment.search.metric_name = "metrics.gauc.value"

    monkeypatch.setattr(
        "taac2026.application.search.trial.profile_trial_budget",
        lambda experiment: {
            "model_profile": {},
            "budget_status": {"constraints_met": True},
        },
    )
    monkeypatch.setattr(
        "taac2026.application.search.trial.budget_status",
        lambda model_profile, search_config: {"constraints_met": True},
    )
    monkeypatch.setattr(
        "taac2026.application.search.trial.run_training",
        lambda experiment: {
            "model_profile": {"parameter_size_mb": 1.0, "flops_per_sample": 1.0e12},
            "metrics": {"auc": 0.81},
        },
    )

    with pytest.raises(KeyError, match="not present"):
        execute_search_trial(experiment)
