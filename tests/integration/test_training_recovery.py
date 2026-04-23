from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from taac2026.application.training.service import run_training
from taac2026.domain.config import ModelConfig, TrainConfig
from taac2026.domain.experiment import ExperimentSpec
from tests.support import (
    TestWorkspace,
    build_local_data_pipeline,
    build_local_loss_stack,
    build_local_model_component,
    build_local_optimizer_component,
    create_test_workspace,
)


@pytest.fixture
def test_workspace(tmp_path: Path) -> TestWorkspace:
    return create_test_workspace(tmp_path)


def _build_experiment(
    test_workspace: TestWorkspace,
    *,
    output_dir: Path,
    epochs: int = 1,
    enable_amp: bool = False,
) -> ExperimentSpec:
    return ExperimentSpec(
        name="training_recovery_test",
        data=test_workspace.data_config,
        model=ModelConfig(name="training_recovery_test", **test_workspace.model_kwargs),
        train=TrainConfig(
            seed=7,
            epochs=epochs,
            batch_size=2,
            eval_batch_size=2,
            num_workers=0,
            output_dir=str(output_dir),
            latency_warmup_steps=0,
            latency_measure_steps=1,
            enable_amp=enable_amp,
            amp_dtype="bfloat16" if enable_amp else "float16",
        ),
        build_data_pipeline=build_local_data_pipeline,
        build_model_component=build_local_model_component,
        build_loss_stack=build_local_loss_stack,
        build_optimizer_component=build_local_optimizer_component,
    )


def test_run_training_recovers_after_interrupt(test_workspace: TestWorkspace) -> None:
    output_dir = test_workspace.root / "interrupted_run"
    interrupted_experiment = _build_experiment(test_workspace, output_dir=output_dir, epochs=2)

    class _InterruptingLoader:
        def __init__(self, base_loader) -> None:
            self._base_loader = base_loader
            self._iteration_count = 0

        def __iter__(self):
            self._iteration_count += 1
            if self._iteration_count == 2:
                raise KeyboardInterrupt("simulated interrupt during second epoch")
            yield from self._base_loader

        def __len__(self) -> int:
            return len(self._base_loader)

    def interrupting_data_pipeline(data_config, model_config, train_config):
        train_loader, val_loader, data_stats = build_local_data_pipeline(data_config, model_config, train_config)
        return _InterruptingLoader(train_loader), val_loader, data_stats

    interrupted_experiment.build_data_pipeline = interrupting_data_pipeline

    with pytest.raises(KeyboardInterrupt, match="simulated interrupt"):
        run_training(interrupted_experiment)

    assert (output_dir / "best.pt").exists()
    assert not (output_dir / "summary.json").exists()

    recovered_experiment = _build_experiment(test_workspace, output_dir=output_dir, epochs=2)
    summary = run_training(recovered_experiment)
    curves = json.loads((output_dir / "training_curves.json").read_text(encoding="utf-8"))

    assert summary["best_epoch"] >= 1
    assert len(curves["train_loss"]) == 2
    assert len(curves["val_auc"]) == 2
    assert (output_dir / "summary.json").exists()


def test_run_training_keeps_previous_curve_artifacts_when_plot_write_fails(
    test_workspace: TestWorkspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = test_workspace.root / "curve_artifact_failure"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_target = output_dir / "training_curves.json"
    png_target = output_dir / "training_curves.png"
    json_target.write_text(
        json.dumps({"best_epoch": 9, "train_loss": [9.0], "val_auc": [0.9], "val_loss": [8.0]}, indent=2),
        encoding="utf-8",
    )
    png_target.write_bytes(b"previous-plot")

    monkeypatch.setattr(
        "taac2026.application.training.artifacts.render_training_curves_plot",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("plot render failed")),
    )

    experiment = _build_experiment(test_workspace, output_dir=output_dir, epochs=1)

    with pytest.raises(RuntimeError, match="plot render failed"):
        run_training(experiment)

    assert json.loads(json_target.read_text(encoding="utf-8"))["best_epoch"] == 9
    assert png_target.read_bytes() == b"previous-plot"
    assert not any(path.name.endswith(".tmp") or ".tmp." in path.name for path in output_dir.iterdir())
    assert not (output_dir / "summary.json").exists()


def test_run_training_overwrites_existing_checkpoint_consistently(test_workspace: TestWorkspace) -> None:
    output_dir = test_workspace.root / "checkpoint_overwrite"
    base_experiment = _build_experiment(test_workspace, output_dir=output_dir, epochs=1, enable_amp=False)
    amp_experiment = _build_experiment(test_workspace, output_dir=output_dir, epochs=1, enable_amp=True)

    run_training(base_experiment)
    first_payload = torch.load(output_dir / "best.pt", map_location="cpu")

    run_training(amp_experiment)
    second_payload = torch.load(output_dir / "best.pt", map_location="cpu")

    assert first_payload["runtime_optimization"]["amp"]["requested"] is False
    assert second_payload["runtime_optimization"]["amp"]["requested"] is True
    assert second_payload["runtime_optimization"]["amp"]["active"] is True
    assert second_payload["epoch"] == 1