from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import statistics
import time
from collections.abc import Callable

import torch

from taac2026.infrastructure.nn.transformer import TaacCrossAttentionBlock, TaacTransformerBlock


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "performance" / "transformer_backends.json"
BACKEND_CHOICES = ("torch", "triton", "te")
SCENARIO_CHOICES = ("self-masked", "self-no-mask", "self-causal", "cross-masked", "cross-no-mask")
DEFAULT_PROFILE_NAMES = ("hyformer-default", "deepcontextnet-default")
DEFAULT_SCENARIOS = ("self-masked", "self-no-mask", "self-causal", "cross-masked")
DEFAULT_BACKENDS = BACKEND_CHOICES

TimingSummary = dict[str, float | list[float]]


@dataclass(frozen=True, slots=True)
class ShapeProfile:
    name: str
    batch_size: int
    query_length: int
    context_length: int
    hidden_dim: int
    num_heads: int
    ffn_dim: int


def load_experiment_profile(experiment_name: str) -> ShapeProfile:
    module = importlib.import_module(f"config.{experiment_name}")
    experiment = module.EXPERIMENT
    return ShapeProfile(
        name=f"{experiment_name}-default",
        batch_size=int(experiment.train.resolved_eval_batch_size),
        query_length=int(experiment.data.max_seq_len),
        context_length=max(int(experiment.data.max_seq_len), int(experiment.model.recent_seq_len or 0)),
        hidden_dim=int(experiment.model.hidden_dim),
        num_heads=int(experiment.model.num_heads),
        ffn_dim=int(experiment.model.hidden_dim * experiment.model.ffn_multiplier),
    )


def build_profiles() -> dict[str, ShapeProfile]:
    return {
        "hyformer-default": load_experiment_profile("hyformer"),
        "deepcontextnet-default": load_experiment_profile("deepcontextnet"),
        "medium-reference": ShapeProfile(
            name="medium-reference",
            batch_size=32,
            query_length=128,
            context_length=128,
            hidden_dim=256,
            num_heads=8,
            ffn_dim=1024,
        ),
        "large-reference": ShapeProfile(
            name="large-reference",
            batch_size=16,
            query_length=256,
            context_length=256,
            hidden_dim=512,
            num_heads=8,
            ffn_dim=2048,
        ),
    }


def build_prefix_mask(batch_size: int, sequence_length: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(sequence_length, device=device).unsqueeze(0)
    max_reduction = max(1, sequence_length // 4)
    reductions = torch.arange(batch_size, device=device) % (max_reduction + 1)
    minimum_length = max(1, sequence_length - max_reduction)
    lengths = torch.clamp(sequence_length - reductions, min=minimum_length)
    return positions < lengths.unsqueeze(1)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize_times(times_ms: list[float]) -> TimingSummary:
    ordered = sorted(times_ms)
    median_ms = float(statistics.median(ordered))
    mean_ms = float(statistics.fmean(ordered))
    p95_index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * 0.95)))
    p99_index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * 0.99)))
    return {
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
        "p95_ms": float(ordered[p95_index]),
        "p99_ms": float(ordered[p99_index]),
        "times_ms": [float(value) for value in times_ms],
    }


def measure_callable(
    target: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    steps: int,
) -> TimingSummary:
    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            target()
        synchronize(device)

        times_ms: list[float] = []
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            for _ in range(steps):
                start_event.record()
                target()
                end_event.record()
                torch.cuda.synchronize(device)
                times_ms.append(float(start_event.elapsed_time(end_event)))
        else:
            for _ in range(steps):
                start_time = time.perf_counter()
                target()
                times_ms.append((time.perf_counter() - start_time) * 1e3)
    return summarize_times(times_ms)


def build_case(
    *,
    scenario: str,
    backend: str,
    profile: ShapeProfile,
    device: torch.device,
    te_precision: str,
    te_recipe_mode: str,
) -> Callable[[], torch.Tensor]:
    if scenario.startswith("self-"):
        block = TaacTransformerBlock(
            hidden_dim=profile.hidden_dim,
            num_heads=profile.num_heads,
            ffn_dim=profile.ffn_dim,
            dropout=0.0,
            attention_dropout=0.0,
            norm_type="rmsnorm",
            ffn_type="swiglu",
            attention_type="causal" if scenario == "self-causal" else "standard",
            attention_backend=backend,
            ffn_backend=backend,
            te_precision=te_precision,
            te_recipe_mode=te_recipe_mode,
        ).to(device)
        block.eval()
        hidden_states = torch.randn(
            profile.batch_size,
            profile.query_length,
            profile.hidden_dim,
            device=device,
        )
        token_mask = build_prefix_mask(profile.batch_size, profile.query_length, device) if scenario == "self-masked" else None

        def run() -> torch.Tensor:
            return block(hidden_states, token_mask)

        return run

    block = TaacCrossAttentionBlock(
        hidden_dim=profile.hidden_dim,
        num_heads=profile.num_heads,
        ffn_dim=profile.ffn_dim,
        dropout=0.0,
        attention_dropout=0.0,
        norm_type="layernorm",
        ffn_type="gelu",
        attention_backend=backend,
        ffn_backend=backend,
        te_precision=te_precision,
        te_recipe_mode=te_recipe_mode,
    ).to(device)
    block.eval()
    query_states = torch.randn(
        profile.batch_size,
        profile.query_length,
        profile.hidden_dim,
        device=device,
    )
    context_states = torch.randn(
        profile.batch_size,
        profile.context_length,
        profile.hidden_dim,
        device=device,
    )
    if scenario == "cross-masked":
        query_mask = build_prefix_mask(profile.batch_size, profile.query_length, device)
        context_mask = build_prefix_mask(profile.batch_size, profile.context_length, device)
    else:
        query_mask = None
        context_mask = None

    def run() -> torch.Tensor:
        return block(query_states, context_states, query_mask=query_mask, context_mask=context_mask)

    return run