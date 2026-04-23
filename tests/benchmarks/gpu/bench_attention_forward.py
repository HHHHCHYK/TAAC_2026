from __future__ import annotations

import pytest
import torch

from taac2026.infrastructure.nn.triton_attention import triton_attention


BATCH_SIZE = 16
SEQUENCE_LENGTH = 32
HIDDEN_DIM = 128
NUM_HEADS = 4
HEAD_DIM = HIDDEN_DIM // NUM_HEADS

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for Triton attention benchmarks",
)


def test_attention_forward_triton_phase3(benchmark, benchmark_device, cuda_timer, performance_recorder) -> None:
    query = torch.randn(BATCH_SIZE, NUM_HEADS, SEQUENCE_LENGTH, HEAD_DIM, device=benchmark_device)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, SEQUENCE_LENGTH, HEAD_DIM, device=benchmark_device)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, SEQUENCE_LENGTH, HEAD_DIM, device=benchmark_device)

    def run() -> torch.Tensor:
        return triton_attention(query, key, value, backend="triton")

    benchmark.extra_info.update({
        "component": "attention",
        "phase": "phase-3",
        "label": "phase-3",
        "metric": "latency",
    })
    benchmark(run)

    stats = cuda_timer(run)
    stats.update({
        "name": "attention_forward_triton",
        "component": "attention",
        "phase": "phase-3",
        "label": "phase-3",
        "throughput": float(BATCH_SIZE * SEQUENCE_LENGTH) / max(stats["median_ms"] / 1e3, 1e-9),
    })
    performance_recorder("attention_forward_triton", stats)