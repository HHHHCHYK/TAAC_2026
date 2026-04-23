from __future__ import annotations

import torch


BATCH_SIZE = 16
SEQUENCE_LENGTH = 32
HIDDEN_DIM = 128
NUM_HEADS = 4


def test_attention_forward_baseline(benchmark, benchmark_device, cuda_timer, performance_recorder) -> None:
    attention = torch.nn.MultiheadAttention(HIDDEN_DIM, num_heads=NUM_HEADS, batch_first=True, device=benchmark_device)
    tokens = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_DIM, device=benchmark_device)

    def run() -> torch.Tensor:
        return attention(tokens, tokens, tokens, need_weights=False)[0]

    benchmark.extra_info.update({
        "component": "attention",
        "phase": "baseline",
        "label": "phase-0",
        "metric": "latency",
    })
    benchmark(run)

    stats = cuda_timer(run)
    stats.update({
        "name": "attention_forward",
        "component": "attention",
        "phase": "baseline",
        "label": "phase-0",
        "throughput": float(BATCH_SIZE * SEQUENCE_LENGTH) / max(stats["median_ms"] / 1e3, 1e-9),
    })
    performance_recorder("attention_forward", stats)