from __future__ import annotations

import torch


BATCH_SIZE = 1024
SEQUENCE_LENGTH = 64
VOCAB_SIZE = 131_072
EMBEDDING_DIM = 96


def test_embedding_lookup_baseline(benchmark, benchmark_device, cuda_timer, performance_recorder) -> None:
    embedding = torch.nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, device=benchmark_device)
    tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQUENCE_LENGTH), device=benchmark_device)
    mask = torch.ones_like(tokens, dtype=torch.bool)

    def run() -> torch.Tensor:
        embedded = embedding(tokens)
        weights = mask.unsqueeze(-1)
        return (embedded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

    benchmark.extra_info.update({
        "component": "embedding",
        "phase": "baseline",
        "label": "phase-0",
        "metric": "latency",
    })
    benchmark(run)

    stats = cuda_timer(run)
    stats.update({
        "name": "embedding_lookup",
        "component": "embedding",
        "phase": "baseline",
        "label": "phase-0",
        "throughput": float(tokens.numel()) / max(stats["median_ms"] / 1e3, 1e-9),
    })
    performance_recorder("embedding_lookup", stats)