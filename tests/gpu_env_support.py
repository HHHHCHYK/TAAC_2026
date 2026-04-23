from __future__ import annotations

import json
from pathlib import Path
import sys

import fbgemm_gpu
import torch
import torchao
import torchrec
import triton

from taac2026.domain.features import FeatureSchema, FeatureTableSpec
from taac2026.infrastructure.io.sparse_collate import keyed_jagged_from_masked_batches
from taac2026.infrastructure.nn.embedding import TorchRecEmbeddingBagAdapter
from taac2026.infrastructure.nn.gpu_capability import detect_precision_support
from taac2026.infrastructure.nn.te_backend import detect_transformer_engine_availability


def run_embedding_probe(device: torch.device) -> dict[str, object]:
    schema = FeatureSchema(
        tables=(
            FeatureTableSpec(
                name="probe_tokens",
                family="probe",
                num_embeddings=64,
                embedding_dim=8,
            ),
        ),
        dense_dim=0,
    )
    features = keyed_jagged_from_masked_batches(
        {
            "probe_tokens": (
                torch.tensor([[1, 2, 0], [3, 0, 0]], dtype=torch.long),
                torch.tensor([[True, True, False], [True, False, False]], dtype=torch.bool),
            ),
        }
    ).to(device)

    adapter = TorchRecEmbeddingBagAdapter(schema, table_names=("probe_tokens",), device=device)
    output = adapter(features)
    if output.shape != (2, 8):
        raise RuntimeError(f"Unexpected TorchRec probe output shape: {tuple(output.shape)}")
    if not torch.isfinite(output).all().item():
        raise RuntimeError("TorchRec probe produced non-finite values")

    return {
        "shape": list(output.shape),
        "device": str(output.device),
        "dtype": str(output.dtype),
    }


def run_torchao_extension_probe() -> dict[str, object]:
    extension_paths = sorted(Path(torchao.__file__).resolve().parent.glob("_C*.so"))
    if not extension_paths:
        raise RuntimeError("torchao package does not expose any native extensions")

    loaded: list[str] = []
    for extension_path in extension_paths:
        try:
            torch.ops.load_library(str(extension_path))
        except OSError as exc:
            raise RuntimeError(f"Failed to load torchao native extension {extension_path.name}: {exc}") from exc
        loaded.append(extension_path.name)

    return {
        "count": len(loaded),
        "loaded": loaded,
    }


def build_gpu_env_report(*, allow_missing_gpu: bool) -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    if not cuda_available and not allow_missing_gpu:
        raise RuntimeError("CUDA device not available")

    device = torch.device("cuda:0" if cuda_available else "cpu")
    embedding_probe = run_embedding_probe(device)

    if cuda_available:
        sample = torch.randn(32, 32, device=device)
        _ = sample @ sample.transpose(0, 1)
        torch.cuda.synchronize(device)

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchao": getattr(torchao, "__version__", "unknown"),
        "torchrec": getattr(torchrec, "__version__", "unknown"),
        "fbgemm_gpu": getattr(fbgemm_gpu, "__version__", "unknown"),
        "triton": getattr(triton, "__version__", "unknown"),
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "device": str(device),
        "gpu_precision": detect_precision_support(device if cuda_available else None),
        "transformer_engine": detect_transformer_engine_availability(device if cuda_available else None),
        "embedding_probe": embedding_probe,
        "torchao_extensions": run_torchao_extension_probe() if cuda_available else None,
    }


def write_gpu_env_report(report: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")