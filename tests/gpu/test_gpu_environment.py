from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


REPORT_PATH_ENV = "TAAC_GPU_ENV_REPORT_PATH"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


def test_gpu_runtime_probe_matches_expected_toolchain() -> None:
    from tests.gpu_env_support import build_gpu_env_report, write_gpu_env_report

    report = build_gpu_env_report(allow_missing_gpu=False)

    assert report["cuda_available"] is True
    assert report["cuda_device_count"] >= 1
    assert str(report["device"]).startswith("cuda")
    assert report["embedding_probe"]["shape"] == [2, 8]
    assert str(report["embedding_probe"]["device"]).startswith("cuda")
    assert report["gpu_precision"]["supported_precisions"]
    assert report["transformer_engine"]["compute_capability"] == report["gpu_precision"]["compute_capability"]

    torchao_extensions = report["torchao_extensions"]
    assert torchao_extensions is not None
    assert torchao_extensions["count"] >= 1
    assert torchao_extensions["loaded"]

    requested_report_path = os.environ.get(REPORT_PATH_ENV)
    if requested_report_path:
        write_gpu_env_report(report, Path(requested_report_path))