from __future__ import annotations

import pytest
import torch

from taac2026.infrastructure.nn.triton_attention import reference_attention, triton_attention
from taac2026.infrastructure.nn.triton_ffn import reference_ffn_activation, triton_ffn_activation
from taac2026.infrastructure.nn.triton_norm import triton_rms_norm


def _supports_fp8_cuda() -> bool:
    if not torch.cuda.is_available():
        return False
    capability_major, _ = torch.cuda.get_device_capability()
    return capability_major >= 9


def _reference_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    hidden_states_fp32 = hidden_states.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    variance = hidden_states_fp32.pow(2).mean(dim=-1, keepdim=True)
    normalized = hidden_states_fp32 * torch.rsqrt(variance + eps)
    return (normalized * weight_fp32).to(hidden_states.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernel tests")
def test_triton_rms_norm_matches_reference() -> None:
    hidden_states = torch.randn(4, 16, 64, device="cuda", dtype=torch.float32)
    weight = torch.randn(64, device="cuda", dtype=torch.float32)

    expected = _reference_rms_norm(hidden_states, weight)
    actual = triton_rms_norm(hidden_states, weight)

    assert torch.allclose(actual, expected, atol=1.0e-5, rtol=1.0e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernel tests")
def test_triton_rms_norm_rejects_hidden_dims_above_kernel_limit() -> None:
    hidden_states = torch.randn(1, 1, 65537, device="cuda", dtype=torch.float32)
    weight = torch.randn(65537, device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="supports hidden_dim up to 65536"):
        triton_rms_norm(hidden_states, weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernel tests")
def test_triton_attention_matches_reference() -> None:
    query = torch.randn(2, 2, 8, 16, device="cuda", dtype=torch.float32)
    key = torch.randn(2, 2, 8, 16, device="cuda", dtype=torch.float32)
    value = torch.randn(2, 2, 8, 16, device="cuda", dtype=torch.float32)
    attention_mask = torch.tril(torch.ones(8, 8, device="cuda", dtype=torch.bool)).unsqueeze(0).expand(2, -1, -1)
    key_mask = torch.tensor(
        [
            [True, True, True, True, True, False, False, False],
            [True, True, True, True, True, True, True, False],
        ],
        device="cuda",
    )

    with torch.no_grad():
        expected = reference_attention(query, key, value, attention_mask=attention_mask, key_mask=key_mask)
        actual = triton_attention(query, key, value, attention_mask=attention_mask, key_mask=key_mask, backend="triton")

    assert torch.allclose(actual, expected, atol=1.0e-4, rtol=1.0e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernel tests")
def test_triton_ffn_activation_matches_reference() -> None:
    projected = torch.randn(4, 12, 32, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        expected = reference_ffn_activation(projected, "silu")
        actual = triton_ffn_activation(projected, "silu", backend="triton")

    assert torch.allclose(actual, expected, atol=1.0e-5, rtol=1.0e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernel tests")
def test_triton_swiglu_matches_reference() -> None:
    projected = torch.randn(3, 10, 64, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        expected = reference_ffn_activation(projected, "swiglu")
        actual = triton_ffn_activation(projected, "swiglu", backend="triton")

    assert torch.allclose(actual, expected, atol=1.0e-5, rtol=1.0e-4)


@pytest.mark.skipif(not _supports_fp8_cuda(), reason="Hopper-or-newer CUDA GPU is required for Triton fp8 kernel tests")
def test_triton_attention_fp8_matches_reference() -> None:
    query = torch.randn(2, 2, 8, 16, device="cuda", dtype=torch.float32)
    key = torch.randn(2, 2, 8, 16, device="cuda", dtype=torch.float32)
    value = torch.randn(2, 2, 8, 16, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        expected = reference_attention(query, key, value, precision="fp8-e4m3fn")
        actual = triton_attention(query, key, value, backend="triton", precision="fp8-e4m3fn")

    assert torch.allclose(actual, expected, atol=2.0e-3, rtol=2.0e-3)


@pytest.mark.skipif(not _supports_fp8_cuda(), reason="Hopper-or-newer CUDA GPU is required for Triton fp8 kernel tests")
def test_triton_attention_prequantized_fp8_matches_reference() -> None:
    query = torch.randn(2, 2, 8, 16, device="cuda", dtype=torch.float16)
    key = torch.randn(2, 2, 8, 16, device="cuda", dtype=torch.float16)
    value = torch.randn(2, 2, 8, 16, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        expected = reference_attention(query, key, value, precision="fp8-e4m3fn")
        actual = triton_attention(
            query.to(torch.float8_e4m3fn),
            key.to(torch.float8_e4m3fn),
            value.to(torch.float8_e4m3fn),
            backend="triton",
            precision="fp8-e4m3fn",
        )

    assert actual.dtype == torch.float16
    assert torch.allclose(actual, expected, atol=2.0e-3, rtol=2.0e-3)


@pytest.mark.skipif(not _supports_fp8_cuda(), reason="Hopper-or-newer CUDA GPU is required for Triton fp8 kernel tests")
def test_triton_ffn_fp8_matches_reference() -> None:
    projected = torch.randn(4, 12, 32, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        expected = reference_ffn_activation(projected, "silu", precision="fp8-e4m3fn")
        actual = triton_ffn_activation(projected, "silu", backend="triton", precision="fp8-e4m3fn")

    assert torch.allclose(actual, expected, atol=2.0e-3, rtol=2.0e-3)


@pytest.mark.skipif(not _supports_fp8_cuda(), reason="Hopper-or-newer CUDA GPU is required for Triton fp8 kernel tests")
def test_triton_ffn_prequantized_fp8_matches_reference() -> None:
    projected = torch.randn(4, 12, 32, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        expected = reference_ffn_activation(projected, "silu", precision="fp8-e4m3fn")
        actual = triton_ffn_activation(
            projected.to(torch.float8_e4m3fn),
            "silu",
            backend="triton",
            precision="fp8-e4m3fn",
        )

    assert actual.dtype == torch.float16
    assert torch.allclose(actual, expected, atol=2.0e-3, rtol=2.0e-3)