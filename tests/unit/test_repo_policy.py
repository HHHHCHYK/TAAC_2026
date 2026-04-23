from __future__ import annotations

from pathlib import Path
import textwrap

from tests.repo_policy_support import format_violations, lint_repository, lint_source


def _lint_inline(source: str) -> list[tuple[Path, int, str, str]]:
    return lint_source(textwrap.dedent(source), Path("inline_example.py"))


def test_torch_policy_t001_flags_torch_zeros_without_explicit_dtype() -> None:
    violations = _lint_inline(
        """
        import torch

        def build_tensor():
            return torch.zeros(2, 3)
        """
    )

    assert [(rule_id, lineno) for _, lineno, rule_id, _ in violations] == [("T001", 5)]


def test_torch_policy_t001_allows_explicit_dtype_and_nn_parameter_initialization() -> None:
    violations = _lint_inline(
        """
        import torch
        from torch import nn

        def build_tensor():
            tensor = torch.ones(2, 3, dtype=torch.float16)
            weight = nn.Parameter(torch.zeros(4))
            return tensor, weight
        """
    )

    assert violations == []


def test_torch_policy_t002_flags_stream_without_explicit_device() -> None:
    violations = _lint_inline(
        """
        import torch

        def build_stream():
            return torch.cuda.Stream()
        """
    )

    assert [(rule_id, lineno) for _, lineno, rule_id, _ in violations] == [("T002", 5)]


def test_torch_policy_t003_flags_sync_calls_inside_forward() -> None:
    violations = _lint_inline(
        """
        class Demo:
            def forward(self, tensor):
                value = tensor.item()
                return value
        """
    )

    assert [(rule_id, lineno) for _, lineno, rule_id, _ in violations] == [("T003", 4)]


def test_repository_respects_torch_policy_rules() -> None:
    violations = lint_repository()

    assert violations == [], format_violations(violations)