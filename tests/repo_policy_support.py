from __future__ import annotations

import ast
from pathlib import Path


SCAN_ROOTS = (
    Path("config"),
    Path("src/taac2026"),
)

Violation = tuple[Path, int, str, str]


class TorchPolicyVisitor(ast.NodeVisitor):
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.violations: list[Violation] = []
        self._in_forward = False
        self._nn_parameter_nodes: set[int] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        was_in_forward = self._in_forward
        if node.name == "forward":
            self._in_forward = True
        self.generic_visit(node)
        self._in_forward = was_in_forward

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "Parameter"
            and isinstance(func.value, ast.Name)
            and func.value.id == "nn"
        ):
            for argument in node.args:
                self._nn_parameter_nodes.add(id(argument))

        self._check_t001(node)
        self._check_t002(node)
        self._check_t003(node)
        self.generic_visit(node)

    def _check_t001(self, node: ast.Call) -> None:
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "torch"
            and func.attr in ("zeros", "ones")
        ):
            return

        if any(keyword.arg == "dtype" for keyword in node.keywords):
            return
        if id(node) in self._nn_parameter_nodes:
            return

        self.violations.append((
            self.filepath,
            node.lineno,
            "T001",
            f"torch.{func.attr}() without explicit dtype - prefer tensor.new_{func.attr}() for AMP safety",
        ))

    def _check_t002(self, node: ast.Call) -> None:
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "Stream"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "cuda"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "torch"
        ):
            return

        if node.args:
            return
        if any(keyword.arg == "device" for keyword in node.keywords):
            return

        self.violations.append((
            self.filepath,
            node.lineno,
            "T002",
            "torch.cuda.Stream() without device= - stream may bind to wrong GPU",
        ))

    def _check_t003(self, node: ast.Call) -> None:
        if not self._in_forward:
            return

        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr in ("item", "cpu", "numpy")
        ):
            return

        self.violations.append((
            self.filepath,
            node.lineno,
            "T003",
            f".{func.attr}() inside forward() forces GPU-to-CPU sync every call",
        ))


def lint_source(source: str, filepath: Path) -> list[Violation]:
    tree = ast.parse(source, filename=str(filepath))
    visitor = TorchPolicyVisitor(filepath)
    visitor.visit(tree)
    return visitor.violations


def lint_file(filepath: Path) -> list[Violation]:
    try:
        source = filepath.read_text(encoding="utf-8")
    except OSError:
        return []

    try:
        return lint_source(source, filepath)
    except SyntaxError:
        return []


def lint_repository(scan_roots: tuple[Path, ...] = SCAN_ROOTS) -> list[Violation]:
    files: list[Path] = []
    for root in scan_roots:
        if root.exists():
            files.extend(root.rglob("*.py"))

    violations: list[Violation] = []
    for filepath in sorted(files):
        violations.extend(lint_file(filepath))
    return violations


def format_violations(violations: list[Violation]) -> str:
    return "\n".join(
        f"{filepath}:{lineno}: {rule_id} {message}"
        for filepath, lineno, rule_id, message in violations
    )