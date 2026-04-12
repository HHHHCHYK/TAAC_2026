from __future__ import annotations

from pathlib import Path

import pytest

from tests import conftest as tests_conftest


class _FakeItem:
    def __init__(self, filename: str) -> None:
        self.fspath = Path("tests") / filename
        self.markers: list[str] = []

    def add_marker(self, marker) -> None:
        self.markers.append(marker.name)


def test_collection_marks_files_by_declared_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tests_conftest, "UNIT_TEST_FILES", {"test_unit.py"})
    monkeypatch.setattr(tests_conftest, "INTEGRATION_TEST_FILES", {"test_integration.py"})
    unit_item = _FakeItem("test_unit.py")
    integration_item = _FakeItem("test_integration.py")

    tests_conftest.pytest_collection_modifyitems(object(), [unit_item, integration_item])

    assert unit_item.markers == ["unit"]
    assert integration_item.markers == ["integration"]


def test_collection_rejects_overlapping_phase_sets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tests_conftest, "UNIT_TEST_FILES", {"test_overlap.py"})
    monkeypatch.setattr(tests_conftest, "INTEGRATION_TEST_FILES", {"test_overlap.py"})

    with pytest.raises(pytest.UsageError, match="test_overlap.py"):
        tests_conftest._build_test_file_classification()


def test_collection_rejects_unclassified_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tests_conftest, "UNIT_TEST_FILES", {"test_unit.py"})
    monkeypatch.setattr(tests_conftest, "INTEGRATION_TEST_FILES", {"test_integration.py"})

    with pytest.raises(pytest.UsageError, match="test_unknown.py"):
        tests_conftest.pytest_collection_modifyitems(object(), [_FakeItem("test_unknown.py")])