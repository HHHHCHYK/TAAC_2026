from __future__ import annotations

from types import SimpleNamespace

from taac2026.application.maintenance.github_cleanup import (
    CleanupCounter,
    _prune_actions,
    _prune_pages,
    _validate_args,
)


class FakeClient:
    def __init__(self) -> None:
        self.runs = [
            {"id": 104, "name": "ci", "status": "completed"},
            {"id": 103, "name": "ci", "status": "in_progress"},
            {"id": 102, "name": "ci", "status": "completed"},
            {"id": 101, "name": "ci", "status": "completed"},
        ]
        self.deployments = [
            {"id": 23, "ref": "main"},
            {"id": 22, "ref": "main"},
            {"id": 21, "ref": "main"},
        ]
        self.deleted_logs: list[int] = []
        self.inactivated: list[int] = []
        self.deleted_deployments: list[int] = []

    def list_workflow_runs(self, *, per_page: int = 100):
        assert per_page == 100
        return self.runs

    def delete_workflow_run_logs(self, run_id: int) -> None:
        self.deleted_logs.append(run_id)

    def list_pages_deployments(self, *, per_page: int = 100):
        assert per_page == 100
        return self.deployments

    def mark_deployment_inactive(self, deployment_id: int) -> None:
        self.inactivated.append(deployment_id)

    def delete_deployment(self, deployment_id: int) -> None:
        self.deleted_deployments.append(deployment_id)


def test_validate_args_guards_conflicting_flags() -> None:
    args = SimpleNamespace(
        repo="Puiching-Memory/TAAC_2026",
        token="token",
        keep_action_runs=30,
        keep_pages_deployments=20,
        per_page=100,
        actions_only=True,
        pages_only=True,
    )

    message = _validate_args(args)

    assert message == "--actions-only and --pages-only cannot be used together"


def test_prune_actions_only_completed_runs_dry_run(capsys) -> None:
    client = FakeClient()

    counter = _prune_actions(
        client,
        keep=1,
        dry_run=True,
        per_page=100,
        only_completed_runs=True,
    )
    captured = capsys.readouterr()

    assert isinstance(counter, CleanupCounter)
    assert counter.listed == 4
    assert counter.targeted == 2
    assert counter.deleted == 2
    assert counter.failed == 0
    assert client.deleted_logs == []
    assert "[dry-run][actions] delete logs run_id=102" in captured.out
    assert "[dry-run][actions] delete logs run_id=101" in captured.out


def test_prune_pages_execute_marks_inactive_before_delete() -> None:
    client = FakeClient()

    counter = _prune_pages(
        client,
        keep=1,
        dry_run=False,
        per_page=100,
    )

    assert counter.listed == 3
    assert counter.targeted == 2
    assert counter.deleted == 2
    assert counter.failed == 0
    assert client.inactivated == [22, 21]
    assert client.deleted_deployments == [22, 21]
