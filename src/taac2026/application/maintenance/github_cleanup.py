from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from typing import Any
from urllib import error, parse, request

DEFAULT_API_BASE = "https://api.github.com"
DEFAULT_PER_PAGE = 100
DEFAULT_KEEP_ACTION_RUNS = 30
DEFAULT_KEEP_PAGES_DEPLOYMENTS = 20


@dataclass(slots=True)
class CleanupCounter:
    listed: int = 0
    targeted: int = 0
    deleted: int = 0
    failed: int = 0


@dataclass(slots=True)
class CleanupSummary:
    repo: str
    dry_run: bool
    actions: CleanupCounter
    pages: CleanupCounter


class GitHubApiClient:
    def __init__(
        self,
        *,
        repo: str,
        token: str,
        api_base: str = DEFAULT_API_BASE,
        timeout: float = 30.0,
    ) -> None:
        self.repo = repo
        self.token = token
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "taac2026-github-cleanup",
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> tuple[int, Any]:
        query = f"?{parse.urlencode(params)}" if params else ""
        url = f"{self.api_base}{path}{query}"
        payload = None if body is None else json.dumps(body).encode("utf-8")
        req = request.Request(url=url, method=method, data=payload, headers=self._headers())
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                if not raw:
                    return resp.status, None
                return resp.status, json.loads(raw.decode("utf-8"))
        except error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API {method} {path} failed: {exc.code} {message}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"GitHub API {method} {path} failed: {exc.reason}") from exc

    def list_workflow_runs(self, *, per_page: int = DEFAULT_PER_PAGE) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        page = 1
        while True:
            _, data = self._request(
                "GET",
                f"/repos/{self.repo}/actions/runs",
                params={"per_page": per_page, "page": page},
            )
            batch = data.get("workflow_runs", []) if isinstance(data, dict) else []
            if not batch:
                break
            runs.extend(batch)
            if len(batch) < per_page:
                break
            page += 1
        return runs

    def delete_workflow_run_logs(self, run_id: int) -> None:
        self._request("DELETE", f"/repos/{self.repo}/actions/runs/{run_id}/logs")

    def list_pages_deployments(self, *, per_page: int = DEFAULT_PER_PAGE) -> list[dict[str, Any]]:
        deployments: list[dict[str, Any]] = []
        page = 1
        while True:
            _, data = self._request(
                "GET",
                f"/repos/{self.repo}/deployments",
                params={
                    "environment": "github-pages",
                    "per_page": per_page,
                    "page": page,
                },
            )
            batch = data if isinstance(data, list) else []
            if not batch:
                break
            deployments.extend(batch)
            if len(batch) < per_page:
                break
            page += 1
        return deployments

    def mark_deployment_inactive(self, deployment_id: int) -> None:
        self._request(
            "POST",
            f"/repos/{self.repo}/deployments/{deployment_id}/statuses",
            body={
                "state": "inactive",
                "auto_inactive": False,
                "description": "taac2026 cleanup script marks deployment inactive before deletion",
            },
        )

    def delete_deployment(self, deployment_id: int) -> None:
        self._request("DELETE", f"/repos/{self.repo}/deployments/{deployment_id}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune GitHub Actions logs and GitHub Pages deployments")
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", ""),
        help="Repository in owner/repo format. Defaults to GITHUB_REPOSITORY env.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN", ""),
        help="GitHub token with repo/action permissions. Defaults to GITHUB_TOKEN env.",
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help="GitHub API base URL.",
    )
    parser.add_argument(
        "--keep-action-runs",
        type=int,
        default=DEFAULT_KEEP_ACTION_RUNS,
        help="Keep the newest N workflow runs and delete logs for older runs.",
    )
    parser.add_argument(
        "--keep-pages-deployments",
        type=int,
        default=DEFAULT_KEEP_PAGES_DEPLOYMENTS,
        help="Keep the newest N GitHub Pages deployments and delete older ones.",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=DEFAULT_PER_PAGE,
        help="Page size for list APIs (max 100).",
    )
    parser.add_argument(
        "--only-completed-runs",
        action="store_true",
        help="Delete logs only for workflow runs whose status is completed.",
    )
    parser.add_argument(
        "--actions-only",
        action="store_true",
        help="Only prune Actions logs.",
    )
    parser.add_argument(
        "--pages-only",
        action="store_true",
        help="Only prune GitHub Pages deployments.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete resources. Without this flag the script runs in dry-run mode.",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> str | None:
    if not args.repo or "/" not in args.repo:
        return "invalid --repo, expected owner/repo"
    if not args.token:
        return "missing --token (or GITHUB_TOKEN)"
    if args.keep_action_runs < 0:
        return "--keep-action-runs must be >= 0"
    if args.keep_pages_deployments < 0:
        return "--keep-pages-deployments must be >= 0"
    if args.per_page <= 0 or args.per_page > 100:
        return "--per-page must be between 1 and 100"
    if args.actions_only and args.pages_only:
        return "--actions-only and --pages-only cannot be used together"
    return None


def _prune_actions(
    client: GitHubApiClient,
    *,
    keep: int,
    dry_run: bool,
    per_page: int,
    only_completed_runs: bool,
) -> CleanupCounter:
    counter = CleanupCounter()
    runs = client.list_workflow_runs(per_page=per_page)
    counter.listed = len(runs)

    candidates = runs
    if only_completed_runs:
        candidates = [run for run in runs if run.get("status") == "completed"]

    targets = candidates[keep:]
    counter.targeted = len(targets)

    for run in targets:
        run_id = int(run["id"])
        run_name = run.get("name", "<unknown>")
        status = run.get("status", "<unknown>")
        if dry_run:
            print(f"[dry-run][actions] delete logs run_id={run_id} status={status} name={run_name}")
            counter.deleted += 1
            continue

        try:
            client.delete_workflow_run_logs(run_id)
            print(f"[deleted][actions] run_id={run_id} status={status} name={run_name}")
            counter.deleted += 1
        except RuntimeError as exc:
            print(f"[failed][actions] run_id={run_id}: {exc}")
            counter.failed += 1
    return counter


def _prune_pages(
    client: GitHubApiClient,
    *,
    keep: int,
    dry_run: bool,
    per_page: int,
) -> CleanupCounter:
    counter = CleanupCounter()
    deployments = client.list_pages_deployments(per_page=per_page)
    counter.listed = len(deployments)

    targets = deployments[keep:]
    counter.targeted = len(targets)

    for deployment in targets:
        deployment_id = int(deployment["id"])
        ref = deployment.get("ref", "<unknown>")
        if dry_run:
            print(f"[dry-run][pages] delete deployment_id={deployment_id} ref={ref}")
            counter.deleted += 1
            continue

        try:
            client.mark_deployment_inactive(deployment_id)
            client.delete_deployment(deployment_id)
            print(f"[deleted][pages] deployment_id={deployment_id} ref={ref}")
            counter.deleted += 1
        except RuntimeError as exc:
            print(f"[failed][pages] deployment_id={deployment_id}: {exc}")
            counter.failed += 1
    return counter


def run_cleanup(args: argparse.Namespace) -> CleanupSummary:
    dry_run = not args.execute
    client = GitHubApiClient(
        repo=args.repo,
        token=args.token,
        api_base=args.api_base,
    )

    actions_enabled = not args.pages_only
    pages_enabled = not args.actions_only

    actions_counter = CleanupCounter()
    pages_counter = CleanupCounter()
    if actions_enabled:
        actions_counter = _prune_actions(
            client,
            keep=args.keep_action_runs,
            dry_run=dry_run,
            per_page=args.per_page,
            only_completed_runs=args.only_completed_runs,
        )
    if pages_enabled:
        pages_counter = _prune_pages(
            client,
            keep=args.keep_pages_deployments,
            dry_run=dry_run,
            per_page=args.per_page,
        )

    return CleanupSummary(
        repo=args.repo,
        dry_run=dry_run,
        actions=actions_counter,
        pages=pages_counter,
    )


def _print_summary(summary: CleanupSummary) -> None:
    mode = "dry-run" if summary.dry_run else "execute"
    print(f"repo={summary.repo}")
    print(f"mode={mode}")
    print(
        "actions="
        f"listed:{summary.actions.listed},"
        f"targeted:{summary.actions.targeted},"
        f"deleted:{summary.actions.deleted},"
        f"failed:{summary.actions.failed}"
    )
    print(
        "pages="
        f"listed:{summary.pages.listed},"
        f"targeted:{summary.pages.targeted},"
        f"deleted:{summary.pages.deleted},"
        f"failed:{summary.pages.failed}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    error_message = _validate_args(args)
    if error_message:
        print(error_message)
        return 2

    summary = run_cleanup(args)
    _print_summary(summary)
    if summary.actions.failed or summary.pages.failed:
        return 1
    return 0


__all__ = [
    "CleanupCounter",
    "CleanupSummary",
    "GitHubApiClient",
    "main",
    "parse_args",
    "run_cleanup",
]


if __name__ == "__main__":
    raise SystemExit(main())
