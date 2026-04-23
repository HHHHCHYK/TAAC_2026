from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Iterable

from loguru import logger as _logger
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from .files import ensure_dir


stdout_console = Console()
stderr_console = Console(stderr=True)
logger = _logger
logger.remove()


class _RichSink:
    def __init__(self, console: Console) -> None:
        self.console = console

    def __call__(self, message: Any) -> None:
        text = str(message).rstrip("\n")
        if not text:
            return
        self.console.print(Text.from_ansi(text), soft_wrap=True)


class RichProgressBar:
    def __init__(
        self,
        *,
        total: int | None,
        description: str,
        disable: bool = False,
        transient: bool = False,
    ) -> None:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[status]}", style="dim"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=stderr_console,
            disable=disable,
            transient=transient,
        )
        self._task_id = self._progress.add_task(description, total=total, status="")
        self._progress.start()

    def update(self, advance: int = 1) -> None:
        self._progress.advance(self._task_id, advance)

    def set_postfix(self, values: dict[str, Any], refresh: bool = False) -> None:
        status = "  ".join(f"{key}={value}" for key, value in values.items())
        self._progress.update(self._task_id, status=status, refresh=refresh)

    def close(self) -> None:
        self._progress.stop()


def configure_logging(
    *,
    level: str = "INFO",
    log_path: str | Path | None = None,
) -> None:
    logger.remove()
    logger.add(
        _RichSink(stderr_console),
        level=level,
        colorize=True,
        backtrace=False,
        diagnose=False,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )
    if log_path is not None:
        target = Path(log_path)
        ensure_dir(target.parent)
        logger.add(
            str(target),
            level=level,
            colorize=False,
            encoding="utf-8",
            backtrace=True,
            diagnose=False,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        )


def create_progress_bar(
    *,
    total: int | None,
    description: str,
    disable: bool = False,
    transient: bool = False,
) -> RichProgressBar:
    return RichProgressBar(
        total=total,
        description=description,
        disable=disable,
        transient=transient,
    )


def build_summary_table(title: str, rows: Iterable[tuple[str, Any]]) -> Table:
    table = Table(title=title, box=box.SIMPLE_HEAVY, header_style="bold cyan")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    for key, value in rows:
        table.add_row(str(key), str(value))
    return table


def print_summary_table(title: str, rows: Iterable[tuple[str, Any]]) -> None:
    stdout_console.print(build_summary_table(title, rows))


def print_panel(title: str, body: str) -> None:
    stdout_console.print(Panel.fit(body, title=title, border_style="cyan"))


def print_json(payload: Any) -> None:
    import json

    stdout_console.print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


__all__ = [
    "build_summary_table",
    "configure_logging",
    "create_progress_bar",
    "logger",
    "print_json",
    "print_panel",
    "print_summary_table",
    "stderr_console",
    "stdout_console",
]
