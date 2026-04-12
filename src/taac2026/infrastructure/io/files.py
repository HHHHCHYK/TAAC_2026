from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


def stable_hash64(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    hashed = int.from_bytes(digest, byteorder="big", signed=False)
    return (hashed % ((1 << 63) - 1)) + 1


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def create_temporary_path(path: str | Path, *, suffix: str | None = None) -> Path:
    target = Path(path)
    directory = ensure_dir(target.parent)
    staged_suffix = suffix
    if staged_suffix is None:
        staged_suffix = f".tmp{target.suffix}" if target.suffix else ".tmp"
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=staged_suffix,
        dir=str(directory),
    )
    os.close(fd)
    temp_path = Path(temp_name)
    temp_path.unlink(missing_ok=True)
    return temp_path


def replace_file(path: str | Path, writer: Callable[[Path], None], *, suffix: str | None = None) -> None:
    target = Path(path)
    staged_path = create_temporary_path(target, suffix=suffix)
    try:
        writer(staged_path)
        os.replace(staged_path, target)
    except Exception:
        staged_path.unlink(missing_ok=True)
        raise


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    def _write(staged_path: Path) -> None:
        with staged_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)

    replace_file(target, _write)


__all__ = ["create_temporary_path", "ensure_dir", "replace_file", "stable_hash64", "write_json"]
