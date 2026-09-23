"""Locked, atomic updates for shared benchmark JSON artifacts.

The lock belongs to the artifact, not to one runner.  Readers see either the
old complete JSON or the new complete JSON; concurrent writers merge against
the latest version while holding the same lock.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator


@contextmanager
def locked_report(path: Path, default: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        payload = json.loads(path.read_text()) if path.exists() else (default or {"results": []})
        yield payload
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(payload, output, indent=2, ensure_ascii=False)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_name, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def upsert_result(path: Path, row: dict[str, Any],
                  default: dict[str, Any] | None = None,
                  key: Callable[[dict[str, Any]], object] | None = None) -> None:
    identify = key or (lambda item: item.get("model"))
    identity = identify(row)
    if identity is None:
        raise ValueError("benchmark row has no identity")
    with locked_report(path, default) as payload:
        rows = payload.setdefault("results", [])
        if not isinstance(rows, list):
            raise ValueError(f"{path}: results is not a list")
        payload["results"] = [old for old in rows if identify(old) != identity] + [row]


def patch_result(path: Path, model: str, fields: dict[str, Any]) -> bool:
    found = False
    with locked_report(path) as payload:
        for row in payload.get("results", []):
            if row.get("model") == model:
                row.update(fields)
                found = True
                break
    return found
