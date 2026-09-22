#!/usr/bin/env python3
"""Fetch small public benchmark datasets with pinned content hashes."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]
HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
HUMANEVAL_SHA256 = "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
HUMANEVAL_PATH = ROOT / "data/eval_cache/HumanEval.jsonl.gz"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str, destination: Path, expected_sha256: str) -> None:
    if destination.is_file() and digest(destination) == expected_sha256:
        print(f"verified {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        with urlopen(url, timeout=60) as response:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    actual = digest(temporary)
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"hash mismatch for {url}: {actual}")
    temporary.replace(destination)
    print(f"installed {destination}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        if not HUMANEVAL_PATH.is_file() or digest(HUMANEVAL_PATH) != HUMANEVAL_SHA256:
            raise SystemExit(f"missing or invalid {HUMANEVAL_PATH}")
        print(f"verified {HUMANEVAL_PATH}")
        return 0
    fetch(HUMANEVAL_URL, HUMANEVAL_PATH, HUMANEVAL_SHA256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
