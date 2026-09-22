#!/usr/bin/env python3
"""Build a checksummed, weight-free raw-evidence release archive."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "reports/lm_eval_runs"
DEFAULT_OUTPUT = ROOT / "output/releases/virtualv-llm-evidence-v1.0.0.tar.gz"
SECRET_PATTERNS = (
    re.compile(rb"hf_[A-Za-z0-9]{24,}"),
    re.compile(rb"sk-[A-Za-z0-9_-]{24,}"),
    re.compile(rb"ghp_[A-Za-z0-9]{24,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{40,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan(path: Path) -> None:
    tail = b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            window = tail + block
            if any(pattern.search(window) for pattern in SECRET_PATTERNS):
                raise RuntimeError(f"credential-like value found in {path}")
            tail = window[-256:]


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    files = sorted(path for path in args.source.rglob("*") if path.is_file())
    if not files:
        raise SystemExit(f"no evidence files under {args.source}")
    entries = []
    for path in files:
        scan(path)
        entries.append({
            "path": path.relative_to(args.source).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = {
        "format": "virtualv-llm-raw-evidence-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite_commit": git_commit(),
        "source": "reports/lm_eval_runs",
        "excludes": ["model weights", "secrets", "private evaluation packs"],
        "files": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=6) as archive:
        for path in files:
            archive.add(path, arcname=f"reports/lm_eval_runs/{path.relative_to(args.source)}")
        data = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
        info = tarfile.TarInfo("MANIFEST.json")
        info.size = len(data)
        info.mtime = int(datetime.now(timezone.utc).timestamp())
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(data))
    temporary.replace(args.output)
    checksum = sha256(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{checksum}  {args.output.name}\n"
    )
    print(f"{args.output} ({len(files)} files, sha256={checksum})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
