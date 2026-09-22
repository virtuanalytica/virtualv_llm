#!/usr/bin/env python3
"""Create a reviewable, weight-free backup of the current VirtualV LLM suite code."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "output/pdf/VirtualV_llm_suite_v1.zip"
EXPLICIT = (
    "config/benchmark_execution_policy.json",
    "docs/LLM_BENCHMARK_OPERATIONS.md",
    "docs/MIXTURE_OF_MODELS_4_20260919.md",
    "docs/QWEN38_FLASH_NEXT_EXECUTION_PLAN.md",
    "docs/VIRTUALV_LLM_TESTSUITE_STANDARD.md",
    "infra/systemd/qwen38-flash-next-gguf-cascade.service",
    "scripts/reporting/build_virtualv_llm_suite_document.py",
    "scripts/reporting/build_virtualv_llm_suite_backup.py",
    "tests/test_mixture_of_models.py",
)


def selected_files() -> list[Path]:
    paths = list((ROOT / "scripts/benchmarks").glob("*.py"))
    paths += list((ROOT / "config/specialist_benchmarks").rglob("*"))
    paths += [ROOT / name for name in EXPLICIT]
    return sorted(
        {path for path in paths if path.is_file()},
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_bytes(archive: ZipFile, name: str, data: bytes) -> None:
    info = ZipInfo(name, date_time=(2026, 9, 22, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    files = selected_files()
    manifest = {
        "archive": args.output.name,
        "suite_version": "VirtualV LLM suite v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Executable benchmark code, controlled configuration, service unit, tests and "
            "documentation; excludes model weights, private evaluation packs, logs and secrets."
        ),
        "root_hint": "/media/knight2/EDS2/projects/numerai-signals",
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in files
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with ZipFile(temporary, "w") as archive:
        for path in files:
            write_bytes(archive, path.relative_to(ROOT).as_posix(), path.read_bytes())
        write_bytes(
            archive,
            "MANIFEST.json",
            (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode(),
        )
    temporary.replace(args.output)
    print(f"{args.output} ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
