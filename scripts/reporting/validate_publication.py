#!/usr/bin/env python3
"""Fail closed when curated public evidence is malformed or host-private."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/well_known_suite_20260917.json"
PRIVATE_PATTERNS = (
    re.compile(r"/home/knight2"),
    re.compile(r"/media/knight2"),
    re.compile(r"projects/numerai-signals"),
    # Generated benchmark answers legitimately contain variable names such as
    # ``token`` and ``password``. Match credential formats, not English words.
    re.compile(r"(?:hf_[A-Za-z0-9]{24,}|sk-[A-Za-z0-9_-]{24,}|ghp_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{40,})"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
)


def main() -> int:
    payload = json.loads(REPORT.read_text())
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise SystemExit("curated report has no results")
    names: set[str] = set()
    for index, row in enumerate(results):
        if not isinstance(row, dict) or not isinstance(row.get("model"), str):
            raise SystemExit(f"result {index} has no model identifier")
        name = row["model"]
        if name in names:
            raise SystemExit(f"duplicate result model identifier: {name}")
        names.add(name)
        if "error" not in row and row.get("eval_protocol"):
            for required in ("gsm8k", "bbh", "mmlu_sample", "humaneval"):
                if required not in row:
                    raise SystemExit(f"{name}: current-protocol row lacks {required}")
    public_files = [REPORT, ROOT / "reports/dual_v100_nvlink_benchmark.html", ROOT / "README.md"]
    for path in public_files:
        text = path.read_text(errors="replace")
        for pattern in PRIVATE_PATTERNS:
            if pattern.search(text):
                raise SystemExit(f"private path or credential-like value in {path}: {pattern.pattern}")
    print(f"validated {len(results)} curated result rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
