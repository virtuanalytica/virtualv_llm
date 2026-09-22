#!/usr/bin/env python3
"""Cron entry point: run exactly one not-yet-benchmarked local model, then exit.

Picks the first model (from benchmark_local_gguf_tp2.MODELS, restricted to the
8 models already on disk -- no Hugging Face downloads here) that has no result
tagged with well_known_suite.EVAL_PROTOCOL yet, and runs the suite for it.
Safe to invoke every cron tick: once every local model has a current-protocol
result, this becomes a no-op (exit 0, nothing printed but a status line).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_local_gguf_tp2 import MODELS  # noqa: E402
import well_known_suite as wks  # noqa: E402

# Only models already downloaded to local disk -- no unattended HF acquisitions.
LOCAL_ONLY = [
    "qwen38-27b-q4", "deepseek-r1-qwen32b-q4", "qwen35-27b-q4",
    "gemma4-26b-a4b-q4", "devstral-small2-24b-q4", "qwen25-72b-q4",
    "llama31-70b-instruct-q4",
]
# qwen36-27b-iq3 is excluded: it is the live production model on the A4000
# (port 11435) and must not be pulled onto the V100 pair by this sweep.


def next_model() -> str | None:
    report = ROOT / "reports/well_known_suite_20260917.json"
    done = set()
    if report.exists():
        payload = json.loads(report.read_text())
        for row in payload.get("results", []):
            if row.get("eval_protocol") == wks.EVAL_PROTOCOL and "error" not in row:
                done.add(row.get("model"))
    for name in LOCAL_ONLY:
        if name in MODELS and name not in done:
            return name
    return None


def main() -> int:
    name = next_model()
    if name is None:
        print("run_next_benchmark: all local models already have a current-protocol result")
        return 0
    print(f"run_next_benchmark: running {name}", flush=True)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/benchmarks/well_known_suite.py"), name],
        cwd=str(ROOT),
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
