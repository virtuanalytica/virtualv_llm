#!/usr/bin/env python3
"""Throughput of cloud models as a distribution, measured with `omp bench`.

One completion is a point value; a hosted endpoint varies from request to
request. This records the median and 95th percentile of time to first token
and decode rate over repeated requests, and puts the median decode rate on the
model's row so the dashboard's t/s column is not a single lucky or unlucky run.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_external_provider_cascade as cascade  # noqa: E402
from result_store import locked_report, patch_result  # noqa: E402

OUT = ROOT / "reports/cloud_cli_throughput.json"
KEPT = ("min", "p50", "p95", "max")


def summarise(bench: dict, selector: str) -> dict:
    """Reduce one `omp bench --json` payload to the figures we publish."""
    model = next(m for m in bench["models"] if m["model"] == selector)
    ok = [r for r in model["results"] if r.get("ok")]
    if not ok:
        raise RuntimeError(f"{selector}: no successful bench request")
    stats = model["stats"]
    return {
        "selector": selector, "profile": bench.get("profile"), "requests": len(ok),
        "failed_requests": len(model["results"]) - len(ok),
        "decode_tokens_per_second": {k: round(stats["generationTps"][k], 2) for k in KEPT},
        "end_to_end_tokens_per_second": {k: round(stats["tokensPerSecond"][k], 2) for k in KEPT},
        "time_to_first_token_ms": {k: round(stats["ttftMs"][k]) for k in KEPT},
        "mean_output_tokens": stats["outputTokens"],
    }


def measure(selector: str, runs: int) -> dict:
    with tempfile.TemporaryDirectory() as cwd:
        proc = subprocess.run(["omp", "bench", selector, "--profile", "chat", "--runs", str(runs), "--json"],
                              text=True, capture_output=True, stdin=subprocess.DEVNULL, cwd=cwd, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f"omp bench {selector} failed (rc={proc.returncode}): {proc.stderr[-300:]}")
    return summarise(json.loads(proc.stdout), selector)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", action="append", help="omp provider key from the cascade; default: all")
    parser.add_argument("--model", action="append", help="only these model slugs")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--row-prefix", default="cloud")
    args = parser.parse_args()
    version = subprocess.run(["omp", "--version"], text=True, capture_output=True).stdout.strip()
    failures = 0
    for key, provider in sorted(cascade.PROVIDERS.items()):
        if provider.get("backend") != "omp" or (args.provider and key not in args.provider):
            continue
        for model in provider["models"]:
            if args.model and model not in args.model:
                continue
            row_name = f"{args.row_prefix}-{key}-{model}"
            try:
                summary = measure(f"{provider['omp_provider']}/{model}", args.runs)
            except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
                failures += 1
                print(f"FAILED {row_name}: {exc}", flush=True)
                continue
            summary.update(measured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), omp_version=version)
            with locked_report(OUT, {"suite": "cloud CLI throughput (omp bench, chat profile)", "results": []}) as payload:
                payload["results"] = [r for r in payload["results"] if r.get("model") != row_name]
                payload["results"].append({"model": row_name, **summary})
            # Only an existing quality row is annotated; throughput never creates a row.
            patch_result(cascade.REPORT, row_name, {
                "completion_tokens_per_second": summary["decode_tokens_per_second"]["p50"],
                "throughput_source": f"omp bench chat, median of {summary['requests']} requests "
                                     f"(p95 {summary['decode_tokens_per_second']['p95']}); see {OUT.name}",
            })
            print(f"{row_name}: decode p50 {summary['decode_tokens_per_second']['p50']} t/s, "
                  f"TTFT p50 {summary['time_to_first_token_ms']['p50']} ms", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
