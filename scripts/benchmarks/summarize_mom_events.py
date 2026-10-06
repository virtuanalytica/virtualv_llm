#!/usr/bin/env python3
"""Summarize MoM request events without conflating system speed with model tok/s."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return round(ordered[index], 4)


def summarize(events: list[dict]) -> dict:
    durations = [float(row["wall_seconds"]) for row in events if row.get("wall_seconds") is not None]
    paths = Counter(row.get("path", "unknown") for row in events)
    tasks = Counter(row.get("task") or "unlabelled" for row in events)
    output_tokens = sum(int(row["model_output_tokens"]) for row in events
                        if isinstance(row.get("model_output_tokens"), int))
    return {"schema": "mom-system-metrics/v1", "requests": len(events), "paths": dict(paths),
            "tasks": dict(tasks), "knowledge_direct_share":
            round(paths.get("knowledge_direct", 0) / len(events), 4) if events else None,
            "request_latency_seconds_p50": percentile(durations, .5),
            "request_latency_seconds_p95": percentile(durations, .95),
            "upstream_reported_model_output_tokens_total": output_tokens,
            "model_tokens_per_second": None,
            "model_tokens_per_second_reason":
            "These events include retrieval, rules, JEV and multiple models; raw decode speed requires a separate engine run."}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("events", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    events = [json.loads(line) for line in args.events.read_text().splitlines() if line.strip()]
    report = summarize(events)
    payload = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
