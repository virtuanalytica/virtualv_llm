#!/usr/bin/env python3
"""Run the active private endpoint suite with subsecond GPU energy windows.

The active suite's generic sampler requires two samples *inside* each request,
which drops short 4B answers. This bridge interpolates adjacent 2s readings
for the chosen board while keeping the suite's private pack and score logic.
"""

from __future__ import annotations

import argparse
import bisect
import importlib.util
import json
import os
import sys
from pathlib import Path


def board_energy(samples: list[dict], start: float, end: float,
                 gpu_indices: set[int] | None = None) -> dict:
    if end <= start or len(samples) < 2 or not gpu_indices:
        raise ValueError("invalid energy window or missing GPU indices")
    times = [row["epoch"] for row in samples]
    if times != sorted(set(times)) or start < times[0] - 2 or end > times[-1] + 2:
        raise ValueError("power samples do not bound request within 2 seconds")
    first = max(0, bisect.bisect_right(times, start) - 1)
    last = min(len(times) - 1, bisect.bisect_left(times, end))
    if any(b - a > 6 for a, b in zip(times[first:last], times[first + 1:last + 1])):
        raise ValueError("power sample gap exceeds 6 seconds")
    points = [start] + [time for time in times[first:last + 1] if start < time < end] + [end]

    def watts(at: float, index: int) -> float:
        left = max(0, min(bisect.bisect_right(times, at) - 1, len(times) - 2))
        right = left + 1
        span = times[right] - times[left]
        if span > 6:
            raise ValueError("power interpolation gap exceeds 6 seconds")
        low = {gpu["index"]: gpu["watts"] for gpu in samples[left]["gpus"]}
        high = {gpu["index"]: gpu["watts"] for gpu in samples[right]["gpus"]}
        fraction = max(0.0, min(1.0, (at - times[left]) / span))
        return low[index] * (1 - fraction) + high[index] * fraction

    joules = {index: sum((watts(a, index) + watts(b, index)) * (b - a) / 2
                          for a, b in zip(points, points[1:])) for index in gpu_indices}
    total = sum(joules.values())
    if total <= 0:
        raise ValueError("nonpositive board energy")
    return {"status": "complete", "method": "linear interpolation of GPU board power",
            "gpu_board_joules": round(total, 3), "gpu_board_wh": round(total / 3600, 6),
            "gpu_joules_by_index": {str(index): round(value, 3) for index, value in joules.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--suite-path", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if not args.suite_path.is_file():
        parser.error("active private endpoint suite not found")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.suite_path.parent))
    spec = importlib.util.spec_from_file_location("active_private_endpoint_suite", args.suite_path)
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    global_hold = suite.HOLD.exists()
    if not global_hold:
        suite.HOLD = args.log_dir / "scoped-gpu-benchmark-lease.active"
        suite.HOLD.write_text(f"pid={os.getpid()} gpu-scoped private benchmark\n")
    suite.k3_energy_report.measure = board_energy
    run_specialists = suite.specialist_suite.run_specialists
    run_audit = suite.audit.run_contamination_audit

    def specialists(*positional, **keyword):
        keyword["out"] = args.log_dir / "specialist-results.json"
        return run_specialists(*positional, **keyword)

    def audit(*positional, **keyword):
        keyword["out"] = args.log_dir / "anti-contamination-results.json"
        return run_audit(*positional, **keyword)

    suite.specialist_suite.run_specialists = specialists
    suite.audit.run_contamination_audit = audit
    sys.argv = [str(args.suite_path), *remaining, "--log-dir", str(args.log_dir),
                "--out", str(args.log_dir / "private-result.json")]
    try:
        code = suite.main()
        report_file = args.log_dir / "private-result.json"
        report = json.loads(report_file.read_text())
        for row in report.get("results", []):
            row["benchmark_lease"] = "global" if global_hold else "scoped_single_gpu"
            row["promotion_eligible"] = False  # Comparison alone is never a model promotion.
        report_file.write_text(json.dumps(report, indent=2) + "\n")
        return code
    finally:
        if not global_hold:
            suite.HOLD.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
