#!/usr/bin/env python3
"""Publish aggregate private benchmark metrics without holdout items or answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ANTI_FIELDS = ("status", "n_samples", "recall_rate", "original_accuracy",
               "paraphrase_accuracy", "accuracy_gap", "accuracy", "private_pack",
               "pack_sha256")
SPECIALIST_FIELDS = ("value", "ci95", "items", "lanes", "required_lanes", "complete")
THROUGHPUT_FIELDS = ("completion_tokens", "completion_tokens_per_second", "timing_source")
ROW_FIELDS = ("model", "protocol", "status", "eight_task_mean", "gpu_board_wh",
              "gpu_board_wh_per_answer", "energy_scope", "energy_method", "requests",
              "promotion_eligible")


def redact(report: dict) -> dict:
    rows = []
    for source in report.get("results", []):
        row = {key: source[key] for key in ROW_FIELDS if key in source}
        row["specialist"] = {key: source.get("specialist", {})[key]
                             for key in SPECIALIST_FIELDS if key in source.get("specialist", {})}
        row["anti_contamination"] = {
            method: {key: result[key] for key in ANTI_FIELDS if key in result}
            for method, result in source.get("anti_contamination", {}).items()
        }
        row["throughput_probe"] = {
            key: source.get("throughput_probe", {})[key]
            for key in THROUGHPUT_FIELDS if key in source.get("throughput_probe", {})
        }
        canary = source.get("canary") or {}
        if canary:
            observations = canary.get("observations", [])
            row["canary"] = {"status": canary.get("status"),
                             "passed": sum(bool(item.get("passed")) for item in observations),
                             "total": len(observations),
                             "repetition_loops": sum(item.get("reason") == "repetition_loop"
                                                     for item in observations)}
        rows.append(row)
    return {"suite": "private-aggregates-only", "protocol": report.get("protocol"),
            "results": rows, "redaction": "allowlisted aggregate metrics; no prompts, answers, item IDs or samples"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = redact(json.loads(args.source.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {len(result['results'])} aggregate rows to {args.out}")


if __name__ == "__main__":
    main()
