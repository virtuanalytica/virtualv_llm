#!/usr/bin/env python3
"""Export only numeric scores for the eight public fixed tasks from raw runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from eval_suite import TASKS

TASK_IDS = tuple(task.id for task in TASKS)


def export_one(label: str, path: Path) -> dict:
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    ids = [row.get("task") for row in rows]
    if len(rows) != len(TASK_IDS) or set(ids) != set(TASK_IDS) or len(ids) != len(set(ids)):
        raise ValueError(f"{label}: eight public task IDs are incomplete or duplicated")
    scores = {}
    for row in rows:
        score = row.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not 0 <= score <= 1:
            raise ValueError(f"{label}: invalid numeric score")
        scores[row["task"]] = {"score": round(score, 4)}
    scores["_mean_score"] = round(sum(scores[key]["score"] for key in TASK_IDS) / len(TASK_IDS), 4)
    return {"model": label, "engine": "1Cat-vLLM TP2", "benchmarks": scores,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_kind": "numeric scores only; public fixed tasks"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True,
                        help="MODEL_ID=/path/to/eight-tasks.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for spec in args.run:
        label, separator, filename = spec.partition("=")
        if not separator or not label or not filename:
            parser.error("--run requires MODEL_ID=/path/to/eight-tasks.jsonl")
        results.append(export_one(label, Path(filename)))
    if len({row["model"] for row in results}) != len(results):
        parser.error("model IDs must be unique")
    output = {"suite": "eight-public-fixed-task-scores", "tasks": list(TASK_IDS),
              "results": results, "redaction": "scores and source hashes only; no prompts, answers or rubric details"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {len(results)} public fixed-task rows to {args.out}")


if __name__ == "__main__":
    main()
