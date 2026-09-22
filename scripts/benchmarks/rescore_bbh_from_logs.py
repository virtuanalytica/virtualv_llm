#!/usr/bin/env python3
"""Atomically repair BBH scores from already persisted lm-eval sample logs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import well_known_suite as wks

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/well_known_suite_20260917.json"
RUNS = ROOT / "reports/lm_eval_runs"
CASCADE_STATE = ROOT / "reports/remaining_large_model_cascade_20260921.json"


def rescore_model(model: str) -> dict[str, Any] | None:
    root = RUNS / model
    per_subtask: dict[str, float] = {}
    correct = 0.0
    total = 0
    for task in wks.BBH_SUBTASKS:
        result = wks.score_bbh_logged_samples(root / task, task)
        if result is None or result.get("n") != wks.BBH_LIMIT or result.get("accuracy") is None:
            return None
        per_subtask[task] = result["accuracy"]
        correct += result["accuracy"] * result["n"]
        total += result["n"]
    return {
        "per_subtask": per_subtask,
        "mean_accuracy": round(correct / total, 4),
        "n_samples": total,
        "scorer": "normalized_logged_answer_v1",
    }


def main() -> int:
    payload = json.loads(REPORT.read_text())
    changed = []
    for row in payload.get("results", []):
        model = row.get("model")
        if not model or row.get("error"):
            continue
        replacement = rescore_model(model)
        if replacement is None:
            continue
        old_bbh = row.get("bbh") or {}
        previous = old_bbh.get(
            "legacy_mean_accuracy_before_normalization", old_bbh.get("mean_accuracy")
        )
        replacement["legacy_mean_accuracy_before_normalization"] = previous
        row["bbh"] = replacement
        changed.append({"model": model, "before": previous, "after": replacement["mean_accuracy"]})
    payload["bbh_scoring_fix"] = {
        "method": "normalized_logged_answer_v1",
        "reason": "strip punctuation/model end tokens before comparing typed BBH answers",
        "changed_rows": changed,
    }
    temporary = REPORT.with_suffix(REPORT.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(REPORT)
    if CASCADE_STATE.exists():
        state = json.loads(CASCADE_STATE.read_text())
        candidates = {candidate.get("name") for candidate in state.get("candidates", [])}
        state["events"] = [event for event in state.get("events", [])
                           if event.get("event") != "bbh_rescored_normalized_v1"]
        rows = {row.get("model"): row for row in payload.get("results", [])}
        for item in changed:
            if item["model"] not in candidates:
                continue
            row = rows[item["model"]]
            values = [
                max(value for key, value in (row.get("gsm8k") or {}).items()
                    if key.startswith("exact_match") and "stderr" not in key),
                row["bbh"]["mean_accuracy"], row["mmlu_sample"]["mean_accuracy"],
                row["humaneval"]["pass_at_1"],
            ]
            state["events"].append({
                "model": item["model"], "event": "bbh_rescored_normalized_v1",
                "bbh_before": item["before"], "bbh_after": item["after"],
                "corrected_composite": sum(values) / len(values),
            })
        winner = state.get("winner") or state.get("current_winner")
        if winner in rows:
            row = rows[winner]
            state["winner_composite"] = sum([
                max(value for key, value in row["gsm8k"].items()
                    if key.startswith("exact_match") and "stderr" not in key),
                row["bbh"]["mean_accuracy"], row["mmlu_sample"]["mean_accuracy"],
                row["humaneval"]["pass_at_1"],
            ]) / 4
            state["current_winner_composite"] = state["winner_composite"]
        temporary = CASCADE_STATE.with_suffix(CASCADE_STATE.suffix + ".tmp")
        temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(CASCADE_STATE)
    print(json.dumps(changed, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
