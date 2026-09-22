#!/usr/bin/env python3
"""Exhaustively rank post-hoc model mixtures from existing sample logs.

This performs no inference. It caches each model/task sample file once, then
tests every fixed-size combination with the same policies as
``mixture_of_models.py``. Member order is fixed by descending solo composite,
which also makes even-vote tie-breaking explicit and reproducible.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import mixture_of_models as mixture
import rank_models
from well_known_suite import BBH_SUBTASKS, EVAL_PROTOCOL, MMLU_SUBJECT_SAMPLE


def solo_composite(row: dict[str, Any]) -> float | None:
    values = (
        rank_models.gsm8k_score(row),
        (row.get("humaneval") or {}).get("pass_at_1"),
        (row.get("mmlu_sample") or {}).get("mean_accuracy"),
        (row.get("bbh") or {}).get("mean_accuracy"),
    )
    return sum(values) / len(values) \
        if all(isinstance(value, (int, float)) for value in values) else None


def score_members(members: tuple[str, ...], report: dict[str, Any]):
    gsm8k = mixture.score_final_answer_task(list(members), "gsm8k")["accuracy"]
    humaneval = mixture.score_humaneval_mixture(
        list(members), report)["pass_at_1"]
    mmlu = mixture.score_mmlu_mixture(
        list(members), MMLU_SUBJECT_SAMPLE)["mean_accuracy"]
    bbh = mixture.score_bbh_mixture(
        list(members), BBH_SUBTASKS)["mean_accuracy"]
    values = (gsm8k, humaneval, mmlu, bbh)
    composite = sum(values) / len(values) \
        if all(value is not None for value in values) else None
    return {
        "members": list(members),
        "core_composite_4task": round(composite, 6) if composite is not None else None,
        "gsm8k": gsm8k,
        "humaneval": humaneval,
        "mmlu": mmlu,
        "bbh": bbh,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=mixture.DEFAULT_OUT)
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--require-member", action="append", default=[])
    args = parser.parse_args()
    if args.size < 2:
        raise SystemExit("--size must be at least 2")
    if args.top < 1:
        raise SystemExit("--top must be positive")

    report = mixture.load_report(args.report)
    rows = {
        row["model"]: row for row in report.get("results", [])
        if row.get("model")
        and not row["model"].startswith("mixture-of-models")
        and row.get("eval_protocol") == EVAL_PROTOCOL
        and (mixture.REPORTS / "lm_eval_runs" / row["model"]).is_dir()
        and solo_composite(row) is not None
    }
    missing_required = [model for model in args.require_member if model not in rows]
    if missing_required:
        raise SystemExit(f"required members are not eligible: {missing_required}")
    if args.size > len(rows):
        raise SystemExit(f"--size {args.size} exceeds {len(rows)} eligible models")

    # Strongest solo model comes first and therefore wins exact vote ties.
    models = sorted(rows, key=lambda model: solo_composite(rows[model]), reverse=True)
    tasks = [
        "gsm8k",
        *(f"mmlu_{subject}_generative" for subject in MMLU_SUBJECT_SAMPLE),
        *BBH_SUBTASKS,
    ]
    original_load = mixture.load_samples
    cache = {(model, task): original_load(model, task)
             for model in models for task in tasks}
    mixture.load_samples = lambda model, task: cache[(model, task)]

    required = set(args.require_member)
    candidates = (members for members in combinations(models, args.size)
                  if required.issubset(members))
    scored = [score_members(members, report) for members in candidates]
    scored.sort(key=lambda row: row["core_composite_4task"], reverse=True)
    output = {
        "eval_protocol": EVAL_PROTOCOL,
        "mixture_size": args.size,
        "eligible_models": len(models),
        "combinations_scored": len(scored),
        "required_members": args.require_member,
        "tie_break_order": models,
        "results": scored[:args.top],
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
