#!/usr/bin/env python3
"""Post-hoc "mixture of models" ensemble over already-benchmarked models.

No new GPU time is spent here: each model was already queried sequentially
by well_known_suite.py (one loaded at a time -- the 24-72B-class models here
don't fit together in the V100s' 64GB combined VRAM, so a literal concurrent
ensemble isn't possible; a sequential-query ensemble is the practical
equivalent and was the explicitly chosen approach). This script re-reads the
per-question logs those runs already wrote (--log_samples for every lm-eval
task, per_task pass/code for HumanEval) and combines them per question:

- MMLU / GSM8K / BBH (final-answer tasks): majority vote of each
  model's own filtered/extracted answer for the same question, scored against
  the same target. Ties broken by the first model in `--models` order (kept
  deterministic, not random).
- HumanEval: "solved if at least one ensemble model's completion passes" --
  a standard best-of-N-models policy; majority vote on raw code text doesn't
  make sense since two correct implementations are rarely textually equal.
- TruthfulQA generation: optionally route to one member with
  ``--truthfulqa-specialist``. Its free-text BLEU/ROUGE evaluation cannot be
  reconstructed by majority-voting answer strings.

``--vote-members`` may select an odd subset for discrete voting while keeping
all declared members available to HumanEval and specialist routes. This avoids
forcing a fourth specialist into unrelated votes merely to call the system a
four-model mixture, and records the routing policy explicitly in the result.

Usage: after well_known_suite.py has been run for every model in --models,
  python3 scripts/benchmarks/mixture_of_models.py qwen38-27b-q4 \
      deepseek-r1-qwen32b-q4 qwen25-72b-q4 llama31-70b-instruct-q4
  python3 scripts/benchmarks/mixture_of_models.py \
      kat-coder-v2.5-dev granite-4.2-30b qwen38-27b-q4 gemma4-26b-a4b-q4 \
      --name mixture-of-models-4-gemma-routed \
      --vote-members kat-coder-v2.5-dev granite-4.2-30b qwen38-27b-q4 \
      --truthfulqa-specialist gemma4-26b-a4b-q4
Writes a "mixture-of-models" entry into the same well_known_suite report
(merged the same way well_known_suite.py merges per-model entries).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "well_known_suite_20260917.json"

FINAL_ANSWER_TASKS = {
    "gsm8k": ["gsm8k"],
    "truthfulqa_gen": ["truthfulqa_gen"],
}


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {"results": []}


_GSM8K_ANSWER_RE = __import__("re").compile(r"####\s*(-?[\d,.]+)")


def load_samples(model: str, task: str) -> dict[int, dict[str, Any]]:
    """Load one model's per-question (target, extracted-answer) pairs for a task.

    2026-09-18 bug found via a mixture-of-models run scoring an impossible
    gsm8k=0.0/truthfulqa=0.0 despite every ensemble member individually
    scoring 0.4-0.9+ on both: (1) gsm8k's samples file logs TWO filter passes
    per question (strict-match, flexible-extract), and this function used to
    key rows by line-enumeration index rather than doc_id -- with 50 real
    questions and 2 filters each, that silently split into 100 "different"
    fake questions and mixed the two filters' answers together across models.
    (2) gsm8k's raw `target` field is the full worked-solution text ending in
    "#### N", not the bare number `filtered_resps` extracts -- the equality
    check in score_final_answer_task() could never match. Fixed by keying on
    the real doc_id (de-duplicating filter passes, preferring flexible-extract
    since that's the more lenient/accurate extractor per rank_models.py's own
    gsm8k_score() comment), and by extracting the bare "#### N" number from
    the target text for gsm8k specifically so it's comparable to the answer.
    """
    out_dir = REPORTS / "lm_eval_runs" / model / task
    files = sorted(out_dir.glob("*/samples_*.jsonl"))
    if not files:
        return {}
    rows: dict[int, dict[str, Any]] = {}
    for line in files[-1].read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        doc_id = row.get("doc_id")
        if doc_id is None:
            continue
        # gsm8k logs one row per filter pass for the same doc_id; keep the
        # flexible-extract row when both are present instead of whichever
        # happened to be read last.
        if doc_id in rows and row.get("filter") == "strict-match":
            continue
        filtered = row.get("filtered_resps")
        answer = filtered[0] if filtered else None
        target = row.get("target")
        if task == "gsm8k" and isinstance(target, str):
            m = _GSM8K_ANSWER_RE.search(target)
            if m:
                target = m.group(1).replace(",", "")
        rows[doc_id] = {"target": target, "answer": answer}
    return rows


def majority_vote(answers: list[Any]) -> Any:
    present = [a for a in answers if a is not None]
    if not present:
        return None
    counts = Counter(present)
    top = counts.most_common()
    best_count = top[0][1]
    tied = [a for a, c in top if c == best_count]
    # deterministic tie-break: first tied value in the original (model-order) list
    for a in answers:
        if a in tied:
            return a
    return tied[0]


def score_final_answer_task(models: list[str], task: str) -> dict[str, Any]:
    per_model = {m: load_samples(m, task) for m in models}
    doc_ids = sorted(set().union(*[set(rows) for rows in per_model.values()])) if per_model else []
    correct = 0
    n = 0
    for doc_id in doc_ids:
        answers = [per_model[m].get(doc_id, {}).get("answer") for m in models]
        targets = [per_model[m].get(doc_id, {}).get("target") for m in models if doc_id in per_model[m]]
        if not targets:
            continue
        target = targets[0]
        mixture_answer = majority_vote(answers)
        n += 1
        correct += int(mixture_answer == target)
    return {"task": task, "n": n, "accuracy": round(correct / n, 4) if n else None}


def score_bbh_mixture(models: list[str], bbh_subtasks: list[str]) -> dict[str, Any]:
    per_subtask = {}
    correct_total = 0
    n_total = 0
    for subtask in bbh_subtasks:
        result = score_final_answer_task(models, subtask)
        per_subtask[subtask] = result["accuracy"]
        if result["accuracy"] is not None:
            correct_total += result["accuracy"] * result["n"]
            n_total += result["n"]
    return {"per_subtask": per_subtask, "mean_accuracy": round(correct_total / n_total, 4) if n_total else None,
            "n_samples": n_total}


def score_mmlu_mixture(models: list[str], subjects: list[str]) -> dict[str, Any]:
    per_subject = {}
    correct_total = 0
    n_total = 0
    for subject in subjects:
        task = f"mmlu_{subject}_generative"
        result = score_final_answer_task(models, task)
        per_subject[subject] = result["accuracy"]
        if result["accuracy"] is not None:
            correct_total += result["accuracy"] * result["n"]
            n_total += result["n"]
    return {"per_subject": per_subject, "mean_accuracy": round(correct_total / n_total, 4) if n_total else None,
            "n_samples": n_total}


def score_humaneval_mixture(models: list[str], report: dict[str, Any]) -> dict[str, Any]:
    by_model = {r["model"]: r for r in report.get("results", []) if r.get("model") in models}
    per_task_by_model = {m: (by_model.get(m, {}).get("humaneval") or {}).get("per_task", {}) for m in models}
    all_task_ids = sorted(set().union(*[set(d) for d in per_task_by_model.values()])) if per_task_by_model else []
    solved = 0
    for task_id in all_task_ids:
        if any(per_task_by_model[m].get(task_id, {}).get("passed") for m in models):
            solved += 1
    total = len(all_task_ids)
    return {"n_problems": total, "n_passed": solved, "pass_at_1": round(solved / total, 4) if total else None,
            "policy": "solved if >=1 ensemble model's completion passes (best-of-N-models)"}


def truthfulqa_accuracy(metrics: dict[str, Any]) -> float | None:
    """Mean of TruthfulQA's four BLEU/ROUGE accuracy indicators."""
    values = [value for key, value in metrics.items()
              if key.endswith("_acc,none") and isinstance(value, (int, float))]
    return sum(values) / len(values) if values else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="model names already benchmarked by well_known_suite.py")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--name", default="mixture-of-models",
                    help="result-row name (allows several mixtures to coexist)")
    ap.add_argument("--vote-members", nargs="+", default=None,
                    help="subset of ensemble members used for GSM8K/MMLU/BBH voting")
    ap.add_argument("--truthfulqa-specialist", default=None,
                    help="ensemble member whose measured free-text TruthfulQA result is routed through")
    args = ap.parse_args()

    # Preserve caller order because it is the deterministic tie-break order.
    args.models = list(dict.fromkeys(args.models))
    vote_models = list(dict.fromkeys(args.vote_members or args.models))
    if not args.name.strip():
        raise SystemExit("--name must not be empty")
    if not vote_models:
        raise SystemExit("--vote-members must not be empty")
    unknown_voters = [model for model in vote_models if model not in args.models]
    if unknown_voters:
        raise SystemExit(f"--vote-members are not ensemble members: {unknown_voters}")
    if args.truthfulqa_specialist and args.truthfulqa_specialist not in args.models:
        raise SystemExit("--truthfulqa-specialist must also be one of the ensemble members")

    report = load_report(args.out)
    tested = {r.get("model") for r in report.get("results", [])}
    missing = [m for m in args.models if m not in tested]
    if missing:
        raise SystemExit(f"these models have no well_known_suite.py entry yet in {args.out}: {missing}")

    from well_known_suite import BBH_SUBTASKS, EVAL_PROTOCOL, MMLU_SUBJECT_SAMPLE  # noqa: E402

    gsm8k = score_final_answer_task(vote_models, "gsm8k")
    bbh = score_bbh_mixture(vote_models, BBH_SUBTASKS)
    mmlu = score_mmlu_mixture(vote_models, MMLU_SUBJECT_SAMPLE)
    humaneval = score_humaneval_mixture(args.models, report)

    truthfulqa: dict[str, Any]
    if args.truthfulqa_specialist:
        specialist_row = next(r for r in report.get("results", [])
                              if r.get("model") == args.truthfulqa_specialist)
        source_metrics = specialist_row.get("truthfulqa_gen") or {}
        if truthfulqa_accuracy(source_metrics) is None:
            raise SystemExit(
                f"TruthfulQA metrics missing for specialist {args.truthfulqa_specialist}")
        truthfulqa = dict(source_metrics)
        truthfulqa.update({
            "policy": "task-routed to one measured free-text specialist",
            "specialist": args.truthfulqa_specialist,
        })
    else:
        truthfulqa = {
            "note": "not majority-voted (free-text/BLEU-ROUGE task, no discrete target)"
        }

    # The mixture inherits eval_protocol from its members: it's the same
    # underlying per-model generations, just recombined, so it's only
    # comparable in rank_models.py's ranking when every member was itself
    # measured under the CURRENT protocol (mixing a stale-protocol member in
    # would silently blend results scored under different generation-config
    # fixes).
    member_protocols = {r.get("model"): r.get("eval_protocol") for r in report.get("results", []) if r.get("model") in args.models}
    mixed_protocol = EVAL_PROTOCOL if all(p == EVAL_PROTOCOL for p in member_protocols.values()) else None

    routed = vote_models != args.models or args.truthfulqa_specialist is not None
    core_values = (gsm8k["accuracy"], humaneval["pass_at_1"],
                   mmlu["mean_accuracy"], bbh["mean_accuracy"])
    core_composite = sum(core_values) / len(core_values) \
        if all(value is not None for value in core_values) else None
    tqa_accuracy = truthfulqa_accuracy(truthfulqa)
    diagnostic_five_task_mean = (sum(core_values) + tqa_accuracy) / 5.0 \
        if core_composite is not None and tqa_accuracy is not None else None

    result = {
        "model": args.name, "ensemble_members": args.models,
        # 2026-09-19 (user feedback): naming the members isn't enough to make this
        # reproducible -- HOW they're combined matters too. Discrete tasks use
        # equal-weight majority vote over vote_models; HumanEval uses best-of-N
        # because two correct implementations are rarely textually identical.
        # ensemble_weights is explicitly None: no numeric weighting is applied.
        "ensemble_method": "task_router" if routed else "majority_vote",
        "ensemble_weights": None,
        "routing": {
            "discrete_vote_members": vote_models,
            "humaneval_best_of_members": args.models,
            "truthfulqa_specialist": args.truthfulqa_specialist,
        },
        "eval_protocol": mixed_protocol,
        "engine": f"mixture({len(args.models)})",
        "policy": ("sequential task-routed ensemble; majority vote on discrete final answers, "
                   "best-of-N for HumanEval, optional specialist for free-text TruthfulQA"),
        "evaluation_caveat": (
            "Post-hoc recombination of existing benchmark samples. When members are selected "
            "using these same scores, the result is a selection-set estimate and requires a "
            "fresh held-out run before production claims."),
        "benchmark_summary": {
            "core_composite_4task": round(core_composite, 6) if core_composite is not None else None,
            # Diagnostic only: rank_models.py deliberately keeps TruthfulQA
            # outside its official four-task composite because its generation
            # metrics have different semantics. Recording this mean still makes
            # specialist experiments easy to compare without relabelling it as
            # the official leaderboard score.
            "diagnostic_mean_with_truthfulqa": (
                round(diagnostic_five_task_mean, 6)
                if diagnostic_five_task_mean is not None else None),
        },
        "gsm8k": {"exact_match,flexible-extract": gsm8k["accuracy"], "n": gsm8k["n"]},
        # truthfulqa_gen is NOT majority-voted: it's a free-text generation task scored
        # by BLEU/ROUGE similarity to reference answers, not a discrete final-answer
        # task -- its raw `target` field is blank (lm-eval scores against separate
        # correct/incorrect reference lists, not this field), so an exact-match
        # majority vote against it is meaningless (found 2026-09-18: previously
        # produced a bogus 0.0 for every ensemble regardless of member quality).
        # Not part of the official composite (see rank_models.py). Without a
        # specialist it is omitted rather than reported as a fake number; with
        # a specialist its original measured metrics are copied with provenance.
        "truthfulqa_gen": truthfulqa,
        "bbh": bbh, "mmlu_sample": mmlu, "humaneval": humaneval,
    }
    report["results"] = [r for r in report.get("results", [])
                         if r.get("model") != args.name] + [result]
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(args.out)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
