#!/usr/bin/env python3
"""Rank local GGUF models for the virtualpc model-selection decision.

Selection rule (fixed before more sweep data comes in, so the rule isn't
tuned to whatever happens to be ranked first): a model is eligible only once
it has a result row tagged with the CURRENT well_known_suite.EVAL_PROTOCOL
and no "error" key -- partial/stale-protocol rows are reported separately,
never blended into the ranking. The composite score is the unweighted mean
of four 0-1 accuracy metrics (gsm8k, humaneval, mmlu_sample, bbh); truthfulqa
and throughput are reported alongside as context, not folded into the score,
because their scales aren't comparable to plain accuracy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import well_known_suite as wks  # noqa: E402

REPORT = ROOT / "reports/well_known_suite_20260917.json"


def gsm8k_score(row: dict) -> float | None:
    # max(strict-match, flexible-extract): each filter independently
    # under-counts genuine correct answers for different, model-specific
    # formatting reasons, so neither alone is a fair capability estimate.
    # deepseek-r1-qwen32b-q4: a response ending exactly "#### 64" (correct)
    # scores flexible-extract="[invalid]" because that filter's
    # group_select=-1 regex grabs the LAST number-like token anywhere in
    # the text, not the one after "####" -- strict-match=0.68 is the real
    # number there, flexible-extract=0.34 is the artifact.
    # qwen25-72b-q4: the opposite problem -- several correct answers are
    # stated in prose without a trailing "#### N" line at all (e.g. "...
    # have a total of 260 sheep together." with no "####"), so
    # strict-match=0.54 undercounts; flexible-extract=0.72 is closer to
    # the real number there.
    g = row.get("gsm8k") or {}
    vals = [v for k, v in g.items() if k.startswith("exact_match") and "stderr" not in k]
    return max(vals) if vals else None


def truthfulqa_score(row: dict) -> float | None:
    t = row.get("truthfulqa_gen") or {}
    vals = [v for k, v in t.items() if k.endswith("_acc,none")]
    return sum(vals) / len(vals) if vals else None


# Physical GPU index -> card, fixed for this box (checked via nvidia-smi -L):
#   0 = RTX A4000 15GB, 1/2 = Tesla V100-SXM2-32GB (the pair every result so
#   far was measured on), 3 = RTX 4000 Ada 20GB (the live desktop GPU).
# 2026-09-18: box now also benchmarks on GPU0 (A4000) in addition to the
# V100 pair -- results are no longer all on one uniform hardware config, so
# every row needs its hardware surfaced instead of assumed.
_GPU_LABELS = {"0": "A4000-15GB", "1": "V100-32GB", "2": "V100-32GB", "3": "RTX4000Ada-20GB"}


def hardware_label(row: dict) -> str:
    devices = row.get("cuda_visible_devices")
    if not devices:
        return "unknown"
    ids = [d.strip() for d in str(devices).split(",") if d.strip()]
    cards = [_GPU_LABELS.get(d, f"gpu{d}") for d in ids]
    if len(cards) > 1 and len(set(cards)) == 1:
        return f"{len(cards)}x{cards[0]}"
    return "+".join(cards)


def main() -> int:
    if not REPORT.exists():
        print(f"no report yet at {REPORT}")
        return 1
    payload = json.loads(REPORT.read_text())
    results = payload.get("results", [])

    eligible = []
    excluded = []
    for row in results:
        name = row.get("model")
        if row.get("eval_protocol") != wks.EVAL_PROTOCOL or "error" in row:
            excluded.append((name, row.get("eval_protocol"), "error" if "error" in row else "stale-protocol"))
            continue
        gsm8k = gsm8k_score(row)
        humaneval = (row.get("humaneval") or {}).get("pass_at_1")
        mmlu = (row.get("mmlu_sample") or {}).get("mean_accuracy")
        bbh = (row.get("bbh") or {}).get("mean_accuracy")
        tqa = truthfulqa_score(row)
        tps = row.get("completion_tokens_per_second")
        parts = [v for v in (gsm8k, humaneval, mmlu, bbh) if v is not None]
        if len(parts) < 4:
            excluded.append((name, row.get("eval_protocol"), f"incomplete ({len(parts)}/4 metrics)"))
            continue
        composite = sum(parts) / len(parts)
        eligible.append({
            "model": name, "engine": row.get("engine", "llama.cpp-gguf"), "composite": composite,
            "gsm8k": gsm8k, "humaneval": humaneval,
            "mmlu": mmlu, "bbh": bbh, "truthfulqa_acc": tqa, "tokens_per_sec": tps,
            "hardware": hardware_label(row),
        })

    eligible.sort(key=lambda r: r["composite"], reverse=True)

    print(f"Eligible (protocol={wks.EVAL_PROTOCOL}, complete evidence): {len(eligible)}")
    print(f"{'model':<26} {'engine':<22} {'hardware':<16} {'composite':>9} {'gsm8k':>7} {'human':>7} {'mmlu':>7} {'bbh':>7} {'tqa_acc':>7} {'tok/s':>7}")
    for r in eligible:
        print(f"{r['model']:<26} {r['engine']:<22} {r['hardware']:<16} {r['composite']:>9.3f} {r['gsm8k']:>7.3f} {r['humaneval']:>7.3f} "
              f"{r['mmlu']:>7.3f} {r['bbh']:>7.3f} "
              f"{(r['truthfulqa_acc'] if r['truthfulqa_acc'] is not None else float('nan')):>7.3f} "
              f"{(r['tokens_per_sec'] if r['tokens_per_sec'] is not None else float('nan')):>7.1f}")

    if excluded:
        print(f"\nNot eligible yet ({len(excluded)}):")
        for name, proto, reason in excluded:
            print(f"  {name}: {reason} (eval_protocol={proto})")

    if len(eligible) >= 2:
        picks = eligible[:3]
        print(f"\nTop {len(picks)} by composite score: {', '.join(r['model'] for r in picks)}")
    else:
        print("\nFewer than 2 models have complete, current-protocol evidence -- not enough to select yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
