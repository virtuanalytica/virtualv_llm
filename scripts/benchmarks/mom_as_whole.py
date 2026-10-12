#!/usr/bin/env python3
"""Mixture-of-models as one system: combine per-item results of all members offline.

Every member answered the same public items with the same prompts (week plan 2026-10-12, GPU queue
driver gpuqueue/drivers/bench.py). This script reads those per-item files, keeps the fixed test half
(seed 20261012), and reports per set: each member, the oracle of all members (upper bound), and
escalation selectors "local members agree -> local answer, else Haiku 5.5".

The pre-registered selector (KAT + Gemma4, thinking off) is reported as such; the best-k selectors
rank members on the same test half and are labelled exploratory. Development measurement on public
sets, not promotion evidence.

    python3 scripts/benchmarks/mom_as_whole.py --results /media/knight2/EDS2/gpuqueue/results/w42 \
        --out reports/mom_as_whole_20261012.json
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from math import comb
from pathlib import Path

SETS = ("gsm8k", "tqa_mc1", "mmlu_pro", "ifeval")
PREREGISTERED = ("kat_off", "gemma4_off")
MODEL_NAMES = {"haiku": "Claude Haiku 5.5 (cloud)", "q36": "Qwen3.6-35B-A3B NVFP4 (mom-live)",
               "qwen38": "Qwen3.8 Flash-Next AP IQ2_S", "kat": "KAT-Coder-V2.5-Dev Q4_K_M",
               "nemotron": "Nemotron-3.5-Lightning-30B-A3B", "gemma4": "Gemma4-26B-A4B Q4_K_M"}


def answer(st: str, tail: str):
    """Normalised final answer, used only to decide whether members agree (scoring is the driver's)."""
    t = re.sub(r"(?s)^.*</think>", "", tail or "")
    if st == "gsm8k":
        m = re.findall(r"answer\s*[:=]\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", t, re.I) or re.findall(r"-?[\d,]*\.?\d+", t)
        try:
            return float(m[-1].replace(",", "")) if m else None
        except ValueError:
            return None
    if st in ("tqa_mc1", "mmlu_pro"):
        m = re.search(r"answer\s*(?:is)?\s*:?\s*\**\s*([A-Z])\b", t, re.I) or \
            re.search(r"(?<![A-Za-z'])([A-Z])(?![A-Za-z'])", t)
        return m.group(1).upper() if m else None
    return None  # free text (IFEval): agreement is not defined


def mcnemar(a: list[int], b: list[int]) -> dict:
    x = sum(1 for i, j in zip(a, b) if i and not j)
    y = sum(1 for i, j in zip(a, b) if j and not i)
    n = x + y
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(x, y) + 1)) / 2 ** n)
    return {"better": x, "worse": y, "p_exact": round(p, 4)}


def load_members(results: Path, st: str) -> tuple[dict, list[str]]:
    members, skipped = {}, []
    for f in sorted(results.glob(f"{st}_*.json")):
        name = re.sub(r"_r\d+$", "", f.stem[len(st) + 1:])  # a rerun (_r2) replaces the earlier file
        if name.endswith("_dev"):
            continue
        d = json.loads(f.read_text())
        if d.get("errors", 0) > 5:
            if name not in members:
                skipped.append(f"{name}: {d['errors']} errors")
            continue
        skipped = [x for x in skipped if not x.startswith(name + ":")]
        members[name] = {"rows": {r["idx"]: (answer(st, r.get("reply_tail")), int(r["correct"]), r["half"])
                                  for r in d["items"]},
                         "tokens": d.get("mean_tokens"), "source": d.get("source")}
    return members, skipped


def summarise(st: str, members: dict, skipped: list[str]) -> dict:
    if "haiku" not in members:
        return {"status": "no Haiku run", "skipped": skipped}
    test = sorted(i for i, v in members["haiku"]["rows"].items() if v[2] == "test")
    for n in list(members):
        if not set(test) <= set(members[n]["rows"]):
            skipped.append(f"{n}: does not cover the test half")
            members.pop(n)
    acc = {n: sum(m["rows"][i][1] for i in test) / len(test) for n, m in members.items()}
    out_members = [{"name": n, "model": MODEL_NAMES.get(n.split("_")[0], n.split("_")[0]),
                    "thinking": {"on": True, "off": False}.get(n.split("_")[-1]) if "_" in n else None,
                    "variant": n.split("_", 1)[1] if "_" in n else "", "score": round(acc[n], 4),
                    "mean_tokens": m["tokens"]} for n, m in sorted(members.items(), key=lambda kv: -acc[kv[0]])]
    oracle = sum(1 for i in test if any(members[n]["rows"][i][1] for n in members)) / len(test)
    haiku = [members["haiku"]["rows"][i][1] for i in test]
    selectors = []
    if st != "ifeval":
        local = [n for n in members if n != "haiku"]
        best_variant = {}
        for n in sorted(local, key=lambda n: -acc[n]):
            best_variant.setdefault(n.split("_")[0], n)
        candidates = [(list(PREREGISTERED), True)] + [(list(best_variant.values())[:k], False) for k in (2, 3)]
        seen = set()
        for top, pre in candidates:
            if not all(t in members for t in top) or len(top) < 2 or tuple(top) in seen:
                continue
            seen.add(tuple(top))
            picks, escalated = [], 0
            for i in test:
                ans = [members[n]["rows"][i][0] for n in top]
                if ans[0] is not None and all(a == ans[0] for a in ans):
                    picks.append(members[top[0]]["rows"][i][1])
                else:
                    picks.append(members["haiku"]["rows"][i][1])
                    escalated += 1
            score = sum(picks) / len(test)
            entry = {"members": top, "preregistered": pre, "score": round(score, 4),
                     "haiku_share": round(escalated / len(test), 4), "vs_haiku": mcnemar(picks, haiku)}
            if pre:
                entry["rule"] = "pass if score >= Haiku - 0.01 and Haiku share <= 0.25"
                entry["passed"] = score >= acc["haiku"] - 0.01 and escalated / len(test) <= 0.25
            selectors.append(entry)
    return {"n_test": len(test), "haiku_score": round(acc["haiku"], 4), "members": out_members,
            "oracle_all_members": round(oracle, 4), "selectors": selectors, "skipped": skipped}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    report = {"schema": "mom-as-whole/v1", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "split": "fixed test half per set, seed 20261012", "access": "public sets; development measurement",
              "selector_note": ("agree -> local answer of the first member, else Haiku 5.5; the pre-registered pair "
                                "is fixed, best-k pairs are ranked on the same test half (exploratory)"),
              "sets": {}}
    for st in SETS:
        members, skipped = load_members(a.results, st)
        report["sets"][st] = summarise(st, members, skipped)
    a.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
