#!/usr/bin/env python3
"""Sampling uncertainty for the public composite and the specialist lanes.

The suite scores small fixed samples (GSM8K 50, BBH 48, MMLU 160, HumanEval
40), so a ranking without error bars overstates what was measured. This module
derives, from the curated evidence only:

* a Wilson 95% interval for every component accuracy;
* a standard error and 95% interval for the unweighted four-part composite;
* which rows are statistically indistinguishable from the leader of their
  access profile (unpaired two-sided z-test, alpha 0.05);
* Wilson intervals for the specialist lanes, whose packs hold 2-6 items.

The comparison is unpaired because the curated JSON keeps per-item outcomes
for HumanEval only. Models answer identical items, so a paired test would be
tighter; the unpaired test is therefore conservative: it may fail to separate
two models that really differ, but it does not invent a difference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WELL_KNOWN = ROOT / "reports/well_known_suite_20260917.json"
SPECIALIST = ROOT / "reports/specialist_suite_20260922.json"
OUT = ROOT / "reports/score_confidence.json"
Z95 = 1.959963984540054
COMPONENTS = ("gsm8k", "bbh", "mmlu", "humaneval")


def wilson(successes: float, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval; stays inside [0, 1] and is non-degenerate at 0 and n."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def variance(p: float, n: int) -> float:
    """Agresti-Coull-adjusted binomial variance: a perfect score on a small
    sample must not report zero uncertainty."""
    adjusted = (p * n + 2) / (n + 4)
    return adjusted * (1 - adjusted) / (n + 4)


def min_items_for_lower_bound(target: float, z: float = Z95) -> int:
    """Smallest n for which a perfect n/n score has a Wilson lower bound >= target."""
    return math.ceil(target * z * z / (1 - target))


def components(row: dict) -> dict[str, tuple[float, int]] | None:
    """Return {component: (accuracy, n)} for a complete row, else None."""
    gsm = row.get("gsm8k") or {}
    gsm_scores = [v for k, v in gsm.items() if k.startswith("exact_match") and "stderr" not in k]
    bbh = row.get("bbh") or {}
    mmlu = row.get("mmlu_sample") or {}
    human = row.get("humaneval") or {}
    parts = {
        "gsm8k": (max(gsm_scores) if gsm_scores else None, gsm.get("sample_len")),
        "bbh": (bbh.get("mean_accuracy"), bbh.get("n_samples")),
        "mmlu": (mmlu.get("mean_accuracy"), mmlu.get("n_samples")),
        "humaneval": (human.get("pass_at_1"), human.get("n_problems")),
    }
    for score, n in parts.values():
        if not isinstance(score, (int, float)) or not isinstance(n, int) or n <= 0:
            return None
    return parts  # type: ignore[return-value]


def composite_stats(parts: dict[str, tuple[float, int]]) -> dict:
    composite = sum(score for score, _ in parts.values()) / len(parts)
    se = math.sqrt(sum(variance(score, n) for score, n in parts.values())) / len(parts)
    detail = {}
    for name, (score, n) in parts.items():
        low, high = wilson(score * n, n)
        detail[name] = {"accuracy": round(score, 4), "n": n,
                        "ci95": [round(low, 4), round(high, 4)]}
    return {
        "composite": round(composite, 4),
        "standard_error": round(se, 4),
        "ci95": [round(max(0.0, composite - Z95 * se), 4), round(min(1.0, composite + Z95 * se), 4)],
        "components": detail,
    }


def well_known_confidence(payload: dict, protocol: str) -> dict:
    by_profile: dict[str, list[dict]] = {}
    for row in payload.get("results", []):
        if "error" in row or row.get("eval_protocol") != protocol:
            continue
        parts = components(row)
        if parts is None:
            continue
        entry = {"model": row["model"], **composite_stats(parts)}
        by_profile.setdefault(row.get("access_profile") or "sandbox", []).append(entry)
    for entries in by_profile.values():
        entries.sort(key=lambda e: e["composite"], reverse=True)
        leader = entries[0]
        for rank, entry in enumerate(entries, 1):
            diff = leader["composite"] - entry["composite"]
            se_diff = math.hypot(leader["standard_error"], entry["standard_error"])
            entry["rank"] = rank
            entry["z_vs_leader"] = round(diff / se_diff, 2) if se_diff else 0.0
            entry["tied_with_leader"] = diff <= Z95 * se_diff
    return by_profile


def specialist_confidence(payload: dict) -> list[dict]:
    rows = []
    for result in payload.get("results", []):
        for lane, cell in (result.get("results") or {}).items():
            accuracy, n = cell.get("accuracy"), cell.get("n_samples")
            if not isinstance(accuracy, (int, float)) or not isinstance(n, int) or n <= 0:
                continue
            low, high = wilson(accuracy * n, n)
            rows.append({"model": result.get("model"), "lane": lane, "accuracy": round(accuracy, 4),
                         "n": n, "ci95": [round(low, 4), round(high, 4)]})
    return rows


def build(protocol: str) -> dict:
    well_known = json.loads(WELL_KNOWN.read_text())
    report = {
        "method": "Wilson 95% per component; composite SE from Agresti-Coull binomial variances; "
                  "unpaired z-test against the profile leader (conservative, see module docstring)",
        "protocol": protocol,
        "source_sha256": {WELL_KNOWN.name: hashlib.sha256(WELL_KNOWN.read_bytes()).hexdigest()},
        "min_items_for_perfect_score_claim": {
            "lower_bound_0.90": min_items_for_lower_bound(0.90),
            "lower_bound_0.95": min_items_for_lower_bound(0.95),
        },
        "well_known": well_known_confidence(well_known, protocol),
    }
    if SPECIALIST.exists():
        report["source_sha256"][SPECIALIST.name] = hashlib.sha256(SPECIALIST.read_bytes()).hexdigest()
        report["specialist"] = specialist_confidence(json.loads(SPECIALIST.read_text()))
    return report


def current_protocol() -> str:
    # Read the constant instead of importing well_known_suite, which pulls in
    # the runner's GPU and lm-eval plumbing.
    source = (Path(__file__).resolve().parent / "well_known_suite.py").read_text()
    for line in source.splitlines():
        if line.startswith("EVAL_PROTOCOL ="):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError("EVAL_PROTOCOL not found in well_known_suite.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    report = build(current_protocol())
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for profile, entries in report["well_known"].items():
        tied = sum(e["tied_with_leader"] for e in entries)
        print(f"{profile}: {len(entries)} complete rows, {tied} statistically tied with the leader")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
