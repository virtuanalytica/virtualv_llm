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
AUDIT = ROOT / "reports/contamination_audit_20260923.json"
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


def mcnemar_exact(only_first: int, only_second: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n, k = only_first + only_second, min(only_first, only_second)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def humaneval_paired(payload: dict, protocol: str) -> dict[str, list[dict]]:
    """Paired comparison with the profile leader on the HumanEval items both ran.

    HumanEval is the one task whose per-item outcomes are in the curated JSON.
    Pairing uses only the items where the two models disagree, which is why it
    can separate models the unpaired composite test leaves tied.
    """
    ranked = well_known_confidence(payload, protocol)
    rows = {r.get("model"): r for r in payload.get("results", [])}

    def outcomes(model: str) -> dict[str, bool]:
        tasks = (rows[model].get("humaneval") or {}).get("per_task") or {}
        return {task: bool(cell.get("passed")) for task, cell in tasks.items() if isinstance(cell, dict)}

    result: dict[str, list[dict]] = {}
    for profile, entries in ranked.items():
        leader = outcomes(entries[0]["model"])
        if not leader:
            continue
        for entry in entries[1:]:
            other = outcomes(entry["model"])
            shared = sorted(set(leader) & set(other))
            if not shared:
                continue
            leader_only = sum(leader[t] and not other[t] for t in shared)
            other_only = sum(other[t] and not leader[t] for t in shared)
            p_value = mcnemar_exact(leader_only, other_only)
            result.setdefault(profile, []).append({
                "model": entry["model"], "leader": entries[0]["model"], "shared_items": len(shared),
                "leader_only_pass": leader_only, "model_only_pass": other_only,
                "p_value": round(p_value, 4), "differs_from_leader": p_value < 0.05,
            })
    return result


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


# Vision needs a multimodal model, so the text-model composite leaves it out.
SPECIALIST_LANES = ("chemistry", "physics", "iq", "eq", "fq", "qq", "finance", "video")
CANARY_FLAG = 0.2      # share of held-out GSM8K items continued verbatim
FORM_GAP_FLAG = 0.2    # accuracy lost when MMLU options are reordered


def _mean_with_interval(parts: list[tuple[float, int]]) -> dict:
    value = sum(score for score, _ in parts) / len(parts)
    se = math.sqrt(sum(variance(score, n) for score, n in parts)) / len(parts)
    return {"value": round(value, 4), "standard_error": round(se, 4), "items": sum(n for _, n in parts),
            "ci95": [round(max(0.0, value - Z95 * se), 4), round(min(1.0, value + Z95 * se), 4)]}


def specialist_composite(row: dict) -> dict | None:
    """Unweighted mean over the text specialist lanes that completed.

    ``complete`` is true only with all lanes present; a partial mean is kept
    for inspection but must not be ranked against complete ones.
    """
    cells = row.get("results") or {}
    parts = [(cells[lane]["accuracy"], cells[lane]["n_samples"]) for lane in SPECIALIST_LANES
             if cells.get(lane, {}).get("status") == "complete"
             and isinstance(cells[lane].get("accuracy"), (int, float)) and cells[lane].get("n_samples")]
    if not parts:
        return None
    return {**_mean_with_interval(parts), "lanes": len(parts), "complete": len(parts) == len(SPECIALIST_LANES)}


def resistant_composite(row: dict) -> dict | None:
    """Mean of the two scores a model cannot have memorised in this form:
    the post-cutoff holdout and MMLU with reordered options. The canary probe
    and the form gap are reported as flags, not folded into the number."""
    methods = row.get("results") or {}
    holdout, paraphrase, canary = (methods.get(k) or {} for k in
                                   ("post_cutoff_holdout", "paraphrase_invariance", "canary_recall"))
    if holdout.get("status") != "complete" or paraphrase.get("status") != "complete":
        return None
    result = _mean_with_interval([(holdout["accuracy"], holdout["n_samples"]),
                                  (paraphrase["paraphrase_accuracy"], paraphrase["n_samples"])])
    flags = []
    if canary.get("status") == "complete" and canary.get("recall_rate", 0) >= CANARY_FLAG:
        flags.append("canary-recall")
    if paraphrase.get("accuracy_gap", 0) >= FORM_GAP_FLAG:
        flags.append("form-sensitive")
    return {**result, "holdout": holdout["accuracy"], "paraphrased_mmlu": paraphrase["paraphrase_accuracy"],
            "canary_recall": canary.get("recall_rate"), "flags": flags}


def resistant_ranking(well_known: dict, specialist: dict, audit: dict, protocol: str) -> list[dict]:
    """Per model: public composite next to the two contamination-resistant ones.

    Sorted by the resistant composite; models without an audit follow, sorted
    by specialist composite. Sandbox rows only.
    """
    public = {e["model"]: e for e in well_known_confidence(well_known, protocol).get("sandbox", [])}

    def sandbox(rows: list[dict]) -> dict[str, dict]:
        return {r["model"]: r for r in rows if r.get("access_profile", "sandbox") == "sandbox"}

    spec, aud = sandbox(specialist.get("results", [])), sandbox(audit.get("results", []))
    rows = []
    for model in sorted(set(spec) | set(aud)):
        entry = {"model": model,
                 "public_composite": public.get(model, {}).get("composite"),
                 "specialist": specialist_composite(spec[model]) if model in spec else None,
                 "resistant": resistant_composite(aud[model]) if model in aud else None}
        if entry["specialist"] or entry["resistant"]:
            if entry["public_composite"] is not None and entry["resistant"]:
                entry["public_minus_resistant"] = round(entry["public_composite"] - entry["resistant"]["value"], 4)
            rows.append(entry)
    rows.sort(key=lambda e: ((e["resistant"] or {}).get("value", -1), (e["specialist"] or {}).get("value", -1)),
              reverse=True)
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
        "humaneval_paired": humaneval_paired(well_known, protocol),
    }
    if SPECIALIST.exists():
        report["source_sha256"][SPECIALIST.name] = hashlib.sha256(SPECIALIST.read_bytes()).hexdigest()
        report["specialist"] = specialist_confidence(json.loads(SPECIALIST.read_text()))
        audit = json.loads(AUDIT.read_text()) if AUDIT.exists() else {"results": []}
        report["resistant_ranking"] = resistant_ranking(well_known, json.loads(SPECIALIST.read_text()), audit, protocol)
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
