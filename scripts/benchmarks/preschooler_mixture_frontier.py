#!/usr/bin/env python3
"""Enumerate every small-model subset on identical, retained private items.

The oracle column is an unattainable upper bound, not a mixture score. A live
router/cascade/verifier must be run separately on a fresh sealed holdout.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


def extract(row: dict) -> dict[tuple[str, str, str, str], bool]:
    outcomes = {}
    for lane, result in row.get("results", {}).items():
        if result.get("status") != "complete" or not result.get("pack_sha256"):
            continue
        for sample in result.get("samples", []):
            correct = sample.get("correct", sample.get("passed"))
            if not isinstance(correct, bool) or not sample.get("id"):
                continue
            key = (lane, result["pack_sha256"], str(sample["id"]), str(sample.get("target", "")))
            if key in outcomes:
                raise ValueError(f"duplicate item for {row['model']}: {key}")
            outcomes[key] = correct
    return outcomes


def enumerate_subsets(rows: list[dict], families: dict[str, str], minimum_items: int = 20) -> dict:
    if not 2 <= len(rows) <= 12:
        raise ValueError("2–12 complete model rows required")
    names = [row["model"] for row in rows]
    if len(set(names)) != len(names) or any(name not in families for name in names):
        raise ValueError("duplicate model or missing family")
    protocols = {row.get("protocol") for row in rows}
    if len(protocols) != 1 or None in protocols:
        raise ValueError("models use different private protocols")
    maps = {row["model"]: extract(row) for row in rows}
    if any(not x for x in maps.values()):
        raise ValueError("a model has no saved item outcomes")
    shared = sorted(set.intersection(*(set(x) for x in maps.values())))
    if len(shared) < minimum_items:
        raise ValueError(f"only {len(shared)} identical items; require {minimum_items}")
    solo = {name: sum(maps[name][item] for item in shared) / len(shared) for name in names}
    best = max(names, key=lambda name: (solo[name], name))
    subsets = []
    for size in range(2, len(names) + 1):
        for members in itertools.combinations(names, size):
            oracle = sum(any(maps[name][item] for name in members) for item in shared) / len(shared)
            random = sum(sum(maps[name][item] for name in members) / size for item in shared) / len(shared)
            best_member = max(solo[name] for name in members)
            subsets.append({"models": list(members), "size": size,
                            "families": len({families[name] for name in members}),
                            "oracle_upper_bound": round(oracle, 6),
                            "oracle_gain_over_global_best": round(oracle - solo[best], 6),
                            "best_member_accuracy": round(best_member, 6),
                            "random_router_expected_accuracy": round(random, 6)})
    subsets.sort(key=lambda item: (-item["oracle_upper_bound"], item["size"],
                                   -item["families"], item["models"]))
    return {"schema": "preschooler-mixture-frontier/v1", "protocol": next(iter(protocols)),
            "status": "offline_oracle_not_live_mixture", "models": names,
            "shared_private_items": len(shared),
            "shared_lanes": sorted({item[0] for item in shared}),
            "best_fixed": {"model": best, "accuracy": round(solo[best], 6)},
            "solo_accuracy": {name: round(value, 6) for name, value in solo.items()},
            "subset_count": len(subsets), "subsets": subsets,
            "next_gate": "Measure top complementary 2–4-model routes/cascades live on a fresh sealed pack, including latency and GPU-board Wh."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specialist-report", type=Path, required=True)
    parser.add_argument("--roster", type=Path, default=Path(__file__).resolve().parents[2] /
                        "config/preschooler_teacher_roster.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-shared-items", type=int, default=20)
    args = parser.parse_args()
    payload = json.loads(args.specialist_report.read_text())
    roster = json.loads(args.roster.read_text())
    families = {model["alias"]: model["family"] for model in roster["models"]}
    rows = [row for row in payload["results"] if row.get("model") in families]
    if len(rows) != len(roster["models"]):
        parser.error("all ten roster models must have specialist rows before exhaustive enumeration")
    result = enumerate_subsets(rows, families, minimum_items=args.min_shared_items)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"subset_count": result["subset_count"],
                      "shared_private_items": result["shared_private_items"],
                      "best_fixed": result["best_fixed"], "top": result["subsets"][:5]}))


if __name__ == "__main__":
    main()
