#!/usr/bin/env python3
"""Offline per-(layer, expert) tier plan from observed activation counts.

The manifest is advisory until a serving engine consumes it. Bandwidth
estimates are traffic fractions, never an end-to-end token-rate prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

LAYERS = 60
EXPERTS = 384
SCHEMA = "kimi-k25-placement/v1"


def read_counts(path: Path) -> tuple[list[tuple[int, list[int]]], str]:
    payload = path.read_bytes()
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    by_layer = {}
    for row in rows:
        layer, counts = row["layer"], row["counts"]
        if (not isinstance(layer, int) or layer < 1 or layer > LAYERS or layer in by_layer
                or len(counts) != EXPERTS or any(type(value) is not int or value < 0 for value in counts)
                or sum(counts) == 0):
            raise ValueError("invalid or duplicate layer/expert activation counts")
        by_layer[layer] = counts
    if set(by_layer) != set(range(1, LAYERS + 1)):
        raise ValueError("activation counts must cover MoE layers 1..60")
    return sorted(by_layer.items()), hashlib.sha256(payload).hexdigest()


def plan(rows: list[tuple[int, list[int]]], source_sha256: str, *, expert_bytes: int,
         gpu_bytes: int, dram_bytes: int, pmem0_bytes: int, pmem1_bytes: int) -> dict:
    if expert_bytes <= 0 or any(v < 0 for v in (gpu_bytes, dram_bytes, pmem0_bytes, pmem1_bytes)):
        raise ValueError("capacities and expert_bytes must be nonnegative; expert_bytes positive")
    budget = {"gpu": gpu_bytes, "dram": dram_bytes, "pmem0": pmem0_bytes,
              "pmem1": pmem1_bytes}
    used = {tier: 0 for tier in (*budget, "disk")}
    traffic = {tier: 0.0 for tier in used}
    instances = []
    for layer, counts in rows:
        total = sum(counts)
        instances.extend((count / total, layer, expert) for expert, count in enumerate(counts))
    # Placement keys are independent per layer. The hottest instances fill
    # expensive tiers first; PMem loads are balanced by weighted access count.
    assignments = []
    for weight, layer, expert in sorted(instances, key=lambda item: (-item[0], item[1], item[2])):
        if used["gpu"] + expert_bytes <= budget["gpu"]:
            tier = "gpu"
        elif used["dram"] + expert_bytes <= budget["dram"]:
            tier = "dram"
        else:
            candidates = [tier for tier in ("pmem0", "pmem1")
                          if used[tier] + expert_bytes <= budget[tier]]
            tier = min(candidates, key=lambda item: (traffic[item], used[item], item)) if candidates else "disk"
        used[tier] += expert_bytes
        traffic[tier] += weight
        assignments.append({"layer": layer, "expert": expert, "tier": tier})
    assignments.sort(key=lambda row: (row["layer"], row["expert"]))
    return {"schema": SCHEMA, "source_sha256": source_sha256,
            "expert_bytes_assumed": expert_bytes, "capacity_bytes": budget,
            "used_bytes": used, "estimated_routed_traffic_share":
            {tier: round(weight / LAYERS, 6) for tier, weight in traffic.items()},
            "assignments": assignments,
            "status": "offline_plan_only_engine_loader_unverified"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("counts", type=Path)
    ap.add_argument("--expert-bytes", required=True, type=int,
                    help="measured native weight bytes per (layer, expert); no guessed default")
    for tier in ("gpu", "dram", "pmem0", "pmem1"):
        ap.add_argument(f"--{tier}-bytes", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    rows, source = read_counts(args.counts)
    manifest = plan(rows, source, expert_bytes=args.expert_bytes, gpu_bytes=args.gpu_bytes,
                    dram_bytes=args.dram_bytes, pmem0_bytes=args.pmem0_bytes,
                    pmem1_bytes=args.pmem1_bytes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({key: manifest[key] for key in ("status", "used_bytes", "estimated_routed_traffic_share")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
