#!/usr/bin/env python3
"""Fail closed for vLLM deployments that cannot execute on the V100 pair.

This protects the EDS2 volume from downloading a model whose weight format or
minimum VRAM is incompatible with the two 32GB SM70 devices.  It is deliberately
not a benchmark: a benchmark is meaningful only after this check passes.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys


EXACT_FLASH_MODELS = {
    "deepseek-ai/DeepSeek-V4-Flash-0731": {"parameters_b": 304, "weight_format": "FP8 + DSpark"},
    "deepseek-ai/DeepSeek-V4-Flash": {"parameters_b": 284, "weight_format": "FP8 mixed"},
    "deepseek-ai/DeepSeek-V4.1-Flash": {"parameters_b": 552, "weight_format": "FP8 / FP4 mixed"},
}


def gpus() -> list[dict[str, str]]:
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap", "--format=csv,noheader,nounits",
    ], text=True)
    return [dict(zip(("index", "name", "memory_mib", "compute_cap"), (x.strip() for x in row.split(",")), strict=True))
            for row in raw.splitlines()]


def host_ram_gib() -> dict[str, float]:
    fields = {}
    for line in open("/proc/meminfo", encoding="utf-8"):
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            fields[key] = int(value.split()[0]) / 1024**2
    return {"total_gib": fields["MemTotal"], "available_gib": fields["MemAvailable"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tensor-parallel-size", type=int, default=2)
    args = ap.parse_args()
    selected = [gpu for gpu in gpus() if gpu["index"] in {"1", "2"}]
    total_vram_gib = sum(int(gpu["memory_mib"]) for gpu in selected) / 1024
    result: dict[str, object] = {"model": args.model, "v100_gpus": selected, "v100_total_vram_gib": total_vram_gib, "host_ddr4": host_ram_gib(),
                                 "tensor_parallel_size": args.tensor_parallel_size, "supported": True, "reasons": []}
    if args.tensor_parallel_size != 2:
        result["supported"] = False; result["reasons"].append("This host's V100 serving policy requires TP=2.")
    if any(float(gpu["compute_cap"]) < 7.0 for gpu in selected):
        result["supported"] = False; result["reasons"].append("Selected devices do not satisfy vLLM's SM70 baseline.")
    if args.model in EXACT_FLASH_MODELS:
        spec = EXACT_FLASH_MODELS[args.model]
        fp8_weight_floor = spec["parameters_b"] * 1e9 / 1024**3
        # Even a hypothetical 2-bit packing would require at least 71--138GiB
        # before metadata, DSpark and KV cache.  The released models use FP8/FP4,
        # which V100 cannot execute natively.
        result["supported"] = False
        result["reasons"].extend([
            f"{args.model} has {spec['parameters_b']}B parameters ({spec['weight_format']}); its released weights exceed 64GiB V100 VRAM.",
            "Tesla V100 (SM70) has no native FP8/FP4 inference path. CPU offload uses DDR4 but does not make these released FP8/FP4 kernels runnable on V100.",
            f"Even FP8 weights alone have a {fp8_weight_floor:.1f}GiB floor, leaving no safe headroom in host RAM for runtime state, DSpark or the KV cache.",
            "Do not download this model to EDS2 for this host. Use a separately validated SM70-compatible <=~50GiB FP16/INT8 checkpoint, or newer FP8 hardware.",
        ])
    print(json.dumps(result, indent=2))
    return 0 if result["supported"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
