#!/usr/bin/env python3
"""Measured report of the live mixture-of-models (mom-live) for the dashboard.

Everything here is measured on the box, nothing is estimated:
  - layout       the systemd --user units of infra/model_serve_configs/mom-live.sh (model, GPU, port)
  - throughput   llama-server's own `print_timing` lines per unit (prompt and generation tokens/s,
                 tokens, requests) inside the measurement window
  - energy       GPU board power per GPU (nvidia-smi, 1 s) and CPU package energy (RAPL, both sockets,
                 1 s) inside the same window; DRAM, disks, fans and PSU losses are NOT included
  - per answer   aggregator requests in the window = answered mixture turns; energy per answer is the
                 whole measured system (all six GPUs + CPU packages) divided by those answers

    python3 scripts/reporting/live_mixture_report.py --gpu-log power_gpu.csv --cpu-log power_cpu.csv \
        --since "2026-10-05 11:46:00" --until "2026-10-05 13:00:00" --out reports/live_mixture_of_models_20261005.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

PROPOSERS = {
    "mom-devstral": ("proposer", "devstral-small2-24b-q4", (0,), 8022, "llama.cpp"),
    "mom-qwen35": ("proposer", "qwen35-27b-q4", (1,), 8023, "llama.cpp"),
    "mom-qwen35b": ("proposer", "qwen35-27b-q4", (5,), 8025, "llama.cpp"),
    "mom-gemma4": ("proposer", "gemma4-26b-a4b-q4", (2,), 8024, "llama.cpp"),
}
LAYOUTS = {  # unit: (role, model, CUDA indices in PCI order, port, engine)
    # aggregator as two data-parallel llama.cpp replicas, one per V100
    "replicas": {"mom-qwen38": ("aggregator", "qwen38-27b-q4", (3,), 8021, "llama.cpp"),
                 "mom-qwen38b": ("aggregator", "qwen38-27b-q4", (4,), 8026, "llama.cpp"), **PROPOSERS},
    # aggregator as one 1Cat-vLLM tensor-parallel (TP2) instance over the NVLinked V100 pair
    "tp2": {"mom-qwen38-tp2": ("aggregator", "qwen38-1cat-nvfp4-tp2", (3, 4), 8027, "vllm"), **PROPOSERS},
    # aggregator Qwen3.6-35B-A3B NVFP4 (the strongest NVFP4 model measured alone) as 1Cat-vLLM TP2
    "q36tp2": {"mom-qwen36-tp2": ("aggregator", "qwen36-35b-a3b-1cat-nvfp4-tp2", (3, 4), 8028, "vllm"), **PROPOSERS},
}
_T = re.compile(r"(prompt eval|eval) time =\s*([\d.]+) ms /\s*(\d+) tokens")


def unit_timings(unit: str, since: str, until: str) -> dict:
    out = subprocess.run(["journalctl", "--user", "-u", unit, "--since", since, "--until", until, "--no-pager",
                          "-o", "cat"], capture_output=True, text=True).stdout
    prompt_ms = prompt_tok = gen_ms = gen_tok = 0.0
    gen_rates, requests = [], 0
    for line in out.splitlines():
        m = _T.search(line)
        if not m:
            continue
        kind, ms, tok = m.group(1), float(m.group(2)), int(m.group(3))
        if kind == "prompt eval":
            prompt_ms, prompt_tok = prompt_ms + ms, prompt_tok + tok
        else:
            gen_ms, gen_tok, requests = gen_ms + ms, gen_tok + tok, requests + 1
            if ms > 0 and tok > 0:
                gen_rates.append(tok / ms * 1000)
    return {"requests": requests, "generated_tokens": int(gen_tok), "prompt_tokens": int(prompt_tok),
            "generation_tokens_per_second": round(gen_tok / gen_ms * 1000, 2) if gen_ms else None,
            "generation_tokens_per_second_median_request": round(statistics.median(gen_rates), 2) if gen_rates else None,
            "prompt_tokens_per_second": round(prompt_tok / prompt_ms * 1000, 2) if prompt_ms else None}


def _metrics(path: Path | None) -> dict[str, float]:
    """Prometheus text snapshot of a vLLM /metrics endpoint -> {metric_name: summed value}."""
    out: dict[str, float] = {}
    if not path or not path.exists():
        return out
    for line in path.read_text().splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, _, value = line.rpartition(" ")
        base = name.split("{", 1)[0]
        try:
            out[base] = out.get(base, 0.0) + float(value)
        except ValueError:
            pass
    return out


def vllm_timings(start: Path | None, end: Path | None) -> dict:
    a, b = _metrics(start), _metrics(end)
    if not a or not b:
        return {"requests": 0, "generated_tokens": 0, "prompt_tokens": 0, "generation_tokens_per_second": None,
                "prompt_tokens_per_second": None, "note": "no vLLM /metrics snapshots for this window"}
    d = {k: b.get(k, 0.0) - a.get(k, 0.0) for k in b}
    gen, prompt = d.get("vllm:generation_tokens_total", 0.0), d.get("vllm:prompt_tokens_total", 0.0)
    decode_s = d.get("vllm:request_decode_time_seconds_sum", 0.0)
    prefill_s = d.get("vllm:request_prefill_time_seconds_sum", 0.0)
    return {"requests": int(d.get("vllm:request_success_total", 0.0)), "generated_tokens": int(gen),
            "prompt_tokens": int(prompt),
            "generation_tokens_per_second": round(gen / decode_s, 2) if decode_s else None,
            "prompt_tokens_per_second": round(prompt / prefill_s, 2) if prefill_s else None}


def aggregator_answers(unit: str, engine: str, since: str, until: str) -> int:
    """Answered mixture turns of one aggregator unit in a window, from its own journal: one llama.cpp
    generation timing line, or one vLLM access-log line for /v1/chat/completions, per answer."""
    out = subprocess.run(["journalctl", "--user", "-u", unit, "--since", since, "--until", until, "--no-pager",
                          "-o", "cat"], capture_output=True, text=True).stdout
    if engine == "vllm":
        return sum(1 for line in out.splitlines() if '"POST /v1/chat/completions' in line and " 200" in line)
    return sum(1 for line in out.splitlines() if _T.search(line) and "prompt eval" not in line)


def _ts(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()


def gpu_energy(path: Path, t0: float, t1: float) -> dict[int, dict]:
    """Trapezoid-free 1 s sampling: energy = sum(power) * sample period (measured from timestamps)."""
    per: dict[int, list[tuple[float, float, float]]] = {}
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            t = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            idx, watt, util = int(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        if t0 <= t <= t1:
            per.setdefault(idx, []).append((t, watt, util))
    out = {}
    for idx, rows in per.items():
        rows.sort()
        joules = sum((rows[i + 1][0] - rows[i][0]) * rows[i][1] for i in range(len(rows) - 1))
        out[idx] = {"mean_watt": round(statistics.fmean(r[1] for r in rows), 1),
                    "mean_utilisation_pct": round(statistics.fmean(r[2] for r in rows), 1),
                    "joules": round(joules, 1), "samples": len(rows)}
    return out


def cpu_energy(path: Path, t0: float, t1: float) -> dict:
    rows = []
    for line in path.read_text().splitlines():
        try:
            t, e0, e1, wrap = (float(x) for x in line.split(","))
        except ValueError:
            continue
        if t0 <= t <= t1:
            rows.append((t, e0, e1, wrap))
    if len(rows) < 2:
        return {}
    joules = 0.0
    for a, b in zip(rows, rows[1:]):
        for i in (1, 2):
            d = b[i] - a[i]
            joules += (d if d >= 0 else d + a[3]) / 1e6        # RAPL counter wraps at max_energy_range_uj
    span = rows[-1][0] - rows[0][0]
    return {"joules": round(joules, 1), "mean_watt": round(joules / span, 1) if span else None,
            "scope": "RAPL package energy, sockets 0+1 (no DRAM)"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-log", type=Path, required=True)
    ap.add_argument("--cpu-log", type=Path, required=True)
    ap.add_argument("--since", required=True)
    ap.add_argument("--until", required=True)
    ap.add_argument("--suite-model", default="mom-live-4")
    ap.add_argument("--layout", choices=sorted(LAYOUTS), default="replicas")
    ap.add_argument("--vllm-metrics-start", type=Path)
    ap.add_argument("--vllm-metrics-end", type=Path)
    ap.add_argument("--phase-since", help="start of a sub-window used for like-for-like comparisons")
    ap.add_argument("--phase-until")
    ap.add_argument("--phase-label", default="")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t0, t1 = _ts(a.since), _ts(a.until)
    gpus, cpu = gpu_energy(a.gpu_log, t0, t1), cpu_energy(a.cpu_log, t0, t1)
    units = {}
    for unit, (role, model, idxs, port, engine) in LAYOUTS[a.layout].items():
        timing = (vllm_timings(a.vllm_metrics_start, a.vllm_metrics_end) if engine == "vllm"
                  else unit_timings(unit, a.since, a.until))
        u = {"role": role, "model": model, "cuda_index": ",".join(map(str, idxs)), "port": port, "engine": engine,
             **timing}
        parts = [gpus[i] for i in idxs if i in gpus]
        g = {"mean_watt": round(sum(p["mean_watt"] for p in parts), 1),
             "mean_utilisation_pct": round(statistics.fmean(p["mean_utilisation_pct"] for p in parts), 1),
             "joules": round(sum(p["joules"] for p in parts), 1)} if parts else {}
        u["gpu"] = g
        if g.get("joules") and u["generated_tokens"]:
            u["gpu_joules_per_generated_token"] = round(g["joules"] / u["generated_tokens"], 3)
        units[unit] = u
    answers = sum(u["requests"] for u in units.values() if u["role"] == "aggregator")
    gpu_j = sum(g["joules"] for g in gpus.values())
    total_j = gpu_j + cpu.get("joules", 0.0)
    gen_tokens = sum(u["generated_tokens"] for u in units.values())
    agg_tokens = sum(u["generated_tokens"] for u in units.values() if u["role"] == "aggregator")
    report = {
        "name": "mom-live", "suite_model": a.suite_model, "layout": a.layout, "method": "Mixture-of-Agents (proposers draft, aggregator answers)",
        "window": {"since": a.since, "until": a.until, "seconds": round(t1 - t0)},
        "units": units, "gpus": {str(k): v for k, v in sorted(gpus.items())}, "cpu": cpu,
        "totals": {
            "answers": answers, "generated_tokens_all_members": gen_tokens, "answer_tokens": agg_tokens,
            "gpu_kwh": round(gpu_j / 3.6e6, 4), "cpu_kwh": round(cpu.get("joules", 0.0) / 3.6e6, 4),
            "system_kwh_measured": round(total_j / 3.6e6, 4),
            "mean_system_watt_measured": round(total_j / (t1 - t0), 1) if t1 > t0 else None,
            "wh_per_answer": round(total_j / 3600 / answers, 3) if answers else None,
            "joules_per_answer_token": round(total_j / agg_tokens, 2) if agg_tokens else None,
            "joules_per_generated_token_all_members": round(total_j / gen_tokens, 2) if gen_tokens else None,
        },
        "scope_note": "Gemeten: GPU-bordvermogen (nvidia-smi) en CPU-pakketenergie (RAPL). Niet gemeten: DRAM, "
                      "schijven, ventilatoren en voedingsverlies. Het rustverbruik van alle zes GPU's telt mee, dus de "
                      "cijfers zijn wat de mix op deze machine werkelijk kost, niet de marginale kost per verzoek.",
    }
    if a.phase_since and a.phase_until:
        p0, p1 = _ts(a.phase_since), _ts(a.phase_until)
        pg, pc = gpu_energy(a.gpu_log, p0, p1), cpu_energy(a.cpu_log, p0, p1)
        pj = sum(g["joules"] for g in pg.values()) + pc.get("joules", 0.0)
        pans = sum(aggregator_answers(u, e, a.phase_since, a.phase_until)
                   for u, (role, _m, _i, _p, e) in LAYOUTS[a.layout].items() if role == "aggregator")
        report["phase"] = {"label": a.phase_label, "since": a.phase_since, "until": a.phase_until,
                           "seconds": round(p1 - p0), "answers": pans,
                           "system_kwh_measured": round(pj / 3.6e6, 4),
                           "mean_system_watt_measured": round(pj / (p1 - p0), 1) if p1 > p0 else None,
                           "wh_per_answer": round(pj / 3600 / pans, 3) if pans else None,
                           "answers_per_minute": round(pans / (p1 - p0) * 60, 2) if p1 > p0 else None}
    a.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["totals"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
