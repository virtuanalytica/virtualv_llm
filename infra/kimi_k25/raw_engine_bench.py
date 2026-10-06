#!/usr/bin/env python3
"""Raw-engine measurement for a running Kimi K2.5 OpenAI-compatible endpoint (condition A).

B1 and B4 decode, prefill, per-request latency and GPU energy from 1 s nvidia-smi samples, with unique
prompts (no prompt-cache hits), instant mode (chat_template_kwargs thinking=false), temperature 0.
Writes one JSON with every raw request timing; it never estimates a missing number.
"""
from __future__ import annotations

import argparse, json, statistics, subprocess, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

FILLER = "Numerai Signals scores 20-day forward returns; point-in-time data must exclude look-ahead. " * 30


def request(base, model, i, max_tokens, timeout):
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"thinking": False},
            "messages": [{"role": "user", "content": f"Request {i}. Context: {FILLER} Question: explain in detail "
                                                       "how point-in-time validation prevents look-ahead bias."}]}
    t = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=timeout).read())
    tm = r.get("timings", {})
    return {"i": i, "wall_s": round(time.time() - t, 3), "pp_tokens": tm.get("prompt_n"), "pp_tps": tm.get("prompt_per_second"),
            "tg_tokens": tm.get("predicted_n"), "tg_tps": tm.get("predicted_per_second"),
            "completion_tokens": (r.get("usage") or {}).get("completion_tokens")}


def sampler(stop, rows):
    while not stop.is_set():
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,power.draw,memory.used,utilization.gpu",
                                           "--format=csv,noheader,nounits"], text=True)
            rows.append({"t": time.time(), "gpus": [list(map(float, l.split(","))) for l in out.strip().splitlines()]})
        except Exception:
            pass
        stop.wait(1.0)


def phase(base, model, label, concurrency, rounds, max_tokens, timeout, start_i):
    samples, stop = [], threading.Event()
    th = threading.Thread(target=sampler, args=(stop, samples), daemon=True); th.start()
    t0 = time.time(); reqs = []; i = start_i
    for _ in range(rounds):
        with ThreadPoolExecutor(concurrency) as ex:
            futs = [ex.submit(request, base, model, i + k, max_tokens, timeout) for k in range(concurrency)]
            reqs += [f.result() for f in futs]
        i += concurrency
    wall = time.time() - t0; stop.set(); th.join(2)
    energy_wh = 0.0
    for a, b in zip(samples, samples[1:]):
        energy_wh += sum(g[1] for g in a["gpus"]) * (b["t"] - a["t"]) / 3600
    tg = sum(r["tg_tokens"] or 0 for r in reqs)
    return {"phase": label, "concurrency": concurrency, "requests": reqs, "wall_s": round(wall, 2),
            "aggregate_tg_tps": round(tg / wall, 3) if wall else None,
            "per_request_tg_tps_median": statistics.median([r["tg_tps"] for r in reqs if r["tg_tps"]]),
            "per_request_pp_tps_median": statistics.median([r["pp_tps"] for r in reqs if r["pp_tps"]]),
            "latency_p50_s": statistics.median([r["wall_s"] for r in reqs]),
            "latency_p95_s": sorted(r["wall_s"] for r in reqs)[max(0, int(round(0.95 * len(reqs))) - 1)],
            "gpu_energy_wh": round(energy_wh, 3), "gpu_wh_per_answer": round(energy_wh / len(reqs), 4),
            "gpu_peak_memory_mib": {int(g[0]): max(s["gpus"][k][2] for s in samples) for k, g in enumerate(samples[0]["gpus"])} if samples else {},
            "cpu_energy": "not measured (RAPL unavailable without root)"}, i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18019"); ap.add_argument("--model", default="k")
    ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=128); ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--b1-requests", type=int, default=3); ap.add_argument("--b4-rounds", type=int, default=2)
    a = ap.parse_args()
    request(a.url, a.model, 0, 16, a.timeout)  # warmup, not recorded
    b1, i = phase(a.url, a.model, "B1", 1, a.b1_requests, a.max_tokens, a.timeout, 1)
    b4, _ = phase(a.url, a.model, "B4", 4, a.b4_rounds, a.max_tokens, a.timeout, i)
    json.dump({"label": a.label, "url": a.url, "max_tokens": a.max_tokens, "instant_mode": True, "phases": [b1, b4]},
              open(a.out, "w"), indent=2)
    print(json.dumps({p["phase"]: {k: p[k] for k in ("aggregate_tg_tps", "per_request_tg_tps_median",
          "per_request_pp_tps_median", "latency_p50_s", "gpu_wh_per_answer")} for p in (b1, b4)}))


if __name__ == "__main__":
    main()
