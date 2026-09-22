#!/usr/bin/env python3
"""Matched local quality/throughput probe for 1Cat target and DFlash2 modes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
import sys
from pathlib import Path
import subprocess
import threading
import time
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/benchmarks"))
import eval_suite  # noqa: E402

PERF_PROMPT = ("Write a continuous technical explanation of point-in-time validation for financial machine "
               "learning. Use complete sentences and keep writing until the token budget ends.")


def post(url: str, payload: dict[str, Any], timeout: int = 300) -> dict[str, Any]:
    req = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def sample_gpus() -> dict[int, dict[str, float]]:
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits"], text=True)
    out = {}
    for line in raw.splitlines():
        values = [v.strip() for v in line.split(",")]
        idx = int(values[0])
        if idx in (1, 2):
            out[idx] = dict(zip(("util_pct", "memory_mib", "power_w", "temperature_c"),
                                map(float, values[1:]), strict=True))
    return out


def monitor(stop: threading.Event, rows: list[dict[int, dict[str, float]]]) -> None:
    while not stop.is_set():
        try:
            rows.append(sample_gpus())
        except Exception:
            pass
        stop.wait(.25)


def summarize(rows: list[dict[int, dict[str, float]]]) -> dict[str, Any]:
    result = {}
    for idx in (1, 2):
        found = [row[idx] for row in rows if idx in row]
        result[str(idx)] = {"samples": len(found), **{
            f"max_{key}": max((item[key] for item in found), default=None)
            for key in ("util_pct", "memory_mib", "power_w", "temperature_c")}}
    return result


def completion(base: str, model: str, prompt: str, tokens: int, ignore_eos: bool = False) -> dict[str, Any]:
    return post(base + "/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "seed": 42, "max_tokens": tokens, "ignore_eos": ignore_eos,
        "chat_template_kwargs": {"enable_thinking": False}})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("target", "dflash2"), required=True)
    parser.add_argument("--port", type=int, default=18012)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    with urlopen(base + "/v1/models", timeout=10) as response:
        model = json.loads(response.read())["data"][0]["id"]
    completion(base, model, "Reply with READY.", 16)
    benchmarks = eval_suite.run_tasks(lambda prompt, max_tokens: completion(base, model, prompt, max_tokens))
    samples: list[dict[int, dict[str, float]]] = []
    stop = threading.Event()
    thread = threading.Thread(target=monitor, args=(stop, samples), daemon=True)
    thread.start(); started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(completion, base, model, PERF_PROMPT, 256, True)
                       for _ in range(args.concurrency)]
            perf_rows = [future.result() for future in futures]
    finally:
        elapsed = time.perf_counter() - started
        stop.set(); thread.join(timeout=2)
    output_tokens_per_request = [int(row.get("usage", {}).get("completion_tokens") or 0)
                                 for row in perf_rows]
    output_tokens = sum(output_tokens_per_request)
    telemetry = summarize(samples)
    metrics = ""
    try:
        with urlopen(base + "/metrics", timeout=10) as response:
            metrics = response.read().decode()
    except Exception:
        pass
    speculative = [line for line in metrics.splitlines()
                   if not line.startswith("#") and any(term in line.lower() for term in
                   ("spec_decode", "speculative", "acceptance", "accepted_tokens"))]
    def metric_total(name: str) -> float | None:
        match = re.search(rf"^{re.escape(name)}\{{[^\n]*\}} ([0-9.eE+-]+)$", metrics, re.MULTILINE)
        return float(match.group(1)) if match else None
    drafts = metric_total("vllm:spec_decode_num_drafts_total")
    drafted_tokens = metric_total("vllm:spec_decode_num_draft_tokens_total")
    accepted_tokens = metric_total("vllm:spec_decode_num_accepted_tokens_total")
    result = {
        "mode": args.mode, "model": model, "engine": "1Cat-vLLM 1.5.0", "tensor_parallel_size": 2,
        "concurrency": args.concurrency, "output_tokens": output_tokens,
        "output_tokens_per_request": output_tokens_per_request, "wall_seconds": round(elapsed, 4),
        "wall_output_tokens_per_second": round(output_tokens / elapsed, 4) if output_tokens else None,
        "benchmarks": benchmarks, "benchmark_mean_score": benchmarks["_mean_score"],
        "benchmark_task_count": len(eval_suite.TASKS), "gpu_telemetry": telemetry,
        "both_v100_active": all((telemetry[str(i)].get("max_util_pct") or 0) >= 10 for i in (1, 2)),
        "speculative_summary": {
            "rounds": drafts, "draft_tokens": drafted_tokens, "accepted_tokens": accepted_tokens,
            "acceptance_pct": round(100 * accepted_tokens / drafted_tokens, 4)
            if accepted_tokens is not None and drafted_tokens else None,
            "accepted_tokens_per_round": round(accepted_tokens / drafts, 4)
            if accepted_tokens is not None and drafts else None,
        },
        "speculative_metrics": speculative,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
