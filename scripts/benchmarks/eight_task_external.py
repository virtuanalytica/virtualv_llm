#!/usr/bin/env python3
"""Run the shared 8-task battery (eval_suite.TASKS) against an already running OpenAI-compatible endpoint.

benchmark_local_gguf_tp2.py starts its own llama.cpp server per model; this runner covers endpoints it
cannot start itself, such as a live mixture-of-models proxy or a model served with expert offload. Same
prompts, same deterministic scoring, temperature 0. Rows are upserted (keyed by model name) into
reports/eight_task_external_20261005.json, which the dashboard merges into the 8-task table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_suite  # noqa: E402
from result_store import upsert_result  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "reports" / "eight_task_external_20261005.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="report row name (same name as in well_known_suite)")
    ap.add_argument("--external-url", required=True)
    ap.add_argument("--external-model", required=True, help="model alias the endpoint expects")
    ap.add_argument("--topology", default="")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    base = args.external_url.rstrip("/")

    def complete(prompt: str, max_tokens: int) -> dict:
        body = json.dumps({"model": args.external_model, "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": max_tokens, "temperature": 0}).encode()
        req = Request(f"{base}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=args.timeout) as response:
            return json.loads(response.read())

    try:  # preflight: never publish a row for a dead endpoint
        complete("Reply with the single word OK.", 16)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"endpoint preflight failed for {base}: {exc}")
    started = time.time()
    results = eval_suite.run_tasks(complete)
    row = {"model": args.model, "external_base_url": base, "external_model_alias": args.external_model,
           "topology": args.topology, "benchmarks": results, "benchmark_mean_score": results["_mean_score"],
           "runtime_sec": round(time.time() - started, 1), "at": datetime.now(timezone.utc).isoformat()}
    upsert_result(args.out, row, default={"suite": "eval_suite 8-task battery (external endpoints)", "results": []})
    print(f"DONE {args.model}: mean={results['_mean_score']} "
          + " ".join(f"{k}={v['score']}" for k, v in results.items() if k != "_mean_score"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
