#!/usr/bin/env python3
"""Measure completion throughput from the locally served V100 vLLM endpoint."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from openai import OpenAI


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--url", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--requests", type=int, default=8); ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path("/media/knight2/EDS2/logs/vllm/tokens_per_second.json"))
    args = ap.parse_args(); client = OpenAI(base_url=args.url, api_key="local")
    rows = []
    for request in range(args.requests):
        start = time.perf_counter()
        result = client.completions.create(model=args.model, prompt="Explain why a chronological validation split prevents leakage in forecasting.",
                                           max_tokens=args.max_tokens, temperature=0.0)
        elapsed = time.perf_counter() - start
        tokens = result.usage.completion_tokens if result.usage else None
        rows.append({"request": request, "elapsed_seconds": elapsed, "completion_tokens": tokens,
                     "tokens_per_second": (tokens / elapsed if tokens else None)})
    valid = [row["tokens_per_second"] for row in rows if row["tokens_per_second"]]
    payload = {"model": args.model, "requests": rows, "mean_tokens_per_second": sum(valid) / len(valid) if valid else None}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
