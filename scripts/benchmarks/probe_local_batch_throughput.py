#!/usr/bin/env python3
"""Public B1/B4/B16 capacity probe for a local OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

PROMPT = (
    "Explain sorted merge join on two large tables in about 180 words. "
    "Include one example with join key k={key}, then name one limitation."
)


async def one_request(base: str, alias: str, case: int) -> dict:
    import urllib.error
    import urllib.request

    body = {"model": alias, "messages": [{"role": "user", "content": PROMPT.format(key=case+10)}],
            "temperature": 0, "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"}}
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})

    def perform() -> dict:
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=180) as reply:
                payload = json.load(reply)
            tokens = (payload.get("usage") or {}).get("completion_tokens")
            return {"case": case, "status": "ok" if isinstance(tokens, int) else "missing_usage",
                    "completion_tokens": tokens, "wall_seconds": round(time.monotonic()-started, 3),
                    "finish_reason": (payload.get("choices") or [{}])[0].get("finish_reason")}
        except urllib.error.HTTPError as exc:
            return {"case": case, "status": "http_error", "http_status": exc.code,
                    "wall_seconds": round(time.monotonic()-started, 3)}
        except Exception as exc:
            return {"case": case, "status": type(exc).__name__,
                    "wall_seconds": round(time.monotonic()-started, 3)}

    return await asyncio.to_thread(perform)


async def run(args: argparse.Namespace) -> dict:
    waves = []
    case = 1
    for batch in (1, 4, 16):
        for repeat in range(args.repeats + 1):
            started = time.monotonic()
            rows = await asyncio.gather(*(one_request(args.base, args.alias, i)
                                          for i in range(case, case + batch)))
            wall = time.monotonic()-started
            waves.append({"batch": batch, "repeat": repeat,
                          "warmup": repeat == 0, "wall_seconds": round(wall, 3),
                          "requests": rows})
            print(json.dumps({"batch": batch, "repeat": repeat, "warmup": repeat == 0,
                              "ok": sum(row["status"] == "ok" for row in rows),
                              "wall_seconds": round(wall, 3)}), flush=True)
            case += batch
    summaries = {}
    for batch in (1, 4, 16):
        selected = [wave for wave in waves if wave["batch"] == batch and not wave["warmup"]]
        good = [row for wave in selected for row in wave["requests"] if row["status"] == "ok"]
        wall = sum(wave["wall_seconds"] for wave in selected)
        complete = len(good) == batch * args.repeats
        summaries[str(batch)] = {
            "status": "complete" if complete else "partial",
            "requests": len(good), "expected_requests": batch * args.repeats,
            "completion_tokens": sum(row["completion_tokens"] for row in good),
            "wall_seconds": round(wall, 3),
            "aggregate_completion_tokens_per_wall_second":
                round(sum(row["completion_tokens"] for row in good) / wall, 3) if complete and wall else None,
            "failures": [row for wave in selected for row in wave["requests"] if row["status"] != "ok"],
        }
    return {"at": datetime.now(timezone.utc).isoformat(), "model": args.model_id,
            "alias": args.alias, "endpoint": args.base, "protocol": "public-synthetic-batch-throughput-v1",
            "unit": "aggregate completion tokens per full wave wall second including prefill; not pure decode",
            "decode_profile": {"temperature": 0, "max_tokens": 256, "enable_thinking": False,
                               "reasoning_effort": "low"},
            "summaries": summaries, "waves": waves}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    url = urlparse(args.base)
    if url.scheme != "http" or url.hostname != "127.0.0.1":
        parser.error("endpoint must be local 127.0.0.1")
    if args.repeats < 1:
        parser.error("repeats must be positive")
    report = asyncio.run(run(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summaries"]), flush=True)


if __name__ == "__main__":
    main()
