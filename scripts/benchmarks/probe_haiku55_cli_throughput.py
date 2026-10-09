#!/usr/bin/env python3
"""Public B1/B4/B16 Claude CLI throughput probe; no sealed evaluation items.

Output tokens include thinking tokens. Visible tokens are estimated as output
tokens minus thinking tokens. Wall time includes CLI startup, prefill and API
latency, so neither rate is a server-side pure decode measurement.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

PROMPT = (
    "Explain a sorted merge join of two large sorted database tables in about "
    "180 words. Include a small concrete example and one limitation. "
    "Use example case {case} with join key k={key}."
)


async def one_request(case: int) -> dict:
    env = {key: value for key, value in os.environ.items()
           if not (key.startswith("CLAUDE_CODE_") or key == "CLAUDECODE")}
    with tempfile.TemporaryDirectory(prefix="haiku55-throughput-") as cwd:
        started = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            "claude", "-p", "--model", "claude-haiku-5-5",
            "--setting-sources", "", "--strict-mcp-config",
            "--no-session-persistence", "--tools", "",
            "--output-format", "json", "--effort", "medium",
            "--system-prompt", "Answer directly.",
            PROMPT.format(case=case, key=case + 10),
            cwd=cwd, env=env, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            return {"case": case, "status": "timeout", "wall_seconds": round(time.monotonic()-started, 3)}
        wall = time.monotonic() - started
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return {"case": case, "status": "bad_json", "returncode": proc.returncode,
                "wall_seconds": round(wall, 3), "stderr_type": "nonempty" if stderr else "empty"}
    usage = (data.get("modelUsage") or {}).get("claude-haiku-5-5") or {}
    output = usage.get("outputTokens")
    thinking = usage.get("thinkingTokens")
    status = "ok" if proc.returncode == 0 and not data.get("is_error") and isinstance(output, int) else "error"
    return {"case": case, "status": status, "returncode": proc.returncode,
            "error_subtype": data.get("subtype") if status != "ok" else None,
            "api_error_status": data.get("api_error_status") if status != "ok" else None,
            "terminal_reason": data.get("terminal_reason") if status != "ok" else None,
            "wall_seconds": round(wall, 3),
            "api_seconds": round((data.get("duration_api_ms") or 0) / 1000, 3),
            "output_tokens": output, "thinking_tokens": thinking,
            "visible_tokens_estimate": output - thinking if isinstance(output, int) and isinstance(thinking, int) else None,
            "input_tokens": usage.get("inputTokens"),
            "cost_usd": usage.get("costUSD"), "result_characters": len(data.get("result") or "")}


async def run(args: argparse.Namespace) -> dict:
    waves = []
    case = 1
    for batch, repeats in ((1, args.b1_repeats), (4, args.b4_repeats), (16, args.b16_repeats)):
        for repeat in range(repeats):
            started = time.monotonic()
            rows = await asyncio.gather(*(one_request(i) for i in range(case, case + batch)))
            wall = time.monotonic() - started
            waves.append({"batch": batch, "repeat": repeat + 1,
                          "wall_seconds": round(wall, 3), "requests": rows})
            print(json.dumps({"batch": batch, "repeat": repeat + 1,
                              "wall_seconds": round(wall, 3),
                              "ok": sum(row["status"] == "ok" for row in rows)}), flush=True)
            case += batch
    summaries = {}
    for batch in (1, 4, 16):
        selected = [wave for wave in waves if wave["batch"] == batch]
        good = [row for wave in selected for row in wave["requests"] if row["status"] == "ok"]
        total = sum(wave["wall_seconds"] for wave in selected)
        complete = len(good) == batch * len(selected)
        output_rate = round(sum(row["output_tokens"] for row in good) / total, 3) if total else None
        visible_rate = round(sum(row["visible_tokens_estimate"] or 0 for row in good) / total, 3) if total else None
        summaries[str(batch)] = {
            "status": "complete" if complete else "partial",
            "waves": len(selected), "requests": len(good), "expected_requests": batch * len(selected),
            "output_tokens": sum(row["output_tokens"] for row in good),
            "thinking_tokens": sum(row["thinking_tokens"] or 0 for row in good),
            "visible_tokens_estimate": sum(row["visible_tokens_estimate"] or 0 for row in good),
            "wall_seconds": round(total, 3),
            "aggregate_output_tokens_per_wall_second": output_rate if complete else None,
            "aggregate_visible_tokens_per_wall_second_estimate": visible_rate if complete else None,
            "successful_only_output_tokens_per_wall_second": output_rate if not complete else None,
            "successful_only_visible_tokens_per_wall_second_estimate": visible_rate if not complete else None,
            "http_429_failures": sum(row.get("api_error_status") == 429 for wave in selected for row in wave["requests"]),
            "median_request_wall_seconds": round(statistics.median(row["wall_seconds"] for row in good), 3) if good else None,
            "cost_usd": round(sum(row["cost_usd"] or 0 for row in good), 6),
        }
    return {"at": datetime.now(timezone.utc).isoformat(), "model": "claude-haiku-5-5",
            "effort": "medium", "thinking": "adaptive", "prompt": "public synthetic sorted merge join (~180 words)",
            "unit": "aggregate completion tokens per wall second; includes CLI startup, prefill and thinking",
            "scope": "synthetic throughput probe, not sealed quality benchmark or pure decode",
            "summaries": summaries, "waves": waves}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--b1-repeats", type=int, default=3)
    parser.add_argument("--b4-repeats", type=int, default=2)
    parser.add_argument("--b16-repeats", type=int, default=2)
    args = parser.parse_args()
    report = asyncio.run(run(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summaries"]), flush=True)


if __name__ == "__main__":
    main()
