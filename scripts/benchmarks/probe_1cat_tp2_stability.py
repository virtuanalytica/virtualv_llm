#!/usr/bin/env python3
"""Cheap, public diagnostic for Qwen3.8 1Cat TP2 repetition loops.

These fixed prompts are not a benchmark and never enter a composite. A
failed diagnostic does not cancel the full private measurement.
"""

from __future__ import annotations

import argparse
import ast
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


CANARIES = (
    ("arithmetic", "Compute 19 * 23. Reply with only the integer.", "437"),
    ("json", 'Return only JSON: an object with alpha=2+3 and beta=7*8.', {"alpha": 5, "beta": 56}),
    ("code", "Write only Python code defining solve(payload), which returns payload['x'] * 2.", "solve"),
)


def answer_ok(kind: str, answer: str, expected: object) -> bool:
    value = answer.strip()
    if not value or len(value) > 1500:
        return False
    if kind == "arithmetic":
        return value == expected
    if kind == "json":
        try:
            return json.loads(value) == expected
        except ValueError:
            return False
    if value.startswith("```python") and value.endswith("```"):
        value = value[len("```python"):-3].strip()
    try:
        tree = ast.parse(value)
    except SyntaxError:
        return False
    return len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef) and tree.body[0].name == expected


def has_repetition_loop(answer: str) -> bool:
    words = answer.lower().split()
    grams = [tuple(words[index:index + 6]) for index in range(max(0, len(words) - 5))]
    return any(grams.count(gram) >= 3 for gram in set(grams))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.base.startswith("http://127.0.0.1:"):
        raise ValueError("probe only accepts a local endpoint")
    observations = []
    for kind, prompt, expected in CANARIES:
        for repetition in range(3):
            body = {"model": args.alias, "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0, "max_tokens": 256,
                    "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"}}
            request = Request(args.base.rstrip("/") + "/v1/chat/completions",
                              data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            started = time.monotonic()
            try:
                with urlopen(request, timeout=180) as response:
                    data = json.load(response)
                choice = data["choices"][0]
                answer = choice["message"].get("content") or ""
                looping = has_repetition_loop(answer)
                ok = not looping and answer_ok(kind, answer, expected)
                reason = "repetition_loop" if looping else "ok" if ok else "incorrect_empty_or_malformed"
                finish = choice.get("finish_reason")
                output_tokens = (data.get("usage") or {}).get("completion_tokens")
            except Exception as exc:
                ok, reason, finish, output_tokens = False, type(exc).__name__, None, None
            observations.append({"canary": kind, "repetition": repetition + 1,
                                 "passed": ok, "reason": reason, "finish_reason": finish,
                                 "output_tokens": output_tokens,
                                 "latency_seconds": round(time.monotonic() - started, 3)})
    result = {"at": datetime.now(timezone.utc).isoformat(), "model_alias": args.alias,
              "endpoint": args.base, "purpose": "stability_preflight_not_benchmark",
              "status": "passed" if all(row["passed"] for row in observations) else "failed",
              "observations": observations}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "passed": sum(row["passed"] for row in observations),
                      "total": len(observations)}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
