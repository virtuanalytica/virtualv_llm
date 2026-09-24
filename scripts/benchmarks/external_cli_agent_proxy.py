#!/usr/bin/env python3
"""Local OpenAI-compatible proxy in front of an agent CLI (codex exec, claude -p).

Neither Codex nor Claude Code exposes a raw chat-completions HTTP endpoint --
both are interactive-first CLIs with a non-interactive "print one answer and
exit" mode instead. well_known_suite.py, specialist_suite.py and
contamination_audit.py all talk to models exclusively through an
OpenAI-compatible /v1/chat/completions (and lm-eval's local-chat-completions
model type does the same). Rather than teach three separate suites a second,
CLI-subprocess code path, this proxy translates one into the other: it starts
a small HTTP server on 127.0.0.1, and each POST /v1/chat/completions request
is answered by shelling out to the configured CLI once, synchronously, and
wrapping its final answer in the standard choices[0].message.content shape.

Sandboxing: every CLI call runs with the most restrictive read-only/no-tools
mode each agent supports. A benchmark prompt (HumanEval task text, a GSM8K
word problem, etc.) must never be able to trigger a real file write or shell
command as a side effect of being asked to "explain" or "solve" it.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# Wall-clock ceiling per single completion. Codex/Claude agent CLIs carry far
# more startup overhead (session init, tool registration) than a raw chat
# endpoint -- observed 5-20s for a trivial prompt even with tools disabled.
# lm-eval's default per-request timeout is longer than this, so a genuine
# hang still surfaces as a clean proxy-side error instead of dangling.
CLI_TIMEOUT_SEC = 180


def _extract_prompt(messages: list[dict[str, Any]]) -> str:
    """Collapse an OpenAI chat message list into one plain-text prompt.

    Every caller in this repo sends a single user message (no system/multi-
    turn chat history) -- well_known_suite.py's complete_text, lm-eval's
    local-chat-completions with a single doc per call, and specialist_suite's
    custom-pack prompts are all one-shot. Concatenating any additional roles
    defensively (rather than raising) keeps this proxy usable if that ever
    changes, at the cost of losing role structure the CLI would ignore anyway.
    """
    parts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):  # OpenAI multi-part content (text/image blocks)
            content = " ".join(block.get("text", "") for block in content if isinstance(block, dict))
        parts.append(str(content or ""))
    return "\n\n".join(parts)


def complete_via_codex(prompt: str, model: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        out_file = Path(tmp) / "last_message.txt"
        proc = subprocess.run(
            ["codex", "exec", "-m", model, "-s", "read-only", "--skip-git-repo-check",
             "-o", str(out_file), "-"],
            input=prompt, text=True, capture_output=True, timeout=CLI_TIMEOUT_SEC,
        )
        if out_file.exists() and out_file.read_text().strip():
            return out_file.read_text()
        raise RuntimeError(f"codex exec produced no output (rc={proc.returncode}): "
                           f"{(proc.stderr or proc.stdout)[-500:]}")


CLAUDE_BENCH_SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's question directly."
# Last per-request accounting from the claude backend, returned in the HTTP
# response. Handler threads each run one request at a time per backend call,
# so the value is read back right after the call on the same thread.
_claude_usage = threading.local()


def _isolated_claude_env() -> dict[str, str]:
    # A parent Claude Code session exports CLAUDE_CODE_* / CLAUDECODE; a child
    # inheriting them attaches to the parent's messaging socket instead of
    # running as an independent, stateless benchmark call.
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("CLAUDE_CODE_") or k == "CLAUDECODE")}
    return env


def complete_via_claude_cli(prompt: str, model: str) -> str:
    # --bare would be the ideal isolation but only accepts ANTHROPIC_API_KEY
    # (never OAuth), so the same effect is built from parts: no settings
    # sources (so no user hooks such as an autonomous Stop-hook loop), no
    # CLAUDE.md (the user's global instructions would otherwise steer
    # language/format), no MCP servers, no tools, no persisted session, and an
    # empty cwd. Verified 2026-09-24: 314 input tokens for a one-line prompt,
    # model reports no CLAUDE.md/user instructions.
    # max_tokens is NOT forwarded: CLAUDE_CODE_MAX_OUTPUT_TOKENS turns reaching
    # the budget into an API error with no partial text (checked 2026-09-24),
    # so Claude rows run uncapped while local rows get the request budget.
    with tempfile.TemporaryDirectory() as cwd:
        proc = subprocess.run(
            ["claude", "-p", "--model", model, "--setting-sources", "", "--strict-mcp-config",
             "--no-session-persistence", "--tools", "", "--output-format", "json",
             "--system-prompt", CLAUDE_BENCH_SYSTEM_PROMPT, prompt],
            text=True, capture_output=True, timeout=CLI_TIMEOUT_SEC, cwd=cwd, env=_isolated_claude_env(),
        )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        data = {}
    if proc.returncode == 0 and not data.get("is_error") and str(data.get("result", "")).strip():
        usage = data.get("usage") or {}
        _claude_usage.value = {
            "prompt_tokens": int(usage.get("input_tokens") or 0),
            "completion_tokens": int(usage.get("output_tokens") or 0),
            "duration_api_ms": data.get("duration_api_ms"),
            # ttft_ms lands at ~the full API duration (not first token); the
            # streamed first-token/first-content timestamps are the real split
            # between prefill and decode (checked 2026-09-24: 1041 of 1658 ms).
            "ttft_ms": data.get("ttft_stream_ms") or data.get("first_content_frame_ms"),
            "cost_usd": data.get("total_cost_usd"),
        }
        return str(data["result"])
    raise RuntimeError(f"claude -p failed (rc={proc.returncode}): {(proc.stderr or proc.stdout)[-500:]}")


def _timings(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    """llama.cpp-style timings from real CLI accounting (API-side decode, not local)."""
    if not usage or not usage.get("duration_api_ms") or not usage.get("ttft_ms"):
        return None
    decode_sec = (usage["duration_api_ms"] - usage["ttft_ms"]) / 1000
    prefill_sec = usage["ttft_ms"] / 1000
    return {
        "predicted_n": usage["completion_tokens"],
        "predicted_per_second": round(usage["completion_tokens"] / decode_sec, 2) if decode_sec > 0 else None,
        "prompt_n": usage["prompt_tokens"],
        "prompt_per_second": round(usage["prompt_tokens"] / prefill_sec, 2) if prefill_sec > 0 else None,
        "source": "claude -p --output-format json (API decode after first token; network included)",
    }


BACKENDS = {"codex": complete_via_codex, "claude-cli": complete_via_claude_cli}


class Handler(BaseHTTPRequestHandler):
    backend_name: str = ""
    default_model: str = ""

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet; caller's own log captures stdout
        pass

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler naming)
        chat = self.path in ("/v1/chat/completions", "/chat/completions")
        # Legacy /v1/completions exists only so well_known_suite.py's throughput
        # probe gets llama.cpp-style "timings"; the prompt is still answered as one chat turn.
        if not chat and not (self.path == "/v1/completions" and self.backend_name == "claude-cli"):
            self._send_json(404, {"error": f"unsupported path {self.path}; this proxy only implements "
                                            "chat completions (legacy /v1/completions: claude-cli only)"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"invalid JSON body: {exc}"})
            return
        # The alias well_known_suite.py sends (MODEL_ALIAS) is not a real model slug;
        # the proxy is bound to exactly one model at startup.
        model = self.default_model
        prompt = str(payload.get("prompt", "")) if not chat else _extract_prompt(payload.get("messages", []))
        started = time.time()
        _claude_usage.value = None
        try:
            content = BACKENDS[self.backend_name](prompt, model)
        except Exception as exc:
            self._send_json(502, {"error": f"{type(exc).__name__}: {exc}"})
            return
        elapsed = time.time() - started
        # Real accounting only exists for claude-cli (--output-format json); codex stays zero.
        usage = getattr(_claude_usage, "value", None)
        tokens = {"prompt_tokens": usage["prompt_tokens"] if usage else 0,
                  "completion_tokens": usage["completion_tokens"] if usage else 0}
        tokens["total_tokens"] = tokens["prompt_tokens"] + tokens["completion_tokens"]
        body: dict[str, Any] = {
            "id": f"cli-proxy-{int(started)}",
            "object": "chat.completion" if chat else "text_completion",
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop"} if chat else
                        {"index": 0, "text": content, "finish_reason": "stop"}],
            "usage": tokens,
            "cli_proxy_elapsed_sec": round(elapsed, 2),
        }
        if usage:
            body["cost_usd"] = usage["cost_usd"]
            timings = _timings(usage)
            if timings:
                body["timings"] = timings
        self._send_json(200, body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    parser.add_argument("--model", required=True, help="Model slug passed through to the backend CLI")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    for name in ("codex", "claude"):
        if name in args.backend and shutil.which(name) is None:
            raise SystemExit(f"{name} CLI not found on PATH")

    handler = type("BoundHandler", (Handler,), {"backend_name": args.backend, "default_model": args.model})
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"external_cli_agent_proxy: backend={args.backend} model={args.model} "
          f"listening on 127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
