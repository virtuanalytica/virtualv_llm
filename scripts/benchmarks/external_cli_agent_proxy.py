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


def complete_via_claude_cli(prompt: str, model: str) -> str:
    # --disallowed-tools blocks every tool this repo's Claude Code install
    # knows about, so a benchmark prompt can only ever produce text, never a
    # real file/shell side effect. --permission-mode default (not "plan")
    # avoids the plan-mode detour seen when testing this integration.
    proc = subprocess.run(
        ["claude", "-p", "--model", model,
         "--disallowed-tools", "Bash", "Read", "Write", "Edit", "NotebookEdit",
         "WebFetch", "WebSearch", "Agent", "Artifact", "Skill",
         prompt],
        text=True, capture_output=True, timeout=CLI_TIMEOUT_SEC,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout
    raise RuntimeError(f"claude -p failed (rc={proc.returncode}): {(proc.stderr or proc.stdout)[-500:]}")


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
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            self._send_json(404, {"error": f"unsupported path {self.path}; this proxy only implements "
                                            "chat completions (no legacy /v1/completions)"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"invalid JSON body: {exc}"})
            return
        model = payload.get("model") or self.default_model
        prompt = _extract_prompt(payload.get("messages", []))
        started = time.time()
        try:
            content = BACKENDS[self.backend_name](prompt, model)
        except Exception as exc:
            self._send_json(502, {"error": f"{type(exc).__name__}: {exc}"})
            return
        elapsed = time.time() - started
        self._send_json(200, {
            "id": f"cli-proxy-{int(started)}",
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop"}],
            # Not real token accounting -- these CLIs don't expose it uniformly. Downstream
            # composite scoring never reads usage, only choices[0].message.content.
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "cli_proxy_elapsed_sec": round(elapsed, 2),
        })


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
