#!/usr/bin/env python3
"""Local OpenAI-compatible proxy that strips fields a strict remote endpoint rejects.

lm-eval's LocalChatCompletion model (used by well_known_suite.py's
run_lm_eval_task for gsm8k/bbh/mmlu_sample/truthfulqa_gen) always adds a top-
level "seed" field and a "stop" array to every request body, regardless of
model_args. Most OpenAI-compatible servers (llama-server, vLLM, the Codex/
Claude proxy in this repo) silently ignore fields they don't recognize.
Gemini's OpenAI-compat layer does not: it 400s on any unknown top-level field
("seed") and on a non-string entry inside "stop" (lm-eval sends [None] when a
task has no explicit stop sequence -- confirmed via curl 2026-09-23).

This proxy sits between well_known_suite.py/lm-eval and the real remote
endpoint: it receives the exact request lm-eval or complete_text would have
sent to that endpoint directly, removes the fields the target is known to
reject, and forwards the rest with the real Authorization header. Nothing
about the scoring logic changes -- this only removes noise the target never
needed to see in the first place.
"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import HTTPError


def sanitize(payload: dict[str, Any], strip_fields: set[str], inject_fields: dict[str, Any]) -> dict[str, Any]:
    cleaned = {k: v for k, v in payload.items() if k not in strip_fields}
    stop = cleaned.get("stop")
    if isinstance(stop, list):
        stop = [s for s in stop if isinstance(s, str)]
        if stop:
            cleaned["stop"] = stop
        else:
            cleaned.pop("stop", None)
    elif stop is None:
        cleaned.pop("stop", None)
    # Applied after stripping so an injected field can't be immediately
    # removed by an overlapping --strip-fields entry, and after the request
    # already has its real content -- this only adds provider-required
    # defaults (e.g. Gemini 3's reasoning_effort), never overrides caller intent.
    for key, value in inject_fields.items():
        cleaned.setdefault(key, value)
    return cleaned


class Handler(BaseHTTPRequestHandler):
    target_base_url: str = ""
    target_path: str = "/chat/completions"
    api_key: str = ""
    strip_fields: set[str] = set()
    inject_fields: dict[str, Any] = {}

    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"invalid JSON body: {exc}"})
            return
        cleaned = sanitize(payload, self.strip_fields, self.inject_fields)
        body = json.dumps(cleaned).encode()
        req = Request(f"{self.target_base_url}{self.target_path}", data=body,
                      headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        try:
            with urlopen(req, timeout=240) as resp:
                self._send_raw(resp.status, resp.read())
        except HTTPError as exc:
            self._send_raw(exc.code, exc.read())

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        self._send_raw(status, json.dumps(payload).encode())

    def _send_raw(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--target-base-url", required=True)
    parser.add_argument("--target-path", default="/chat/completions")
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--strip-fields", default="seed", help="comma-separated top-level field names to drop")
    parser.add_argument("--inject-fields-json", default="{}",
                        help='JSON object merged into every request that doesn\'t already set that '
                             'key, e.g. \'{"reasoning_effort": "none"}\'')
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"--api-key-env {args.api_key_env} is unset or empty")

    handler = type("BoundHandler", (Handler,), {
        "target_base_url": args.target_base_url.rstrip("/"), "target_path": args.target_path,
        "api_key": api_key, "strip_fields": {f.strip() for f in args.strip_fields.split(",") if f.strip()},
        "inject_fields": json.loads(args.inject_fields_json),
    })
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"sanitizing_proxy: target={args.target_base_url}{args.target_path} "
          f"strip={handler.strip_fields} listening on 127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
