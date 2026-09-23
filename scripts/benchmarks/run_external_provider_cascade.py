#!/usr/bin/env python3
"""Run the well-known + specialist (+ contamination-audit, once wired) suites
against cloud/agent-CLI providers -- Codex, Gemini/Antigravity, and (once the
broken ~/.claude.json symlink is fixed -- see NOTES.md) Claude -- using the
exact same well_known_suite.py entry point local GGUF models go through, so
every table in build_dual_v100_html.py treats these rows identically.

Two connection shapes, both proxied through a local server this script starts
per model and tears down afterward:
  * "cli": the provider is an agent CLI with no HTTP surface (codex exec,
    claude -p). external_cli_agent_proxy.py shells out to it per request.
  * "sanitizing_http": the provider has a real OpenAI-compatible endpoint,
    but is stricter than lm-eval's LocalChatCompletion model assumes.
    Confirmed 2026-09-23: Gemini's /v1beta/openai layer 400s on the "seed"
    field lm-eval always sends, and on a null entry inside "stop" (sent when
    a task has no explicit stop sequence) -- this broke gsm8k/bbh/mmlu_sample/
    truthfulqa_gen (all routed through lm-eval) while complete_text-based
    tasks (humaneval, specialists) worked fine going straight to the real
    endpoint. sanitizing_proxy.py strips the fields the target rejects and
    injects reasoning_effort=none universally (previously only applied to
    complete_text/complete_vision via well_known_suite.py's own
    --external-extra-body-json, missing the lm-eval-routed tasks entirely).
    A provider with a genuinely compliant endpoint would use well_known_suite.py's
    --external-api-key-env/--completions-path directly instead -- this kind
    exists specifically for endpoints that need request massaging.

Z.ai has no key or CLI installed on this machine (checked 2026-09-23) -- not
included here; add a PROVIDERS entry once credentials exist.
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WKS = ROOT / "scripts/benchmarks/well_known_suite.py"
CLI_PROXY = ROOT / "scripts/benchmarks/external_cli_agent_proxy.py"
SANITIZING_PROXY = ROOT / "scripts/benchmarks/sanitizing_proxy.py"
REPORT = ROOT / "reports/well_known_suite_20260917.json"

# Deliberately a small representative slice per provider, not the full model
# list (9 Codex slugs / ~30 Gemini slugs) -- each run costs real cloud spend
# and/or agent-CLI wall time, unlike the sunk local-GPU-electricity cost of
# the GGUF sweep. Widen this once these rows are reviewed and more coverage
# is explicitly wanted.
PROVIDERS = {
    "codex": {
        "kind": "cli", "backend": "codex", "engine": "OpenAI Codex CLI",
        "models": ["gpt-6-astra", "gpt-6-sol", "gpt-6-luna"],
    },
    "antigravity-gemini": {
        "kind": "sanitizing_http",
        "target_base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "target_path": "/chat/completions", "api_key_env": "GEMINI_API_KEY",
        "strip_fields": "seed", "inject_fields_json": '{"reasoning_effort": "none"}',
        "engine": "Google Antigravity (Gemini API)",
        "models": ["gemini-3.6-flash", "gemini-3.1-pro-preview", "gemini-3.5-flash"],
    },
}


def free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_proxy(port: int, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.3)
    raise TimeoutError(f"proxy did not open port {port} within {timeout}s")


def proxy_command(provider: dict, model: str, port: int) -> list[str]:
    if provider["kind"] == "cli":
        return ["python3", str(CLI_PROXY), "--backend", provider["backend"], "--model", model,
               "--port", str(port)]
    if provider["kind"] == "sanitizing_http":
        cmd = ["python3", str(SANITIZING_PROXY), "--port", str(port),
              "--target-base-url", provider["target_base_url"],
              "--target-path", provider["target_path"],
              "--api-key-env", provider["api_key_env"]]
        if provider.get("strip_fields"):
            cmd += ["--strip-fields", provider["strip_fields"]]
        if provider.get("inject_fields_json"):
            cmd += ["--inject-fields-json", provider["inject_fields_json"]]
        return cmd
    raise ValueError(f"unknown provider kind: {provider['kind']}")


def run_one(row_name: str, provider_key: str, model: str, specialists: str) -> int:
    provider = PROVIDERS[provider_key]
    name = f"{row_name}-{model}"
    common = ["python3", str(WKS), name, "--out", str(REPORT),
              "--engine", provider["engine"], "--physical-gpus", "9",
              "--topology", f"cloud/agent-CLI provider; no local GPU ({provider_key})",
              "--specialists", specialists]

    port = free_port()
    proxy_proc = subprocess.Popen(proxy_command(provider, model, port), cwd=ROOT)
    try:
        wait_for_proxy(port)
        cmd = common + ["--external-url", f"http://127.0.0.1:{port}", "--external-model", model]
        return subprocess.run(cmd, cwd=ROOT).returncode
    finally:
        proxy_proc.terminate()
        try:
            proxy_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proxy_proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=sorted(PROVIDERS), action="append",
                        help="repeatable; default: all configured providers")
    parser.add_argument("--specialists", default="all")
    parser.add_argument("--row-prefix", default="cloud")
    args = parser.parse_args()
    providers = args.provider or sorted(PROVIDERS)

    failures = 0
    for provider_key in providers:
        for model in PROVIDERS[provider_key]["models"]:
            row_name = f"{args.row_prefix}-{provider_key}"
            print(f"=== {provider_key} / {model} ===", flush=True)
            rc = run_one(row_name, provider_key, model, args.specialists)
            if rc != 0:
                failures += 1
                print(f"FAILED: {provider_key}/{model} (rc={rc})", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
