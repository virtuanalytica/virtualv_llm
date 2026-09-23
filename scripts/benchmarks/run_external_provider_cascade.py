#!/usr/bin/env python3
"""Run the well-known + specialist (+ contamination-audit, once wired) suites
against cloud/agent-CLI providers -- Codex, Gemini/Antigravity, and (once the
broken ~/.claude.json symlink is fixed -- see NOTES.md) Claude -- using the
exact same well_known_suite.py entry point local GGUF models go through, so
every table in build_dual_v100_html.py treats these rows identically.

Two connection shapes:
  * "http": the provider has a real OpenAI-compatible endpoint (Gemini's
    /v1beta/openai layer). well_known_suite.py talks to it directly with the
    new --external-api-key-env/--completions-path support.
  * "cli": the provider is an agent CLI with no HTTP surface (codex exec,
    claude -p). This script starts external_cli_agent_proxy.py on a free
    local port first, points well_known_suite.py at that, and tears the
    proxy down afterwards -- one proxy process per model, sequential, so two
    concurrent CLI subprocess trees never fight over the same sandbox/session
    state.

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
PROXY = ROOT / "scripts/benchmarks/external_cli_agent_proxy.py"
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
        "kind": "http",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "completions_path": "/chat/completions", "api_key_env": "GEMINI_API_KEY",
        "engine": "Google Antigravity (Gemini API)",
        "models": ["gemini-3.6-flash", "gemini-3.1-pro-preview", "gemini-3.5-flash"],
        # Gemini 3's reasoning tokens share max_tokens with the visible answer
        # (confirmed 2026-09-23: truncated a correct short answer without
        # this). Only covers complete_text/complete_vision-based scoring
        # (humaneval, specialists, contamination_audit) -- see well_known_suite.py's
        # EXTRA_CHAT_BODY comment for what this does not cover yet.
        "extra_body_json": '{"reasoning_effort": "none"}',
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
    raise TimeoutError(f"external_cli_agent_proxy did not open port {port} within {timeout}s")


def run_one(row_name: str, provider_key: str, model: str, specialists: str) -> int:
    provider = PROVIDERS[provider_key]
    name = f"{row_name}-{model}"
    common = ["python3", str(WKS), name, "--out", str(REPORT),
              "--engine", provider["engine"], "--physical-gpus", "9",
              "--topology", f"cloud/agent-CLI provider; no local GPU ({provider_key})",
              "--specialists", specialists]

    if provider["kind"] == "http":
        cmd = common + ["--external-url", provider["base_url"], "--external-model", model,
                        "--completions-path", provider["completions_path"],
                        "--external-api-key-env", provider["api_key_env"]]
        if provider.get("extra_body_json"):
            cmd += ["--external-extra-body-json", provider["extra_body_json"]]
        return subprocess.run(cmd, cwd=ROOT).returncode

    port = free_port()
    proxy_proc = subprocess.Popen(
        ["python3", str(PROXY), "--backend", provider["backend"], "--model", model, "--port", str(port)],
        cwd=ROOT,
    )
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
