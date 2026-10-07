#!/usr/bin/env python3
"""Run the well-known + specialist (+ contamination-audit, once wired) suites
against cloud/agent-CLI providers -- Codex, Gemini/Antigravity, and (wired
since 2026-09-24) Claude -- using the
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
import json
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
    # Claude has no downloadable weights; it is driven through the scripted
    # `claude -p` CLI (OAuth subscription, no API key on this box), isolated
    # from the user's hooks/CLAUDE.md -- see external_cli_agent_proxy.py.
    # Ordered newest/current-session model first; run one with --model.
    "claude": {
        "kind": "cli", "backend": "claude-cli", "engine": "Anthropic Claude Code CLI (claude -p)",
        "models": ["claude-opus-5-5", "claude-fable-5-1", "claude-sonnet-5-5", "claude-sonnet-5",
                   "claude-haiku-4-5-20251001"],
    },
    "codex": {
        "kind": "cli", "backend": "codex", "engine": "OpenAI Codex CLI",
        "models": ["gpt-6-astra", "gpt-6-sol", "gpt-6-luna"],
    },
    # Z.ai ZCode's headless CLI serves whichever model its own config names
    # (no model flag), so the slug below is a label the operator must keep in
    # step with that config. Needs VIRTUALV_ZCODE_CLI; see the proxy.
    "zcode": {
        "kind": "cli", "backend": "zcode", "engine": "Z.ai ZCode CLI (zcode --prompt, plan mode, no tools)",
        "models": ["glm-5.3-flash"],
    },
    # omp fronts many providers with one CLI and one accounting format. The
    # key names the omp provider; a model is sent as "<omp_provider>/<slug>".
    # Only slugs that answered a smoke prompt are listed (2026-10-07).
    "omp-zai": {
        "kind": "cli", "backend": "omp", "omp_provider": "zai",
        "engine": "omp CLI (omp -p, no tools) via Z.ai",
        "models": ["glm-5.3-flash", "glm-5.3"],
    },
    "omp-openai-codex": {
        "kind": "cli", "backend": "omp", "omp_provider": "openai-codex",
        "engine": "omp CLI (omp -p, no tools) via OpenAI Codex subscription",
        "models": ["gpt-6-luna", "gpt-6-astra", "gpt-6-sol"],
    },
    "omp-google": {
        "kind": "cli", "backend": "omp", "omp_provider": "google",
        "engine": "omp CLI (omp -p, no tools) via Google Gemini API",
        "models": ["gemini-3.8-flash"],
    },
    # Not listed: github-copilot (every model answered 400 "not supported" on
    # this subscription) and google-antigravity (omp: unhandled API mapping).
    "antigravity-gemini": {
        "kind": "sanitizing_http",
        "target_base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "target_path": "/chat/completions", "api_key_env": "GEMINI_API_KEY",
        "strip_fields": "seed", "inject_fields_json": '{"reasoning_effort": "none"}',
        "engine": "Google Antigravity (Gemini API)",
        # 2026-09-23: "pro" model variants (e.g. gemini-3.1-pro-preview) 400 on
        # reasoning_effort=none -- "Budget 0 is invalid. This model only works
        # in thinking mode." "low" fixes that but then eats the same small
        # max_tokens budget the "none" fix was for in the first place (flash
        # models tested empty/wrong at max_tokens=50 with "low"). Rather than
        # inject_fields per-model, stick to "flash" variants here, confirmed
        # to work with reasoning_effort=none at well_known_suite.py's normal
        # token budgets: gemini-3.6-flash (0.96/0.975/0.925 gsm8k/humaneval/
        # mmlu) and gemini-3.5-flash (0.90-0.94/0.975/0.9125) both verified.
        "models": ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.8-flash"],
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


def proxy_command(provider: dict, model: str, port: int, usage_log: Path | None = None) -> list[str]:
    if provider["kind"] == "cli":
        if provider.get("omp_provider"):
            model = f"{provider['omp_provider']}/{model}"
        cmd = ["python3", str(CLI_PROXY), "--backend", provider["backend"], "--model", model,
               "--port", str(port)]
        return cmd + (["--usage-log", str(usage_log)] if usage_log else [])
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


def usage_log_path(name: str) -> Path:
    return ROOT / "reports/benchmark_logs" / f"{name}_cli_usage.jsonl"


def record_cli_usage(name: str) -> None:
    """Sum the proxy's per-request accounting into the finished row.

    Also states that the row ran without the per-request output budget local
    models get (agent CLIs cannot truncate; see external_cli_agent_proxy.py).
    """
    log = usage_log_path(name)
    if not log.exists():
        return
    entries = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    sys.path.insert(0, str(ROOT / "scripts/benchmarks"))
    from result_store import locked_report  # same lock every suite writer uses
    with locked_report(REPORT) as payload:
        row = next((r for r in payload.get("results", []) if r.get("model") == name), None)
        if row is None:
            return
        row["cli_usage"] = {
            "requests": len(entries),
            "prompt_tokens": sum(e.get("prompt_tokens") or 0 for e in entries),
            "completion_tokens": sum(e.get("completion_tokens") or 0 for e in entries),
            "cost_usd_list_price": round(sum(e.get("cost_usd") or 0 for e in entries), 4),
            "source": str(log.relative_to(ROOT)),
        }
        row["output_budget"] = "uncapped: the agent CLI cannot truncate at max_tokens (local rows are capped)"


def run_one(row_name: str, provider_key: str, model: str, specialists: str) -> int:
    provider = PROVIDERS[provider_key]
    name = f"{row_name}-{model}"
    common = ["python3", str(WKS), name, "--out", str(REPORT),
              "--engine", provider["engine"], "--physical-gpus", "9",
              "--topology", f"cloud/agent-CLI provider; no local GPU ({provider_key})",
              "--specialists", specialists]

    port = free_port()
    accounted = provider.get("backend") in ("claude-cli", "omp")
    usage_log = usage_log_path(name) if accounted else None
    proxy_proc = subprocess.Popen(proxy_command(provider, model, port, usage_log), cwd=ROOT)
    try:
        wait_for_proxy(port)
        cmd = common + ["--external-url", f"http://127.0.0.1:{port}", "--external-model", model]
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if accounted:
            record_cli_usage(name)
        return rc
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
    parser.add_argument("--model", action="append",
                        help="repeatable; only run these model slugs of the selected providers")
    args = parser.parse_args()
    providers = args.provider or sorted(PROVIDERS)

    failures = 0
    for provider_key in providers:
        for model in PROVIDERS[provider_key]["models"]:
            if args.model and model not in args.model:
                continue
            row_name = f"{args.row_prefix}-{provider_key}"
            print(f"=== {provider_key} / {model} ===", flush=True)
            rc = run_one(row_name, provider_key, model, args.specialists)
            if rc != 0:
                failures += 1
                print(f"FAILED: {provider_key}/{model} (rc={rc})", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
