"""Agent-CLI backends: command isolation and parsing of recorded real output."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/benchmarks"))
import external_cli_agent_proxy as proxy  # noqa: E402
import run_external_provider_cascade as cascade  # noqa: E402

# Recorded 2026-10-07 from `omp -p --model zai/glm-5.3-flash` (only the cwd is replaced).
OMP_EVENTS = (ROOT / "tests/fixtures/omp_glm53_flash_events.jsonl").read_text()


def test_omp_answer_excludes_thinking_and_carries_accounting():
    text, usage = proxy.parse_omp_events(OMP_EVENTS)
    assert text == "391"
    assert usage["prompt_tokens"] == 242 and usage["completion_tokens"] == 6
    assert usage["duration_api_ms"] > usage["ttft_ms"] > 0
    assert usage["cost_usd"] > 0


def test_omp_provider_error_is_raised_not_scored():
    failed = OMP_EVENTS.replace('"stopReason": "stop"', '"stopReason": "error"')
    with pytest.raises(RuntimeError, match="provider error"):
        proxy.parse_omp_events(failed)
    with pytest.raises(RuntimeError, match="no assistant message"):
        proxy.parse_omp_events("not json\n")


def test_omp_runs_without_tools_or_local_context():
    command = proxy.omp_command("2+2?", "zai/glm-5.3-flash")
    assert command[-1] == "2+2?" and command[command.index("--model") + 1] == "zai/glm-5.3-flash"
    assert {"--no-tools", "--no-session", "--no-extensions", "--no-skills", "--no-rules"} <= set(command)


def test_cascade_addresses_omp_models_by_provider():
    command = cascade.proxy_command(cascade.PROVIDERS["omp-zai"], "glm-5.3-flash", 1234)
    assert command[command.index("--model") + 1] == "zai/glm-5.3-flash"
    assert command[command.index("--backend") + 1] == "omp"


def test_external_url_with_doubled_v1_is_refused_before_a_run():
    import well_known_suite as wks

    assert wks.external_base_url("http://127.0.0.1:8027/", "/v1/chat/completions") == "http://127.0.0.1:8027"
    # Gemini's OpenAI layer: base carries the version, path does not repeat it.
    assert wks.external_base_url("https://example.test/v1beta/openai", "/chat/completions").endswith("/openai")
    with pytest.raises(SystemExit, match="drop /v1"):
        wks.external_base_url("http://127.0.0.1:8027/v1", "/v1/chat/completions")


def test_omp_bench_payload_reduces_to_published_distribution():
    import json

    import omp_throughput

    # Recorded 2026-10-07: `omp bench openai-codex/gpt-6-luna --profile chat --runs 10 --json`.
    bench = json.loads((ROOT / "tests/fixtures/omp_bench_gpt6_luna.json").read_text())
    summary = omp_throughput.summarise(bench, "openai-codex/gpt-6-luna")
    assert summary["requests"] == 10 and summary["failed_requests"] == 0
    decode = summary["decode_tokens_per_second"]
    assert decode["min"] <= decode["p50"] <= decode["p95"] <= decode["max"]
    assert decode["p50"] == 55.56 and summary["time_to_first_token_ms"]["p50"] == 1480
