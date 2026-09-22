#!/usr/bin/env python3
"""Shared 8-benchmark evaluation suite for local LLM quality scoring.

Used by both scripts/benchmarks/benchmark_local_gguf_tp2.py (llama.cpp) and
infra/vllm_v100/benchmark_1cat_tp2.py (1Cat-vLLM) so every model gets the exact
same prompts and the exact same deterministic, programmatic scoring -- no LLM
judge, no subjective grading. Each task returns a score in [0, 1] plus a detail
dict so failures are inspectable, not just a number.

Every prompt is grounded in a versioned public fixture or an explicit synthetic
case. The fixtures live in this repository so importing the suite from a clean
clone cannot silently change prompt semantics.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, NamedTuple
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]


def extract_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for pos, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[pos:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return {}


def extract_python_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if match:
        return match.group(1)
    match = re.search(r"(def\s+\w+\(.*)", text, re.S)
    return match.group(1) if match else ""


# ---------------------------------------------------------------------------
# Task 1: RIV AU lifecycle extraction (pre-existing gold case, kept as-is)
# ---------------------------------------------------------------------------

RIV_PROMPT = (
    "Extract lifecycle facts for RIV AU ONLY from these source passages. Return exactly one compact JSON object "
    "with scalar-string keys event_type, acquirer, control_date, delisted_date, completion_date, successor. "
    "Dates must be YYYY-MM-DD. Do not infer a field not stated. Return JSON only. "
    "PASSAGE A (Rio Tinto annual report): Rio Tinto acquired 52.6% on 8 April 2011; on 1 August 2011 interest "
    "became 100%; Riversdale was delisted 7 July 2011 and subsequently renamed Rio Tinto Coal Mozambique. "
    "PASSAGE B (ASIC): the Rio Tinto takeover completed in August 2011; Rio Tinto delisted Riversdale and "
    "renamed the assets Rio Tinto Coal Mozambique."
)
RIV_EXPECTED = {
    "event_type": "acquisition",
    "acquirer": "Rio Tinto",
    "control_date": "2011-04-08",
    "delisted_date": "2011-07-07",
    "completion_date": "2011-08-01",
    "successor": "Rio Tinto Coal Mozambique",
}


def score_riv_au(raw: str) -> tuple[float, dict[str, Any]]:
    obj = extract_object(raw)
    checks: dict[str, bool] = {}
    for key, expected in RIV_EXPECTED.items():
        actual = str(obj.get(key, "")).strip()
        if key == "event_type":
            checks[key] = actual.lower() in {"acquisition", "acquired", "acquired_delisted", "takeover"}
        else:
            checks[key] = actual.casefold() == expected.casefold()
    return sum(checks.values()) / len(RIV_EXPECTED), {"extracted": obj, "checks": checks}


# ---------------------------------------------------------------------------
# Task 2: deterministic multi-step arithmetic (finance payout calculation)
# ---------------------------------------------------------------------------

ARITHMETIC_PROMPT = (
    "A Numerai signals stake is 12000 NMR at a stake price of $18.40 per NMR, so "
    "stake_notional = 12000 * 18.40 USD. The round's realized correlation payout factor is "
    "0.023 and the payout multiplier is 1.05x. Compute payout_usd = stake_notional * 0.023 * 1.05. "
    "Return only the final numeric answer rounded to 2 decimals, nothing else."
)
_ARITHMETIC_EXPECTED = 12000 * 18.40 * 0.023 * 1.05


def score_arithmetic(raw: str) -> tuple[float, dict[str, Any]]:
    numbers = [float(n.replace(",", "")) for n in re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", raw)]
    if not numbers:
        return 0.0, {"expected": _ARITHMETIC_EXPECTED, "found_numbers": []}
    candidate = numbers[-1]
    ok = abs(candidate - _ARITHMETIC_EXPECTED) < 0.5
    return (1.0 if ok else 0.0), {"expected": _ARITHMETIC_EXPECTED, "candidate": candidate}


# ---------------------------------------------------------------------------
# Task 3: code generation + real execution (max-drawdown, checked against a
# reference implementation, not a hardcoded literal)
# ---------------------------------------------------------------------------

CODE_DRAWDOWN_PROMPT = (
    "Write a single Python function named `max_drawdown(returns)` that takes a list of daily "
    "simple returns (floats, e.g. 0.01 for +1%) and returns the maximum drawdown (a negative "
    "float, e.g. -0.25 for -25%) of the cumulative wealth index that starts at 1.0. "
    "Return ONLY a python code block containing the function definition, no explanation, no example call."
)
_DRAWDOWN_TEST_RETURNS = [0.05, -0.10, 0.03, -0.20, 0.15, 0.02, -0.05]


def _reference_max_drawdown(returns: list[float]) -> float:
    wealth = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        wealth *= 1 + r
        peak = max(peak, wealth)
        max_dd = min(max_dd, (wealth - peak) / peak)
    return max_dd


def score_code_drawdown(raw: str) -> tuple[float, dict[str, Any]]:
    code = extract_python_code(raw)
    if "def max_drawdown" not in code:
        return 0.0, {"reason": "no max_drawdown function found in output", "code_present": bool(code)}
    harness = code + (
        "\nimport json as _json\n"
        f"_returns = {_DRAWDOWN_TEST_RETURNS!r}\n"
        "try:\n"
        "    _result = max_drawdown(_returns)\n"
        "    print(_json.dumps({'ok': True, 'result': _result}))\n"
        "except Exception as _exc:\n"
        "    print(_json.dumps({'ok': False, 'error': f'{type(_exc).__name__}: {_exc}'}))\n"
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = Path(tmp_dir) / f"exec_{uuid4().hex}.py"
        script_path.write_text(harness)
        try:
            proc = subprocess.run(["python3", str(script_path)], capture_output=True, text=True, timeout=5)
        except subprocess.TimeoutExpired:
            return 0.0, {"reason": "execution timed out"}
    last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        outcome = json.loads(last_line)
    except json.JSONDecodeError:
        return 0.0, {"reason": "no parseable output", "stdout": proc.stdout[-500:], "stderr": proc.stderr[-500:]}
    if not outcome.get("ok"):
        return 0.0, {"reason": "runtime error", "error": outcome.get("error")}
    reference = _reference_max_drawdown(_DRAWDOWN_TEST_RETURNS)
    result = outcome["result"]
    close = isinstance(result, (int, float)) and abs(result - reference) < 1e-3
    return (1.0 if close else 0.0), {"model_result": result, "reference": reference}


# ---------------------------------------------------------------------------
# Task 4: strict JSON schema adherence with factual grounding
# ---------------------------------------------------------------------------

JSON_SCHEMA_PROMPT = (
    "Return ONLY one valid JSON object (no markdown fence, no commentary) with exactly these keys and "
    'types: "ticker" (string), "country_iso2" (2-letter uppercase string), "is_delisted" (boolean). '
    "Fill in the values for Apple Inc, the well-known US technology company that is NOT delisted and "
    "trades under ticker AAPL."
)


def score_json_schema(raw: str) -> tuple[float, dict[str, Any]]:
    obj = extract_object(raw)
    is_delisted = obj.get("is_delisted")
    checks = {
        "ticker": str(obj.get("ticker", "")).strip().upper() == "AAPL",
        "country_iso2": str(obj.get("country_iso2", "")).strip().upper() == "US",
        "is_delisted": is_delisted is False
        or (isinstance(is_delisted, str) and is_delisted.strip().lower() == "false"),
    }
    return sum(checks.values()) / len(checks), {"extracted": obj, "checks": checks}


# ---------------------------------------------------------------------------
# Task 5: constrained-format instruction following (real GICS sector names)
# ---------------------------------------------------------------------------

GICS_SECTORS = {
    "energy", "materials", "industrials", "consumer discretionary", "consumer staples",
    "health care", "financials", "information technology", "communication services",
    "utilities", "real estate",
}
GICS_FORMAT_PROMPT = (
    "List exactly 5 official GICS equity sector names (the real 11-sector classification, e.g. Energy, "
    "Financials). Respond with exactly 5 lines, each line only the sector name in Title Case, nothing else: "
    "no numbering, no bullets, no preamble, no closing remarks."
)


def score_gics_format(raw: str) -> tuple[float, dict[str, Any]]:
    lines = [line.strip(" .-•*\t") for line in raw.strip().splitlines() if line.strip()]
    line_count_ok = len(lines) == 5
    valid = {line.lower() for line in lines if line.lower() in GICS_SECTORS}
    score = 0.5 * (1.0 if line_count_ok else 0.0) + 0.5 * (len(valid) / 5)
    return round(score, 4), {"lines": lines, "line_count_ok": line_count_ok, "distinct_valid_sectors": len(valid)}


# ---------------------------------------------------------------------------
# Task 6: long-context needle-in-haystack (real repo doc as haystack, real
# RIV AU fact as the needle)
# ---------------------------------------------------------------------------

_HAYSTACK_DOC = ROOT / "config/benchmark_fixtures/needle_haystack.md"
_NEEDLE = (
    "\n\nNOTE (unrelated corporate-lifecycle fact, ignore for the rest of this document): "
    "Rio Tinto obtained control of Riversdale Mining (RIV AU) on 2011-04-08.\n\n"
)


def _build_needle_prompt() -> str:
    haystack = _HAYSTACK_DOC.read_text(encoding="utf-8", errors="strict")[:4000]
    mid = len(haystack) // 2
    body = haystack[:mid] + _NEEDLE + haystack[mid:]
    question = (
        "\n\nQuestion: according to the NOTE embedded above, on what exact date (YYYY-MM-DD) did Rio Tinto "
        "obtain control of Riversdale Mining? Answer with only the date, nothing else."
    )
    return body + question


NEEDLE_HAYSTACK_PROMPT = _build_needle_prompt()


def score_needle_haystack(raw: str) -> tuple[float, dict[str, Any]]:
    found = "2011-04-08" in raw
    return (1.0 if found else 0.0), {"found": found}


# ---------------------------------------------------------------------------
# Task 7: summarization faithfulness on a versioned technical fixture
# ---------------------------------------------------------------------------

_SUMMARY_SOURCE = ROOT / "config/benchmark_fixtures/summary_source.md"


def _build_summary_prompt() -> str:
    note = _SUMMARY_SOURCE.read_text(encoding="utf-8", errors="strict")[:1500]
    return (
        "Summarize the following technical note in 2-3 sentences. Preserve the key numeric facts exactly; "
        "do not invent any number that is not present below.\n\n" + note
    )


SUMMARY_FAITHFULNESS_PROMPT = _build_summary_prompt()


def score_summary_faithfulness(raw: str) -> tuple[float, dict[str, Any]]:
    lower = raw.lower()
    checks = {
        "retains_key_figure": ("62%" in raw) or ("62 %" in raw) or ("13,350" in raw) or ("13350" in raw),
        "mentions_tiingo": "tiingo" in lower,
        "reasonable_length": len(raw.split()) <= 120,
    }
    return sum(checks.values()) / len(checks), checks


# ---------------------------------------------------------------------------
# Task 8: refusal / anti-hallucination calibration (mirrors the repo's own
# factplan_riv_au_lifecycle_llm_pilot.json hallucination_probe methodology)
# ---------------------------------------------------------------------------

REFUSAL_PROMPT = (
    "Based ONLY on this single passage, on what exact date was the company XZQ99 Holdings delisted? "
    "Passage: 'XZQ99 Holdings is a private entity with no public listing history in any available record.' "
    "If the passage does not state a delisting date, respond with exactly: INSUFFICIENT_INFORMATION. "
    "Do not guess a date."
)
_DATE_RE = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")
_ABSTAIN_PHRASES = (
    "insufficient information", "insufficient_information", "cannot be determined", "not stated",
    "no information", "does not state", "not provided", "unknown", "cannot determine",
)


def score_refusal_calibration(raw: str) -> tuple[float, dict[str, Any]]:
    lower = raw.lower()
    abstained = any(phrase in lower for phrase in _ABSTAIN_PHRASES)
    fabricated_date = bool(_DATE_RE.search(raw)) and not abstained
    if fabricated_date:
        score = 0.0
    elif abstained:
        score = 1.0
    else:
        score = 0.5
    return score, {"abstained": abstained, "fabricated_date_found": bool(_DATE_RE.search(raw))}


class Task(NamedTuple):
    id: str
    prompt: str
    max_tokens: int
    score: Callable[[str], tuple[float, dict[str, Any]]]


TASKS: list[Task] = [
    Task("riv_au_lifecycle", RIV_PROMPT, 800, score_riv_au),
    Task("arithmetic_payout", ARITHMETIC_PROMPT, 200, score_arithmetic),
    Task("code_exec_drawdown", CODE_DRAWDOWN_PROMPT, 400, score_code_drawdown),
    Task("json_schema_facts", JSON_SCHEMA_PROMPT, 150, score_json_schema),
    Task("gics_format_following", GICS_FORMAT_PROMPT, 150, score_gics_format),
    Task("needle_in_haystack", NEEDLE_HAYSTACK_PROMPT, 60, score_needle_haystack),
    Task("summary_faithfulness", SUMMARY_FAITHFULNESS_PROMPT, 220, score_summary_faithfulness),
    Task("refusal_calibration", REFUSAL_PROMPT, 60, score_refusal_calibration),
]


def message_text(response: dict[str, Any]) -> str:
    message = response.get("choices", [{}])[0].get("message", {})
    return (message.get("content") or "") + "\n" + (message.get("reasoning_content") or "")


def run_tasks(complete_fn: Callable[[str, int], dict[str, Any]]) -> dict[str, Any]:
    """Run every task through complete_fn(prompt, max_tokens) -> raw OpenAI-style response dict."""
    results: dict[str, Any] = {}
    for task in TASKS:
        response = complete_fn(task.prompt, task.max_tokens)
        raw = message_text(response)
        score, detail = task.score(raw)
        results[task.id] = {"score": round(score, 4), "detail": detail, "raw": raw.strip()[:800]}
    results["_mean_score"] = round(sum(v["score"] for v in results.values()) / len(TASKS), 4)
    return results
