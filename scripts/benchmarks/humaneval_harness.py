#!/usr/bin/env python3
"""Standalone HumanEval pass@1 harness (the official 164-problem OpenAI set).

lm-eval-harness 0.4.13's built-in humaneval/mbpp tasks are broken in this
environment: they import HF `evaluate`'s "code_eval" metric, which fails at
import time because the installed (old, system) `evaluate` calls
`huggingface_hub.hf_api.HfFolder`, an API removed from the newer (user-local)
huggingface_hub 1.30.0. Rather than patch shared, cross-agent Python
dependencies on a machine other sessions rely on, this harness re-implements
the same well-known methodology directly: generate a completion for each
problem, insert it into the canonical test harness, execute in a subprocess,
and score pass@1 -- no HF `evaluate` dependency at all.

Data: the official openai/human-eval GitHub release
(data/HumanEval.jsonl.gz), cached at data/eval_cache/HumanEval.jsonl.gz.
"""
from __future__ import annotations

import gzip
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/eval_cache/HumanEval.jsonl.gz"


def load_problems(limit: int | None = None) -> list[dict[str, Any]]:
    with gzip.open(DATA_PATH) as f:
        problems = [json.loads(line) for line in f if line.strip()]
    return problems[:limit] if limit else problems


def _extract_code(prompt: str, raw_completion: str) -> str:
    text = raw_completion
    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            candidate = part[6:] if part.startswith("python") else part
            if candidate.strip():
                text = candidate
                break
    if text.lstrip().startswith(prompt.lstrip()[:20]):
        return text
    return prompt + text


def _run_one(problem: dict[str, Any], completion: str, timeout: float = 8.0) -> bool:
    program = (
        completion
        + "\n"
        + problem["test"]
        + f"\ncheck({problem['entry_point']})\n"
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = Path(tmp_dir) / "prog.py"
        script_path.write_text(program)
        try:
            proc = subprocess.run(
                ["python3", str(script_path)], capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return False
    return proc.returncode == 0


def run_humaneval(
    complete_fn: Callable[[str, int], str], limit: int | None = 40, max_tokens: int = 512
) -> dict[str, Any]:
    """complete_fn(prompt, max_tokens) -> raw completion text (no execution)."""
    problems = load_problems(limit)
    passed = 0
    failures: list[str] = []
    per_task: dict[str, dict[str, Any]] = {}
    for problem in problems:
        raw = complete_fn(problem["prompt"], max_tokens)
        code = _extract_code(problem["prompt"], raw)
        ok = _run_one(problem, code)
        passed += int(ok)
        # Kept per-task (not just the failures list) so
        # scripts/benchmarks/mixture_of_models.py can apply a best-of-N-models
        # policy across already-tested models without re-running generation.
        per_task[problem["task_id"]] = {"passed": ok, "code": code}
        if not ok:
            failures.append(problem["task_id"])
    total = len(problems)
    return {
        "benchmark": "humaneval",
        "n_problems": total,
        "n_passed": passed,
        "per_task": per_task,
        "pass_at_1": round(passed / total, 4) if total else None,
        "failed_task_ids": failures,
    }
