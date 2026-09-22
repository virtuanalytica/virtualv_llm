#!/usr/bin/env python3
"""Resumable pairwise cascade for the remaining large dual-V100 models.

Every candidate is downloaded from a pinned public Hugging Face revision,
benchmarked with the current well-known protocol on physical GPUs 1+2 only,
rendered into the HTML report, and compared on the fixed four-task composite.
Only a fully validated loser is removed.  The current winner remains installed
until another complete candidate beats it, leaving one large-model winner at
the end while keeping disk use bounded to two candidate weights at a time.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path("/media/knight2/EDS2/models/llm")
REPORT = ROOT / "reports/well_known_suite_20260917.json"
STATE = ROOT / "reports/remaining_large_model_cascade_20260921.json"
SUITE = ROOT / "scripts/benchmarks/well_known_suite.py"
RENDER = ROOT / "scripts/benchmarks/build_dual_v100_html.py"
PROTOCOL = "v4-mmlu-fewshot-20260918"

RESEARCH_NOTES = {
    "qwen2.5-110b": (
        "Not scheduled: no official Qwen2.5-110B release/repository was found; "
        "the already measured Qwen2.5 line tops out at 72B. Qwen3.5-122B-A10B "
        "is the current, verifiable 100B-class replacement."
    ),
    "glm53-flash": (
        "Not in this four-hour sweep: the official/current model is 321B and its "
        "normal GGUF weights do not fit 64GB; community-pruned variants need a "
        "separate provenance and quality audit before they can enter this table."
    ),
    "mixtral_quant": (
        "The instructed public repo available without authentication exposes "
        "Q3_K_S at 61.50GB. IQ3_S is preferable at the same size where an "
        "instruct checkpoint is publicly available, but the discovered IQ3_S "
        "repository is the base checkpoint and is not substituted silently."
    ),
}

CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "llama3-70b-instruct-q4",
        "repo": "NousResearch/Meta-Llama-3-70B-Instruct-GGUF",
        "revision": "875f77142cf5c7b5d12d3c47e882bd8ac5ae660d",
        "include": "Meta-Llama-3-70B-Instruct-Q4_K_M.gguf",
        "file": "Meta-Llama-3-70B-Instruct-Q4_K_M.gguf",
        "published_at": "2024-04-19T01:16:05Z",
        "expected_bytes": 42_520_906_176,
        "quant": "Q4_K_M",
    },
    {
        "name": "qwen35-122b-a10b-iq3s",
        "repo": "unsloth/Qwen3.5-122B-A10B-GGUF",
        "revision": "51eab4d59d53f573fb9206cb3ce613f1d0aa392b",
        "include": "Qwen3.5-122B-A10B-UD-IQ3_S.gguf",
        "file": "Qwen3.5-122B-A10B-UD-IQ3_S.gguf",
        "published_at": "2026-02-24T14:49:45Z",
        "expected_bytes": 46_556_959_936,
        "quant": "UD-IQ3_S",
        "research_reason": "Current 122B-A10B MoE candidate that fits 64GB without CPU weight offload",
    },
    {
        "name": "command-r-plus-104b-0824-iq3m",
        "repo": "bartowski/c4ai-command-r-plus-08-2024-GGUF",
        "revision": "4cb2af9f9be9b2753915e67aa8d658cd031b8a21",
        "include": "c4ai-command-r-plus-08-2024-IQ3_M.gguf",
        "file": "c4ai-command-r-plus-08-2024-IQ3_M.gguf",
        "published_at": "2024-08-30T14:28:25Z",
        "expected_bytes": 47_683_357_504,
        "quant": "IQ3_M",
        "research_reason": "Later Command R+ checkpoint and higher-quality 3-bit imatrix quant",
    },
    {
        "name": "mixtral-8x22b-instruct-q3ks",
        "repo": "MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-GGUF",
        "revision": "9f2f6c5ec37f9bce5f5f3a7ff07b11d573443e62",
        "include": "Mixtral-8x22B-Instruct-v0.1.Q3_K_S-*.gguf",
        "file": "Mixtral-8x22B-Instruct-v0.1.Q3_K_S-00001-of-00003.gguf",
        "published_at": "2024-04-17T17:29:25Z",
        "expected_bytes": 61_504_109_376,
        "quant": "Q3_K_S",
    },
    {
        "name": "wizardlm2-8x22b-iq3s",
        "repo": "bartowski/WizardLM-2-8x22B-GGUF",
        "revision": "0e209c6d5f26385bd9584da7f1f8183532f86217",
        "include": "WizardLM-2-8x22B-IQ3_S.gguf/*.gguf",
        "file": "WizardLM-2-8x22B-IQ3_S.gguf/WizardLM-2-8x22B-IQ3_S-00001-of-00005.gguf",
        "published_at": "2024-04-16T17:30:47Z",
        "expected_bytes": 61_498_188_480,
        "quant": "IQ3_S",
    },
]


def load(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def row_for(name: str) -> dict[str, Any] | None:
    return next((row for row in load(REPORT, {"results": []}).get("results", [])
                 if row.get("model") == name), None)


def validation_errors(row: dict[str, Any] | None) -> list[str]:
    if not row:
        return ["result row missing"]
    errors: list[str] = []
    if row.get("error"):
        errors.append(f"error={row['error']}")
    if row.get("eval_protocol") != PROTOCOL:
        errors.append(f"protocol={row.get('eval_protocol')!r}")
    required = {
        "gsm8k": ((row.get("gsm8k") or {}).get("sample_len"), 50),
        "bbh": ((row.get("bbh") or {}).get("n_samples"), 48),
        "mmlu": ((row.get("mmlu_sample") or {}).get("n_samples"), 160),
        "truthfulqa": ((row.get("truthfulqa_gen") or {}).get("sample_len"), 30),
        "humaneval": ((row.get("humaneval") or {}).get("n_problems"), 40),
    }
    metrics = {
        "gsm8k": [value for key, value in (row.get("gsm8k") or {}).items()
                   if key.startswith("exact_match") and "stderr" not in key],
        "bbh": [(row.get("bbh") or {}).get("mean_accuracy")],
        "mmlu": [(row.get("mmlu_sample") or {}).get("mean_accuracy")],
        "truthfulqa": [(row.get("truthfulqa_gen") or {}).get("rougeL_acc,none")],
        "humaneval": [(row.get("humaneval") or {}).get("pass_at_1")],
    }
    for label, (actual, expected) in required.items():
        if actual != expected:
            errors.append(f"{label} samples={actual}, expected={expected}")
        if not any(isinstance(value, (int, float)) for value in metrics[label]):
            errors.append(f"{label} metric missing")
    if not isinstance(row.get("completion_tokens_per_second"), (int, float)):
        errors.append("decode throughput missing")
    if row.get("topology") != "physical GPU [1, 2], split=layer":
        errors.append(f"topology={row.get('topology')!r}")
    if row.get("cuda_device_order") != "PCI_BUS_ID" or row.get("cuda_visible_devices") != "1,2":
        errors.append("CUDA selection is not physical GPU 1,2 in PCI_BUS_ID order")
    probe = str(row.get("visible_device_probe", ""))
    devices = [line for line in probe.splitlines() if line.strip().startswith("CUDA")]
    if len(devices) != 2 or any("Tesla V100-SXM2-32GB" not in line or "RTX" in line for line in devices):
        errors.append("device probe is not exactly two V100s")
    telemetry = row.get("gpu_decode_telemetry") or {}
    for physical in ("1", "2"):
        sample = telemetry.get(physical) or {}
        if not sample.get("samples") or (sample.get("max_util_pct") or 0) < 10:
            errors.append(f"V100 {physical} lacks active decode telemetry")
    return errors


def composite(row: dict[str, Any]) -> float:
    gsm = max(value for key, value in (row.get("gsm8k") or {}).items()
              if key.startswith("exact_match") and "stderr" not in key)
    values = [gsm, row["bbh"]["mean_accuracy"], row["mmlu_sample"]["mean_accuracy"],
              row["humaneval"]["pass_at_1"]]
    return sum(float(value) for value in values) / len(values)


def model_dir(candidate: dict[str, Any]) -> Path:
    return MODEL_ROOT / candidate["name"]


def model_file(candidate: dict[str, Any]) -> Path:
    return model_dir(candidate) / candidate["file"]


def downloaded_weight_bytes(candidate: dict[str, Any]) -> int:
    """Count model shards only, never Hugging Face cache/lock metadata."""
    first = model_file(candidate)
    if not first.exists():
        return 0
    if "-00001-of-" in first.name:
        return sum(path.stat().st_size for path in first.parent.glob("*.gguf") if path.is_file())
    return first.stat().st_size


def process_uses(path: Path) -> bool:
    needle = str(path.resolve())
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            command = proc.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if needle in command:
            return True
    return False


def record_provenance(candidate: dict[str, Any]) -> None:
    report = load(REPORT, {"results": []})
    row = next(row for row in report["results"] if row.get("model") == candidate["name"])
    row["model_source"] = f"https://huggingface.co/{candidate['repo']}"
    row["source_repo"] = candidate["repo"]
    row["source_revision"] = candidate["revision"]
    row["source_published_at"] = candidate["published_at"]
    row["quantization"] = candidate["quant"]
    row["weight_bytes"] = downloaded_weight_bytes(candidate)
    atomic_write(REPORT, report)


def download(candidate: dict[str, Any], state: dict[str, Any]) -> None:
    target = model_file(candidate)
    if target.exists():
        return
    free = shutil.disk_usage(MODEL_ROOT).free
    required = int(candidate["expected_bytes"]) + 8_000_000_000
    if free < required:
        raise RuntimeError(f"{candidate['name']}: {free} bytes free; {required} required with margin")
    directory = model_dir(candidate)
    directory.mkdir(parents=True, exist_ok=True)
    command = ["hf", "download", candidate["repo"], "--revision", candidate["revision"],
               "--include", candidate["include"], "--local-dir", str(directory)]
    state["events"].append({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "model": candidate["name"], "event": "download_started",
                            "command": command, "free_before": free})
    atomic_write(STATE, state)
    subprocess.run(command, check=True)
    actual = downloaded_weight_bytes(candidate)
    if not target.exists() or actual != candidate["expected_bytes"]:
        raise RuntimeError(f"{candidate['name']}: downloaded {actual} bytes; expected {candidate['expected_bytes']}")
    state["events"].append({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "model": candidate["name"], "event": "download_complete",
                            "weight_bytes": actual})
    atomic_write(STATE, state)


def benchmark(candidate: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    existing = row_for(candidate["name"])
    if validation_errors(existing):
        subprocess.run(["flock", "-n", "/tmp/v100_exclusive.lock", "python3", str(SUITE),
                        candidate["name"], "--profile", "dual-layer", "--out", str(REPORT)],
                       cwd=ROOT, check=True)
    row = row_for(candidate["name"])
    errors = validation_errors(row)
    if errors:
        raise RuntimeError(f"{candidate['name']} incomplete; weights retained: {errors}")
    record_provenance(candidate)
    row = row_for(candidate["name"])
    subprocess.run(["python3", str(RENDER)], cwd=ROOT, check=True)
    state["events"].append({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "model": candidate["name"], "event": "benchmark_complete",
                            "composite": composite(row),
                            "tokens_per_second": row["completion_tokens_per_second"]})
    atomic_write(STATE, state)
    return row


def prune(candidate: dict[str, Any], state: dict[str, Any]) -> None:
    directory = model_dir(candidate)
    if not directory.exists():
        return
    errors = validation_errors(row_for(candidate["name"]))
    if errors:
        raise RuntimeError(f"refusing to remove incomplete {candidate['name']}: {errors}")
    resolved = directory.resolve()
    if directory.is_symlink() or not resolved.is_relative_to(MODEL_ROOT.resolve()):
        raise RuntimeError(f"unsafe model path: {directory}")
    if process_uses(directory):
        raise RuntimeError(f"model is active and cannot be removed: {directory}")
    before = shutil.disk_usage(MODEL_ROOT).free
    size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    state["events"].append({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "model": candidate["name"], "event": "prune_authorized",
                            "validated": True, "bytes_targeted": size,
                            "reinstall": ["hf", "download", candidate["repo"], "--revision",
                                          candidate["revision"], "--include", candidate["include"],
                                          "--local-dir", str(directory)]})
    atomic_write(STATE, state)
    shutil.rmtree(directory)
    after = shutil.disk_usage(MODEL_ROOT).free
    state["events"].append({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "model": candidate["name"], "event": "pruned",
                            "bytes_freed": after - before})
    atomic_write(STATE, state)


def main() -> int:
    state = load(STATE, {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "events": []})
    state["gpu_contract"] = {"cuda_device_order": "PCI_BUS_ID", "visible_physical_gpus": [1, 2],
                             "required_name": "Tesla V100-SXM2-32GB", "excluded_gpus": [0, 3],
                             "split_mode": "layer"}
    state["research_notes"] = RESEARCH_NOTES
    state["candidates"] = CANDIDATES
    atomic_write(STATE, state)
    remembered = state.get("current_winner") or state.get("winner")
    winner: dict[str, Any] | None = next(
        (candidate for candidate in CANDIDATES
         if candidate["name"] == remembered and not validation_errors(row_for(candidate["name"]))
         and model_file(candidate).exists()),
        None,
    )
    for candidate in CANDIDATES:
        existing = row_for(candidate["name"])
        if not validation_errors(existing):
            # A validated loser may already have been pruned in an earlier run.
            # Never redownload it merely to rediscover the same score.
            if winner is None and model_file(candidate).exists():
                winner = candidate
                state["current_winner"] = winner["name"]
                state["current_winner_composite"] = composite(existing)
                atomic_write(STATE, state)
            continue
        download(candidate, state)
        row = benchmark(candidate, state)
        if winner is None:
            winner = candidate
            state["current_winner"] = winner["name"]
            state["current_winner_composite"] = composite(row)
            atomic_write(STATE, state)
            continue
        winner_row = row_for(winner["name"])
        if composite(row) > composite(winner_row):
            prune(winner, state)
            winner = candidate
        else:
            prune(candidate, state)
        state["current_winner"] = winner["name"]
        state["current_winner_composite"] = composite(row_for(winner["name"]))
        atomic_write(STATE, state)
    if winner is not None:
        state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        state["winner"] = winner["name"]
        state["winner_composite"] = composite(row_for(winner["name"]))
        atomic_write(STATE, state)
        subprocess.run(["python3", str(RENDER)], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
