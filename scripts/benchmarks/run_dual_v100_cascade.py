#!/usr/bin/env python3
"""Resumable acquire -> benchmark -> prune cascade for the dual V100 pair.

Only a complete, error-free well-known-suite result authorizes deletion. The
two large finalists coexist until both have results; then only the lower mean
score is removed. Public Hugging Face downloads require no API key.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/well_known_suite_20260917.json"
STATE = ROOT / "reports/dual_v100_cascade_20260918.json"
MODELS_ROOT = Path("/media/knight2/EDS2/models/llm")
SUITE = ROOT / "scripts/benchmarks/well_known_suite.py"
MIXTURE = ROOT / "scripts/benchmarks/mixture_of_models.py"
HTML = ROOT / "scripts/benchmarks/build_dual_v100_html.py"

LOCAL = [
    ("qwen38-27b-q4", MODELS_ROOT / "qwen38-27b"),
    ("deepseek-r1-qwen32b-q4", MODELS_ROOT / "deepseek-r1-qwen32b"),
    ("qwen36-27b-iq3", Path("/media/knight2/EDS2/models/qwen3.6-27b-unsloth-ud-iq3-xxs.gguf")),
    ("qwen35-27b-q4", Path("/media/knight2/EDS2/lmstudio-models/lmstudio-community/Qwen3.5-27B-GGUF")),
    ("gemma4-26b-a4b-q4", Path("/media/knight2/EDS2/lmstudio-models/lmstudio-community/gemma-4-26B-A4B-it-GGUF")),
    ("devstral-small2-24b-q4", Path("/media/knight2/EDS2/lmstudio-models/lmstudio-community/Devstral-Small-2-24B-Instruct-2512-GGUF")),
    ("qwen25-72b-q4", MODELS_ROOT / "qwen25-72b-q4km"),
    ("llama31-70b-instruct-q4", MODELS_ROOT / "llama31-70b-instruct-q4"),
]

# qwen36 is currently served on the A4000 and qwen38 is the production/reference
# model, so neither is pruned. Everything below can be downloaded again publicly.
PRUNE_AFTER_SUCCESS = {
    "deepseek-r1-qwen32b-q4", "qwen35-27b-q4", "gemma4-26b-a4b-q4",
    "devstral-small2-24b-q4", "qwen25-72b-q4", "llama31-70b-instruct-q4",
}

PROTECTED = {"qwen38-27b-q4", "qwen36-27b-iq3"}
ALLOWED_MODEL_ROOTS = [
    Path("/media/knight2/EDS2/models/llm").resolve(),
    Path("/media/knight2/EDS2/lmstudio-models/lmstudio-community").resolve(),
]
LOCAL_PROVENANCE = {
    "deepseek-r1-qwen32b-q4": {
        "repo": "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        "revision": "1dc8cf9ffa5dd333057ea1b09ccf4772d8726dec",
        "include": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf", "published_at": "2025-01-20",
    },
    "qwen35-27b-q4": {
        "repo": "lmstudio-community/Qwen3.5-27B-GGUF",
        "revision": "3b2745e699b00222112360a39f8854bf62774c1a",
        "include": "Qwen3.5-27B-Q4_K_M.gguf", "published_at": "2026-02-24",
    },
    "gemma4-26b-a4b-q4": {
        "repo": "lmstudio-community/gemma-4-26B-A4B-it-GGUF",
        "revision": "f6e6747823b2912661935db7e0009287c4838073",
        "include": "gemma-4-26B-A4B-it-Q4_K_M.gguf", "published_at": "2026-04-02",
    },
    "devstral-small2-24b-q4": {
        "repo": "lmstudio-community/Devstral-Small-2-24B-Instruct-2512-GGUF",
        "revision": "e471f62bf546b027d9f23f679bcd1a295eabf403",
        "include": "Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf", "published_at": "2025-12-08",
    },
    "qwen25-72b-q4": {
        "repo": "Qwen/Qwen2.5-72B-Instruct-GGUF",
        "revision": "7ca3bc388f97b264c4283bc9bf1055e2abc38441",
        "include": "*q4_k_m*.gguf", "published_at": "2024-09-17",
    },
    "llama31-70b-instruct-q4": {
        "repo": "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
        "revision": "83fb6e83d0a8aada42d499259bc929d922e9a558",
        "include": "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf", "published_at": "2024-07-23",
    },
}

FINALISTS = [
    {
        "name": "glm45-air-106b-iq3",
        "repo": "unsloth/GLM-4.5-Air-GGUF",
        "revision": "506d64aa8c5cfe9dbbf00bc7a15739438f83204d",
        "include": "UD-IQ3_XXS/*.gguf",
        "dir": MODELS_ROOT / "glm45-air-106b-iq3",
        "published_at": "2025-08-05",
        "quant": "UD-IQ3_XXS",
        "download_bytes": 51_400_000_000,
        "source": "https://huggingface.co/unsloth/GLM-4.5-Air-GGUF/tree/main/UD-IQ3_XXS",
    },
    {
        "name": "gpt-oss-120b-q4",
        "repo": "unsloth/gpt-oss-120b-GGUF",
        "revision": "ff1a82da6ad466e32284fa3d2b86694db3204789",
        "include": "Q4_K_M/*.gguf",
        "dir": MODELS_ROOT / "gpt-oss-120b-q4",
        "published_at": "2025-08-05",
        "quant": "Q4_K_M (MXFP4-derived MoE weights)",
        "download_bytes": 62_800_000_000,
        "source": "https://huggingface.co/unsloth/gpt-oss-120b-GGUF/tree/main/Q4_K_M",
    },
]


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def validate_result(row: dict[str, Any]) -> list[str]:
    """Return deletion/skip blockers for a purportedly complete result."""
    blockers: list[str] = []
    required = {
        "gsm8k": ((row.get("gsm8k") or {}).get("exact_match,flexible-extract"),
                  (row.get("gsm8k") or {}).get("sample_len"), 50),
        "bbh": ((row.get("bbh") or {}).get("mean_accuracy"),
                (row.get("bbh") or {}).get("n_samples"), 48),
        "mmlu": ((row.get("mmlu_sample") or {}).get("mean_accuracy"),
                 (row.get("mmlu_sample") or {}).get("n_samples"), 160),
        "truthfulqa": ((row.get("truthfulqa_gen") or {}).get("rougeL_acc,none"),
                       (row.get("truthfulqa_gen") or {}).get("sample_len"), 30),
        "humaneval": ((row.get("humaneval") or {}).get("pass_at_1"),
                      (row.get("humaneval") or {}).get("n_problems"), 40),
    }
    if row.get("error"):
        blockers.append(f"error={row['error']}")
    if row.get("eval_protocol") != "v2-chat-template-20260918":
        blockers.append("wrong eval protocol")
    for label, (metric, actual_n, expected_n) in required.items():
        if not isinstance(metric, (int, float)):
            blockers.append(f"{label} metric missing")
        if actual_n != expected_n:
            blockers.append(f"{label} samples={actual_n}, expected={expected_n}")
    for label in ("completion_tokens_per_second", "prompt_tokens_per_second"):
        if not isinstance(row.get(label), (int, float)) or row[label] <= 0:
            blockers.append(f"{label} missing")
    if row.get("topology") != "physical GPU [1, 2], split=layer":
        blockers.append(f"unexpected topology={row.get('topology')!r}")
    visible_rows = [line.strip() for line in str(row.get("visible_device_probe", "")).splitlines()
                    if line.strip().startswith("CUDA")]
    if len(visible_rows) != 2 or any("Tesla V100-SXM2-32GB" not in line or "RTX" in line
                                     for line in visible_rows):
        blockers.append("visible CUDA inventory is not exactly two V100s")
    telemetry = row.get("gpu_decode_telemetry") or {}
    for gpu in ("1", "2"):
        data = telemetry.get(gpu) or {}
        if not data.get("samples") or (data.get("max_util_pct") or 0) < 10:
            blockers.append(f"physical V100 {gpu} has no valid decode telemetry")
    return blockers


def result_for(name: str) -> dict[str, Any] | None:
    report = load_json(REPORT, {"results": []})
    return next((r for r in report.get("results", []) if r.get("model") == name
                 and not validate_result(r)), None)


def dir_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def path_is_active(path: Path) -> bool:
    target = str(path.resolve())
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cmdline = proc.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if target in cmdline:
            return True
    return False


def safe_prune(name: str, path: Path, state: dict[str, Any], event: str) -> None:
    if name in PROTECTED:
        raise RuntimeError(f"refusing to prune protected model {name}")
    row = result_for(name)
    if row is None:
        raise RuntimeError(f"refusing to prune {name}: no fully validated result")
    resolved = path.resolve()
    if path.is_symlink() or not any(resolved.is_relative_to(root) for root in ALLOWED_MODEL_ROOTS):
        raise RuntimeError(f"refusing unsafe prune path for {name}: {path}")
    provenance = LOCAL_PROVENANCE.get(name) or next((f for f in FINALISTS if f["name"] == name), None)
    if not provenance or not provenance.get("repo") or not provenance.get("revision"):
        raise RuntimeError(f"refusing to prune {name}: reinstall provenance incomplete")
    if path_is_active(path):
        raise RuntimeError(f"refusing to prune {name}: model path is in an active process command line")
    freed = dir_bytes(path)
    free_before = shutil.disk_usage(MODELS_ROOT).free
    reinstall = ["hf", "download", provenance["repo"], "--revision", provenance["revision"],
                 "--include", provenance["include"], "--local-dir", str(path)]
    provenance_record = {k: str(v) if isinstance(v, Path) else v for k, v in provenance.items()}
    state["events"].append({"model": name, "event": f"{event}_authorized",
                            "bytes_targeted": freed, "path": str(path), "provenance": provenance_record,
                            "reinstall_command": reinstall, "free_before": free_before,
                            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    atomic_write(STATE, state)
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    free_after = shutil.disk_usage(MODELS_ROOT).free
    state["events"].append({"model": name, "event": event, "bytes_freed": free_after - free_before,
                            "measured_tree_bytes": freed, "free_after": free_after,
                            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    atomic_write(STATE, state)


def run_suite(name: str) -> None:
    if result_for(name):
        print(f"SKIP {name}: complete result already present", flush=True)
        return
    # Lock only the GPU phase. Downloads, report rendering and pruning must not
    # unnecessarily block ad-hoc V100 work.
    subprocess.run(["flock", "-n", "/tmp/v100_exclusive.lock", "python3", str(SUITE), name,
                    "--profile", "dual-layer", "--out", str(REPORT)], cwd=ROOT, check=True)
    if not result_for(name):
        report = load_json(REPORT, {"results": []})
        row = next((r for r in report.get("results", []) if r.get("model") == name), {})
        raise RuntimeError(f"{name}: suite returned without a complete result: {validate_result(row)}")
    # Publish progress after every atomic model result, not only after the
    # multi-hour cascade has completely finished.
    subprocess.run(["python3", str(HTML)], cwd=ROOT, check=True)


def score(row: dict[str, Any]) -> float:
    values = [
        (row.get("gsm8k") or {}).get("exact_match,flexible-extract"),
        (row.get("bbh") or {}).get("mean_accuracy"),
        (row.get("mmlu_sample") or {}).get("mean_accuracy"),
        (row.get("truthfulqa_gen") or {}).get("rougeL_acc,none"),
        (row.get("humaneval") or {}).get("pass_at_1"),
    ]
    usable = [float(v) for v in values if isinstance(v, (int, float))]
    return sum(usable) / len(usable) if usable else -1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-local", action="store_true", help="benchmark local models but do not prune them")
    ap.add_argument("--skip-downloads", action="store_true", help="stop after the already-local sweep")
    args = ap.parse_args()
    state = load_json(STATE, {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "events": []})
    state["finalists"] = [{k: str(v) if isinstance(v, Path) else v for k, v in f.items()} for f in FINALISTS]
    state["local_provenance"] = LOCAL_PROVENANCE
    state["protected_models"] = sorted(PROTECTED)
    state["gpu_contract"] = {"physical_gpus": [1, 2], "required_name": "Tesla V100-SXM2-32GB",
                             "excluded_physical_gpus": [0, 3], "cuda_device_order": "PCI_BUS_ID"}
    atomic_write(STATE, state)

    for name, path in LOCAL:
        run_suite(name)
        if name in PRUNE_AFTER_SUCCESS and not args.keep_local and path.exists():
            safe_prune(name, path, state, "pruned_after_success")

    # The ensemble is post-hoc over recorded per-question outputs; weights are
    # intentionally not required after the individual successful runs.
    members = [name for name, _ in LOCAL if result_for(name)]
    if len(members) >= 2:
        subprocess.run(["python3", str(MIXTURE), *members, "--out", str(REPORT)], cwd=ROOT, check=True)
    if args.skip_downloads:
        return 0

    for item in FINALISTS:
        name, target = item["name"], item["dir"]
        if not result_for(name):
            target.mkdir(parents=True, exist_ok=True)
            if not any(target.rglob("*.gguf")):
                free = shutil.disk_usage(MODELS_ROOT).free
                required = int(item["download_bytes"]) + 8_000_000_000
                if free < required:
                    raise RuntimeError(f"{name}: {free} bytes free, {required} required including safety margin")
                subprocess.run(["hf", "download", item["repo"], "--revision", item["revision"],
                                "--include", item["include"], "--local-dir", str(target)], check=True)
            run_suite(name)

    def rank_key(item: dict[str, Any]) -> tuple[float, float, float, int]:
        row = result_for(item["name"]) or {}
        return (score(row), float((row.get("humaneval") or {}).get("pass_at_1", -1)),
                float(row.get("completion_tokens_per_second", -1)), -int(item["download_bytes"]))

    ranked = sorted(FINALISTS, reverse=True, key=rank_key)
    winner = ranked[0]
    winner_score = score(result_for(winner["name"]) or {})
    for loser in ranked[1:]:
        loser_score = score(result_for(loser["name"]) or {})
        if loser["dir"].exists():
            safe_prune(loser["name"], loser["dir"], state, "finalist_pruned")
    state.update({"completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "winner": winner["name"], "winner_mean_five_benchmarks": winner_score})
    atomic_write(STATE, state)
    subprocess.run(["python3", str(HTML)], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
