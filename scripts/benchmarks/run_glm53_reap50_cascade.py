#!/usr/bin/env python3
"""Resumable, disk-bounded GLM-5.3 REAP50 GGUF benchmark cascade.

Every quant is fetched from the author's immutable revision, checked by size
and SHA-256, evaluated separately on 2xV100 and all four local GPUs, then
deleted only after both complete rows have been written atomically.  The
author's glm5-next fork is isolated through LLAMA_SERVER/LLAMA_BENCH; no
production llama.cpp binary is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rank_models as rm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = Path("/media/knight2/EDS2/models/llm")
REPORTS = ROOT / "reports"
STATE = REPORTS / "glm53_reap50_cascade_20260922.json"
REPORT = REPORTS / "well_known_suite_20260917.json"
WKS = ROOT / "scripts/benchmarks/well_known_suite.py"
RENDER = ROOT / "scripts/benchmarks/build_dual_v100_html.py"
RUNTIME = Path("/media/knight2/EDS2/tools/llama.cpp-glm53-reap50-v1/build-v100")
REPO = "patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF"
REVISION = "8654e38254ee6b80946b5d5d00cbfcc96ed88659"
PUBLISHED_AT = "2026-09-06T22:49:01Z"
SAFETY_BYTES = 8_000_000_000

# Exact size/hash must be verified before a potentially multi-hour suite.
CANDIDATES: list[dict[str, Any]] = [
    {"key": "iq3m", "file": "GLM-5.3-Flash-REAP50-IQ3_M.gguf", "bytes": 72_132_392_352,
     "sha256": "6954383ec7db1cb468cf9e4e948035aa0542b8c51a3766c5253650721e601c7a"},
    {"key": "q3km", "file": "GLM-5.3-Flash-REAP50-Q3_K_M.gguf", "bytes": 78_776_759_712,
     "sha256": None},
    {"key": "iq4xs", "file": "GLM-5.3-Flash-REAP50-IQ4_XS.gguf", "bytes": 88_048_752_032,
     "sha256": None},
    {"key": "q4km", "file": "GLM-5.3-Flash-REAP50-Q4_K_M.gguf", "bytes": 99_330_938_048,
     "sha256": None},
]
PROFILES = (("v100", "dual-layer"), ("allfour", "all-four-layer"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {"events": [], "candidates": CANDIDATES}


def save(state: dict[str, Any]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2) + "\n")
    temp.replace(STATE)


def event(state: dict[str, Any], candidate: dict[str, Any], kind: str, **data: Any) -> None:
    state.setdefault("events", []).append({"at": now(), "quant": candidate["key"], "event": kind, **data})
    save(state)


def target(candidate: dict[str, Any]) -> Path:
    return MODELS / f"glm53-reap50-{candidate['key']}" / candidate["file"]


def complete_rows(candidate: dict[str, Any]) -> bool:
    if not REPORT.exists():
        return False
    rows = {row.get("model"): row for row in load(REPORT).get("results", [])}
    for suffix, _ in PROFILES:
        row = rows.get(f"glm53-reap50-{candidate['key']}-{suffix}")
        if not row or row.get("error"):
            return False
        if not _row_has_full_composite(row):
            return False
    return True


# 2026-09-22 fix: this used to check isinstance(row.get("gsm8k"), (int, float))
# directly, but "gsm8k" is always a nested dict (e.g. {"exact_match,flexible-
# extract": 0.66, ...}), never a bare number -- so this was False for every
# possible result, complete or not. row_complete() therefore always reran an
# already-complete v100 profile (confirmed: it restarted glm53-reap50-iq3m-v100
# from scratch immediately after that profile finished cleanly), and
# complete_rows() could never authorize prune() either. Use the same
# gsm8k_score() extraction rank_models.py/build_dual_v100_html.py already use
# elsewhere instead of reading the raw dict.
def _row_has_full_composite(row: dict[str, Any]) -> bool:
    checks = (rm.gsm8k_score(row), (row.get("bbh") or {}).get("mean_accuracy"),
              (row.get("mmlu_sample") or {}).get("mean_accuracy"), (row.get("humaneval") or {}).get("pass_at_1"),
              row.get("completion_tokens_per_second"))
    return all(isinstance(value, (int, float)) for value in checks)


def row_complete(model: str) -> bool:
    """A stopped cascade resumes at the missing hardware profile, never reruns one."""
    if not REPORT.exists():
        return False
    row = next((item for item in load(REPORT).get("results", []) if item.get("model") == model), None)
    if not row or row.get("error"):
        return False
    return _row_has_full_composite(row)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(candidate: dict[str, Any], state: dict[str, Any]) -> Path:
    path = target(candidate)
    if path.exists() and path.stat().st_size == candidate["bytes"]:
        return path
    free = shutil.disk_usage(MODELS).free
    required = candidate["bytes"] + SAFETY_BYTES
    if free < required:
        raise RuntimeError(f"{candidate['key']}: {free} free bytes, {required} required")
    path.parent.mkdir(parents=True, exist_ok=True)
    event(state, candidate, "download_started", free_before=free, repo=REPO, revision=REVISION)
    subprocess.run(["hf", "download", REPO, "--revision", REVISION, "--include", candidate["file"],
                    "--local-dir", str(path.parent)], check=True)
    if not path.exists() or path.stat().st_size != candidate["bytes"]:
        raise RuntimeError(f"unexpected downloaded size for {path}")
    actual_hash = hash_file(path)
    if candidate["sha256"] and actual_hash != candidate["sha256"]:
        raise RuntimeError(f"SHA256 mismatch for {path}: {actual_hash}")
    event(state, candidate, "download_verified", bytes=path.stat().st_size, sha256=actual_hash)
    return path


def annotate(candidate: dict[str, Any], model: str, profile: str) -> None:
    payload = load(REPORT)
    for row in payload.get("results", []):
        if row.get("model") != model:
            continue
        row.update({
            "source_repo": REPO, "model_source": f"https://huggingface.co/{REPO}",
            "source_revision": REVISION, "source_published_at": PUBLISHED_AT,
            "quantization": candidate["key"].upper(), "weight_bytes": candidate["bytes"],
            "glm5_runtime_commit": "2a4a41238175cc5d0ee3e591865653e49096c782",
            "hardware_profile": profile,
        })
    temp = REPORT.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temp.replace(REPORT)


def benchmark(candidate: dict[str, Any], state: dict[str, Any]) -> None:
    env = os.environ.copy()
    env.update({"LLAMA_SERVER": str(RUNTIME / "bin/llama-server"),
                "LLAMA_BENCH": str(RUNTIME / "bin/llama-bench"),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
    for suffix, profile in PROFILES:
        model = f"glm53-reap50-{candidate['key']}-{suffix}"
        if row_complete(model):
            event(state, candidate, "benchmark_already_complete", model=model, profile=profile)
            continue
        event(state, candidate, "benchmark_started", model=model, profile=profile)
        # 2026-09-22 fix: this used to wrap the subprocess in its own
        # "flock -n /tmp/v100_exclusive.lock" on top of the outer
        # "flock /tmp/v100_exclusive.lock python3 run_glm53_reap50_cascade.py"
        # this script is meant to run under (see this repo's runbook / the
        # systemd unit). A second, non-blocking flock on the same lock file
        # from a child process can never acquire it while the parent already
        # holds it, so every invocation failed instantly with a generic
        # "returned non-zero exit status 1" before well_known_suite.py ever
        # started the server -- no server log, no result row, and two retries
        # were spent chasing stale/misleading error output before this was
        # found. run_qwen38_flash_next_gguf_cascade.py's run_profile() never
        # wrapped its own subprocess in flock for the same reason; match that
        # pattern and rely entirely on the caller's outer flock.
        subprocess.run(["python3", str(WKS), model,
                        "--profile", profile, "--out", str(REPORT)], cwd=ROOT, env=env, check=True)
        annotate(candidate, model, profile)
        event(state, candidate, "benchmark_complete", model=model, profile=profile)
        subprocess.run(["python3", str(RENDER)], cwd=ROOT, check=True)


def prune(candidate: dict[str, Any], state: dict[str, Any]) -> None:
    directory = target(candidate).parent
    if not complete_rows(candidate):
        raise RuntimeError(f"refusing to prune incomplete {candidate['key']}")
    size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    event(state, candidate, "prune_authorized", bytes=size,
          reinstall=["hf", "download", REPO, "--revision", REVISION, "--include", candidate["file"]])
    shutil.rmtree(directory)
    event(state, candidate, "pruned", bytes_freed=size)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=[candidate["key"] for candidate in CANDIDATES])
    parser.add_argument("--keep-weights", action="store_true")
    args = parser.parse_args()
    if not (RUNTIME / "bin/llama-server").exists():
        raise SystemExit(f"missing isolated GLM runtime: {RUNTIME}")
    state = load(STATE)
    selected = [candidate for candidate in CANDIDATES if not args.only or candidate["key"] == args.only]
    for candidate in selected:
        download(candidate, state)
        benchmark(candidate, state)
        if not args.keep_weights:
            prune(candidate, state)
    subprocess.run(["python3", str(RENDER)], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
