#!/usr/bin/env python3
"""Resumable Qwen3.8-Flash-Next GGUF cascade for SM70-compatible llama.cpp.

The former W4A16 vLLM candidates failed their earlier compressed-tensors
backend on the V100 pair (SM70).  That does not rule out 1Cat-vLLM's separate
SM70 W4A16 routes; they are deliberately outside this GGUF runner.  This
runner evaluates two standard GGUF candidates one at a time, records every
durable result before pruning, and never uses the unsafe unattended V100
tensor-split mode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
MODELS = Path("/media/knight2/EDS2/models/llm")
REPORTS = ROOT / "reports"
REPORT = REPORTS / "well_known_suite_20260917.json"
STATE = REPORTS / "qwen38_flash_next_gguf_cascade_20260922.json"
WKS = ROOT / "scripts/benchmarks/well_known_suite.py"
RENDER = ROOT / "scripts/benchmarks/build_dual_v100_html.py"
SERVER = Path("/media/knight2/EDS2/tools/llama.cpp/build-v100/bin/llama-server")
REPO = "agentionai/Qwen3.8-Flash-Next-AP-GGUF"
REVISION = "0061a60e46ad672a73cec31cb627dc841d4760a4"
PUBLISHED_AT = "2026-09-11T22:34:02Z"
RESERVE_BYTES = 24_000_000_000
SERVICES = ("local-chat-qwen38.service", "llama-qwen.service")

# Standard GGUF quant types, selected in increasing quality/memory order.  The
# V100-only Q4 trial begins at 2K context because its published VRAM budget is
# close to the pair's 64 GiB total; a failed smoke test is a valid result, not
# a reason to invent a full-suite score.
CANDIDATES: tuple[dict[str, Any], ...] = (
    {"key": "ap-iq4xs", "file": "AP-IQ4_XS/Qwen3.8-Flash-Next-AP-IQ4_XS.gguf",
     "published_vram_gib": 57.42, "published_size_gib": 84.24, "v100_context": 4096},
    # ap-q4km removed 2026-09-22: its v100 profile is a structural VRAM-fit
    # failure (needs 61.22 GiB, the 2xV100 pair has 64 GiB combined including
    # overhead -- OOM'd even at the smallest 2K context), so it can never get
    # a valid composite() score (that function requires the v100 profile
    # specifically). Its allfour result (91.75%, comparable to ap-iq4xs) is
    # preserved permanently in well_known_suite_20260917.json; weights were
    # deleted to reclaim 88 GiB. Do not re-add without first fixing the
    # composite-scoring gate to accept an allfour-only result.
    {"key": "ap-iq2s", "file": "AP-IQ2_S/Qwen3.8-Flash-Next-AP-IQ2_S.gguf",
     "published_vram_gib": 49.21, "published_size_gib": 76.03, "v100_context": 4096},
)
PROFILES = (
    ("v100", "1,2", "2× Tesla V100-SXM2-32GB · NVLink · layer split", 18031),
    ("allfour", "0,1,2,3", "RTX A4000 + 2× V100 NVLink + RTX 4000 Ada · layer split", 18032),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else default


def atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def event(state: dict[str, Any], candidate: dict[str, Any], kind: str, **data: Any) -> None:
    state.setdefault("events", []).append({"at": now(), "quant": candidate["key"], "event": kind, **data})
    atomic(STATE, state)


def target(candidate: dict[str, Any]) -> Path:
    return MODELS / f"qwen38-flash-next-{candidate['key']}" / candidate["file"]


def model_id(candidate: dict[str, Any], profile: str) -> str:
    return f"qwen38-flash-next-{candidate['key']}-{profile}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def downloaded(candidate: dict[str, Any]) -> bool:
    return target(candidate).is_file() and target(candidate).stat().st_size > 50_000_000_000


def repair_legacy_download_path(candidate: dict[str, Any]) -> None:
    """Move the first-run nested HF layout into the canonical model path.

    Earlier revisions passed ``target.parent`` to ``hf download`` while also
    including the AP directory in ``--include``.  That is a path construction
    bug, not a partial model.  Rename only the verified large GGUF and leave
    Hugging Face cache metadata untouched; never overwrite a canonical file.
    """
    path = target(candidate)
    nested = path.parent / path.parent.name / path.name
    if path.exists() or not nested.is_file() or nested.stat().st_size <= 50_000_000_000:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    nested.replace(path)


def download(candidate: dict[str, Any], state: dict[str, Any]) -> Path:
    path = target(candidate)
    repair_legacy_download_path(candidate)
    if downloaded(candidate):
        return path
    expected = int(candidate["published_size_gib"] * 1024 ** 3)
    free = shutil.disk_usage(MODELS).free
    if free < expected + RESERVE_BYTES:
        raise RuntimeError(f"insufficient disk for {candidate['key']}: {free} < {expected + RESERVE_BYTES}")
    path.parent.mkdir(parents=True, exist_ok=True)
    event(state, candidate, "download_started", repo=REPO, revision=REVISION, free_before=free)
    subprocess.run(["hf", "download", REPO, "--revision", REVISION, "--include", candidate["file"],
                    "--local-dir", str(path.parents[1])], check=True)
    if not downloaded(candidate):
        raise RuntimeError(f"download incomplete: {path}")
    event(state, candidate, "download_verified", bytes=path.stat().st_size, sha256=sha256(path))
    return path


def services_stop() -> list[str]:
    stopped: list[str] = []
    for service in SERVICES:
        if subprocess.run(["systemctl", "--user", "is-active", "--quiet", service]).returncode == 0:
            subprocess.run(["systemctl", "--user", "stop", service], check=True)
            stopped.append(service)
    return stopped


def services_restore(stopped: list[str]) -> None:
    for service in stopped:
        subprocess.run(["systemctl", "--user", "start", service], check=False)


def assert_no_compute_contexts(devices: str) -> None:
    output = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if output:
        raise RuntimeError(f"GPU compute context present before {devices} benchmark: {output}")


def ready(proc: subprocess.Popen[Any], port: int, log: Path) -> None:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited {proc.returncode}; see {log}")
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"llama-server readiness timeout; see {log}")


def complete(model: str) -> bool:
    row = next((r for r in load(REPORT, {"results": []}).get("results", []) if r.get("model") == model), None)
    return bool(row and not row.get("error") and isinstance(row.get("completion_tokens_per_second"), (int, float))
                and (row.get("bbh") or {}).get("mean_accuracy") is not None
                and (row.get("mmlu_sample") or {}).get("mean_accuracy") is not None
                and (row.get("humaneval") or {}).get("pass_at_1") is not None)


def annotate(model: str, candidate: dict[str, Any], profile: str, topology: str, context: int, digest: str) -> None:
    payload = load(REPORT, {"results": []})
    for row in payload["results"]:
        if row.get("model") == model:
            row.update({
                "engine": "llama.cpp qwen4exp", "hardware_profile": profile, "topology": topology,
                "source_repo": REPO, "model_source": f"https://huggingface.co/{REPO}",
                "source_revision": REVISION, "source_published_at": PUBLISHED_AT,
                "quantization": candidate["key"].upper(), "weight_bytes": target(candidate).stat().st_size,
                "weight_sha256": digest, "context_tokens": context,
                "split_policy": "layer; V100 tensor split excluded for unattended long suite",
            })
    atomic(REPORT, payload)


def run_profile(candidate: dict[str, Any], state: dict[str, Any], profile: str, devices: str,
                topology: str, port: int) -> None:
    model = model_id(candidate, profile)
    if complete(model):
        return
    path = target(candidate)
    context = candidate["v100_context"] if profile == "v100" else 4096
    log = REPORTS / "llama_logs" / f"{model}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    stopped: list[str] = []
    proc: subprocess.Popen[Any] | None = None
    try:
        stopped = services_stop()
        assert_no_compute_contexts(devices)
        event(state, candidate, "profile_started", model=model, devices=devices, context_tokens=context)
        env = os.environ.copy()
        env.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": devices})
        command = [str(SERVER), "--model", str(path), "--alias", "qwen38-flash-next",
                   "--host", "127.0.0.1", "--port", str(port), "--ctx-size", str(context),
                   "--parallel", "1", "--split-mode", "layer", "--gpu-layers", "99",
                   "--flash-attn", "on", "--reasoning", "off", "--cache-type-k", "q8_0",
                   "--cache-type-v", "q8_0", "--jinja"]
        with log.open("w") as handle:
            proc = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        ready(proc, port, log)
        subprocess.run(["python3", str(WKS), model, "--external-url", f"http://127.0.0.1:{port}",
                        "--external-model", "qwen38-flash-next", "--physical-gpus", devices,
                        "--topology", topology, "--engine", "llama.cpp qwen4exp", "--out", str(REPORT)],
                       cwd=ROOT, check=True)
        annotate(model, candidate, profile, topology, context, sha256(path))
        event(state, candidate, "profile_complete", model=model)
    except Exception as exc:
        payload = load(REPORT, {"results": []})
        payload["results"] = [r for r in payload["results"] if r.get("model") != model] + [{
            "model": model, "error": f"{type(exc).__name__}: {exc}", "engine": "llama.cpp qwen4exp",
            "hardware_profile": profile, "topology": topology, "source_repo": REPO,
            "source_revision": REVISION, "quantization": candidate["key"].upper(),
        }]
        atomic(REPORT, payload)
        event(state, candidate, "profile_failed", model=model, error=str(exc))
    finally:
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        services_restore(stopped)
        subprocess.run(["python3", str(RENDER)], cwd=ROOT, check=False)


def composite(candidate: dict[str, Any]) -> float | None:
    rows = load(REPORT, {"results": []}).get("results", [])
    row = next((r for r in rows if r.get("model") == model_id(candidate, "v100") and not r.get("error")), None)
    if not row:
        return None
    values = [
        row.get("gsm8k", {}).get("exact_match,flexible-extract"),
        (row.get("bbh") or {}).get("mean_accuracy"),
        (row.get("mmlu_sample") or {}).get("mean_accuracy"),
        (row.get("humaneval") or {}).get("pass_at_1"),
    ]
    return sum(values) / 4 if all(isinstance(value, (int, float)) for value in values) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=[c["key"] for c in CANDIDATES])
    parser.add_argument("--keep-all", action="store_true", help="do not prune the lower-scoring completed quant")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not SERVER.is_file():
        raise SystemExit(f"missing qwen4exp-capable llama-server: {SERVER}")
    state = load(STATE, {"model": "Qwen3.8-Flash-Next", "repo": REPO, "revision": REVISION,
                         "candidates": CANDIDATES, "events": []})
    selected = [candidate for candidate in CANDIDATES if not args.only or candidate["key"] == args.only]
    if args.dry_run:
        print(json.dumps({"selected": selected, "profiles": PROFILES, "server": str(SERVER)}, indent=2))
        return 0
    for candidate in selected:
        download(candidate, state)
        for profile in PROFILES:
            run_profile(candidate, state, *profile)
    if not args.only and not args.keep_all:
        scored = [(composite(candidate), candidate) for candidate in CANDIDATES]
        scored = [(score, candidate) for score, candidate in scored if score is not None]
        if len(scored) == len(CANDIDATES):
            winner = max(scored, key=lambda item: item[0])[1]
            for _, candidate in scored:
                if candidate == winner:
                    continue
                directory = target(candidate).parents[1]
                freed = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
                shutil.rmtree(directory)
                event(state, candidate, "pruned_lower_score", bytes_freed=freed,
                      winner=winner["key"], winner_composite=composite(winner))
    subprocess.run(["python3", str(RENDER)], cwd=ROOT, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
