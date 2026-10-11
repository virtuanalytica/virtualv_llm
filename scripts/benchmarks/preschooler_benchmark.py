#!/usr/bin/env python3
"""Isolated 3–4B loader canary and opt-in private VirtualV battery.

Never preempts an occupied GPU. Full evaluation delegates to the active private
runner, whose holdout ownership and GPU energy checks remain authoritative.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
ROSTER = ROOT / "config/preschooler_teacher_roster.json"
STORE = Path("/media/knight2/claude-data/knight1/models/preschooler-teachers")
VLLM = Path("/media/knight2/claude-data/knight1/1cat-vllm-v1.5.1/env/bin/vllm")
MINISTRAL_PYTHON = Path("/media/knight2/claude-data/knight1/1cat-vllm-v1.5.1/env/bin/python")
MINISTRAL_TRANSFORMERS_COMPAT = Path(
    "/media/knight2/claude-data/knight1/1cat-vllm-v1.5.1/compat-transformers-5.16.1")
CANARY = (("What is 17 plus 25? Reply with only the number.", "42"),
          ("Translate 'good morning' to Dutch. Reply with only the translation.", "goedemorgen"),
          ("What is the capital of France? Reply with only the city.", "paris"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gpu_free(index: int) -> bool:
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used",
                                   "--format=csv,noheader,nounits"], text=True)
    memory = {int(row.split(",")[0].strip()): int(row.split(",")[1].strip())
              for row in raw.splitlines()}
    if index not in memory:
        raise ValueError("physical GPU index is unavailable")
    return memory[index] < 1024


def get_json(url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urlopen(req, timeout=120) as response:
        return json.load(response)


def wait_healthy(base: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during load: {process.returncode}")
        try:
            with urlopen(base + "/health", timeout=3) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError):
            pass
        time.sleep(2)
    raise TimeoutError("server did not become healthy")


def canary(base: str, alias: str) -> dict:
    rows = []
    for prompt, expected in CANARY:
        start = time.monotonic()
        data = get_json(base + "/v1/chat/completions", {
            "model": alias, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 48})
        answer = data["choices"][0]["message"].get("content") or ""
        normalized = answer.strip().strip(". ").casefold()
        rows.append({"prompt": prompt, "expected": expected, "answer": answer,
                     "correct": normalized == expected,
                     "wall_seconds": round(time.monotonic() - start, 3)})
    t0 = time.monotonic()
    probe = get_json(base + "/v1/completions", {
        "model": alias, "prompt": "Explain a merge join with an example.",
        "temperature": 0, "max_tokens": 256, "ignore_eos": True})
    duration = time.monotonic() - t0
    tokens = (probe.get("usage") or {}).get("completion_tokens")
    def readable(answer: str) -> bool:
        words = answer.strip().split()
        return bool(words) and (len(words) <= 8 or max(words.count(word) for word in set(words)) / len(words) < 0.8)

    return {"status": "passed" if all(readable(x["answer"]) for x in rows) and tokens == 256 else "failed",
            "quality": rows, "probe": {"completion_tokens": tokens,
                                      "wall_seconds_including_prefill": round(duration, 3),
                                      "wall_tokens_per_second": round(tokens / duration, 3) if tokens else None},
            "note": "Diagnostic output-integrity canary; exact matches are recorded but do not gate the private quality run."}


def main() -> int:
    # Private scorer output can contain sealed prompts and answer keys.
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--port", type=int, default=8930)
    parser.add_argument("--engine", choices=("transformers", "1cat-vllm"), default="transformers")
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--full", action="store_true", help="run the private battery after a passed canary")
    parser.add_argument("--private-runner", type=Path, help="private_endpoint_suite.py in the active benchmark worktree")
    parser.add_argument("--monitor", type=Path, help="optional llmwatch/metrics sidecar executable")
    args = parser.parse_args()
    roster = json.loads(ROSTER.read_text())
    model = next((m for m in roster["models"] if m["alias"] == args.alias), None)
    if model is None:
        parser.error("unknown model alias")
    if model.get("chat_mode") == "thinking_only":
        parser.error("thinking-only checkpoint requires a separate thinking-on protocol; "
                     "the fixed reasoning-off profile is invalid")
    snapshot = args.store / args.alias
    manifest_file = snapshot / "snapshot-manifest.json"
    if not manifest_file.is_file():
        parser.error("verified snapshot missing; run preschooler_download_roster.py first")
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("revision") != model["revision"] or manifest.get("status") != "download_verified_loader_untested":
        parser.error("snapshot does not match pinned roster")
    if not gpu_free(args.gpu_index):
        parser.error("GPU has more than 1 GiB allocated; no preemption permitted")
    if model["family"] in {"phi4flash", "minicpm3"}:
        parser.error("remote/custom model code requires a separately reviewed runtime adapter")
    if args.full and (not args.private_runner or not args.private_runner.is_file()):
        parser.error("--full requires an existing --private-runner")
    args.out.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.out.chmod(0o700)
    base = f"http://127.0.0.1:{args.port}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu_index), CUDA_DEVICE_ORDER="PCI_BUS_ID")
    if model["family"] == "ministral3":
        if not (MINISTRAL_TRANSFORMERS_COMPAT / "transformers/__init__.py").is_file():
            parser.error("Ministral3 requires the isolated Transformers 5.16.1 compatibility overlay")
        if not MINISTRAL_PYTHON.is_file():
            parser.error("Ministral3 compatibility Python is missing")
        env["PYTHONPATH"] = str(MINISTRAL_TRANSFORMERS_COMPAT) + os.pathsep + env.get("PYTHONPATH", "")
    if args.engine == "transformers":
        python = MINISTRAL_PYTHON if model["family"] == "ministral3" else Path(sys.executable)
        command = [str(python), str(ROOT / "scripts/benchmarks/preschooler_transformers_server.py"),
                   "--model-path", str(snapshot), "--alias", args.alias,
                   "--port", str(args.port)]
    else:
        command = [str(VLLM), "serve", str(snapshot), "--served-model-name", args.alias,
                   "--host", "127.0.0.1", "--port", str(args.port),
                   "--dtype", "auto", "--tensor-parallel-size", "1",
                   "--max-model-len", "4096", "--gpu-memory-utilization", "0.75",
                   "--max-num-seqs", "1", "--generation-config", "vllm",
                   "--attention-backend", "TRITON_ATTN", "--enforce-eager"]
        if model["family"] == "ministral3":
            command.append("--language-model-only")
    result = {"schema": "preschooler-small-model-benchmark/v1", "alias": args.alias,
              "model_repo": model["repo"], "revision": model["revision"],
              "gpu_index": args.gpu_index, "started_at": now(), "status": "running",
              "engine": args.engine,
              "serve_command": command, "snapshot_manifest": str(manifest_file),
              "monitor": "llmwatch" if args.monitor else "none"}
    if model["family"] == "ministral3":
        result["runtime_compat"] = {"transformers": "5.16.1", "accelerate": "1.12.0",
                                    "overlay": str(MINISTRAL_TRANSFORMERS_COMPAT),
                                    "mode": "BF16 dequantized text-only" if args.engine == "transformers"
                                    else "1Cat FP8 experimental"}
    status_file = args.out / "status.json"
    status_file.write_text(json.dumps(result, indent=2) + "\n")
    process = None
    monitor = None
    try:
        with (args.out / "server.log").open("w") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            wait_healthy(base, process, 240)
            if args.monitor:
                with (args.out / "monitor.log").open("w") as monitor_log:
                    monitor = subprocess.Popen([sys.executable, str(args.monitor),
                                                "--base", base, "--out", str(args.out / "system-stats.jsonl")],
                                               stdout=monitor_log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
            result["canary"] = canary(base, args.alias)
            (args.out / "canary.json").write_text(json.dumps(result["canary"], indent=2) + "\n")
            if result["canary"]["status"] != "passed":
                raise RuntimeError("diagnostic canary failed; full battery blocked")
            if args.full:
                private_dir = args.out / "private"
                private_dir.mkdir(exist_ok=True, mode=0o700)
                private_dir.chmod(0o700)
                private_command = [sys.executable, str(ROOT / "scripts/benchmarks/preschooler_private_bridge.py"),
                                   "--suite-path", str(args.private_runner), "--base", base,
                                   "--alias", args.alias, "--model-id", args.alias,
                                   "--gpu-indices", str(args.gpu_index),
                                   "--log-dir", str(private_dir)]
                with (args.out / "private-runner.log").open("w") as private_log:
                    completed = subprocess.run(private_command, stdout=private_log,
                                               stderr=subprocess.STDOUT, check=False)
                if completed.returncode:
                    raise RuntimeError(f"private suite failed with exit {completed.returncode}")
            result["status"] = "complete_private" if args.full else "complete_canary_only"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        raise
    except KeyboardInterrupt:
        result["status"] = "interrupted"
        result["error"] = "operator_interrupted"
        raise
    finally:
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                monitor.kill()
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        result["finished_at"] = now()
        status_file.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
