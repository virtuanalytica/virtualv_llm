#!/usr/bin/env python3
"""Benchmark local >=20B-class GGUF candidates on explicit GPU profiles.

Every profile pins physical devices through PCI_BUS_ID ordering, validates the
llama.cpp device probe, and samples exactly those devices during decode.  This
prevents an A4000/Ada run from being mislabeled as V100 (or vice versa) and
makes single-card, NVLink, and heterogeneous PCIe results distinguishable.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_suite  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SERVER = Path(os.environ.get(
    "LLAMA_SERVER", "/media/knight2/EDS2/tools/llama.cpp/build-v100/bin/llama-server",
))
BENCH = Path(os.environ.get(
    "LLAMA_BENCH", "/media/knight2/EDS2/tools/llama.cpp/build-v100/bin/llama-bench",
))
DEFAULT_OUT = ROOT / "reports/local_gguf_tp2_benchmark_20260917.json"
PORT = 18011

MODELS = {
    "qwen38-27b-q4": Path("/media/knight2/EDS2/models/llm/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf"),
    # 2026-09-22: 4-shard GGUF (unsloth UD-IQ3_XXS, ~104GiB total) -- point at
    # shard 1, llama.cpp discovers the other 3 via GGUF split metadata.
    "deepseek-v4-flash-0731-iq3xxs": Path(
        "/media/knight2/EDS2/models/llm/deepseek-v4-flash-0731-iq3xxs/UD-IQ3_XXS/"
        "DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf"),
    "deepseek-r1-qwen32b-q4": Path("/media/knight2/EDS2/models/llm/deepseek-r1-qwen32b/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf"),
    "qwen36-27b-iq3": Path("/media/knight2/EDS2/models/qwen3.6-27b-unsloth-ud-iq3-xxs.gguf"),
    "qwen35-27b-q4": Path("/media/knight2/EDS2/lmstudio-models/lmstudio-community/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf"),
    "gemma4-26b-a4b-q4": Path("/media/knight2/EDS2/lmstudio-models/lmstudio-community/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf"),
    "devstral-small2-24b-q4": Path("/media/knight2/EDS2/lmstudio-models/lmstudio-community/Devstral-Small-2-24B-Instruct-2512-GGUF/Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf"),
    "qwen25-72b-q4": Path("/media/knight2/EDS2/models/llm/qwen25-72b-q4km/qwen2.5-72b-instruct-q4_k_m-00001-of-00012.gguf"),
    "llama31-70b-instruct-q4": Path("/media/knight2/EDS2/models/llm/llama31-70b-instruct-q4/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf"),
    "glm45-air-106b-iq3": Path("/media/knight2/EDS2/models/llm/glm45-air-106b-iq3/GLM-4.5-Air-UD-IQ3_XXS-00001-of-00002.gguf"),
    "gpt-oss-120b-q4": Path("/media/knight2/EDS2/models/llm/gpt-oss-120b-q4/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf"),
    "gpt-oss-20b-q4": Path("/media/knight2/EDS2/models/llm/gpt-oss-20b-q4/gpt-oss-20b-Q4_K_M.gguf"),
    "llama3-70b-instruct-q4": Path("/media/knight2/EDS2/models/llm/llama3-70b-instruct-q4/Meta-Llama-3-70B-Instruct-Q4_K_M.gguf"),
    "qwen35-122b-a10b-iq3s": Path("/media/knight2/EDS2/models/llm/qwen35-122b-a10b-iq3s/Qwen3.5-122B-A10B-UD-IQ3_S.gguf"),
    "command-r-plus-104b-0824-iq3m": Path("/media/knight2/EDS2/models/llm/command-r-plus-104b-0824-iq3m/c4ai-command-r-plus-08-2024-IQ3_M.gguf"),
    "mixtral-8x22b-instruct-q3ks": Path("/media/knight2/EDS2/models/llm/mixtral-8x22b-instruct-q3ks/Mixtral-8x22B-Instruct-v0.1.Q3_K_S-00001-of-00003.gguf"),
    "wizardlm2-8x22b-iq3s": Path("/media/knight2/EDS2/models/llm/wizardlm2-8x22b-iq3s/WizardLM-2-8x22B-IQ3_S.gguf/WizardLM-2-8x22B-IQ3_S-00001-of-00005.gguf"),
    "glm53-flash-aj-iq2xxs": Path("/media/knight2/EDS2/models/llm/glm53-flash-aj-iq2xxs/AJ-IQ2_XXS/GLM-5.3-Flash-AJ-IQ2_XXS-00001-of-00002.gguf"),
    # GLM5-next requires the author's pinned fork (set by the cascade runner),
    # but otherwise uses the ordinary per-profile guard and telemetry path.
    "glm53-reap50-iq3m": Path("/media/knight2/EDS2/models/llm/glm53-reap50-iq3m/GLM-5.3-Flash-REAP50-IQ3_M.gguf"),
    "glm53-reap50-q3km": Path("/media/knight2/EDS2/models/llm/glm53-reap50-q3km/GLM-5.3-Flash-REAP50-Q3_K_M.gguf"),
    "glm53-reap50-iq4xs": Path("/media/knight2/EDS2/models/llm/glm53-reap50-iq4xs/GLM-5.3-Flash-REAP50-IQ4_XS.gguf"),
    "glm53-reap50-q4km": Path("/media/knight2/EDS2/models/llm/glm53-reap50-q4km/GLM-5.3-Flash-REAP50-Q4_K_M.gguf"),
    "glm53-reap50-iq3m-v100": Path("/media/knight2/EDS2/models/llm/glm53-reap50-iq3m/GLM-5.3-Flash-REAP50-IQ3_M.gguf"),
    "glm53-reap50-iq3m-allfour": Path("/media/knight2/EDS2/models/llm/glm53-reap50-iq3m/GLM-5.3-Flash-REAP50-IQ3_M.gguf"),
    "glm53-reap50-q3km-v100": Path("/media/knight2/EDS2/models/llm/glm53-reap50-q3km/GLM-5.3-Flash-REAP50-Q3_K_M.gguf"),
    "glm53-reap50-q3km-allfour": Path("/media/knight2/EDS2/models/llm/glm53-reap50-q3km/GLM-5.3-Flash-REAP50-Q3_K_M.gguf"),
    "glm53-reap50-iq4xs-v100": Path("/media/knight2/EDS2/models/llm/glm53-reap50-iq4xs/GLM-5.3-Flash-REAP50-IQ4_XS.gguf"),
    "glm53-reap50-iq4xs-allfour": Path("/media/knight2/EDS2/models/llm/glm53-reap50-iq4xs/GLM-5.3-Flash-REAP50-IQ4_XS.gguf"),
    "glm53-reap50-q4km-v100": Path("/media/knight2/EDS2/models/llm/glm53-reap50-q4km/GLM-5.3-Flash-REAP50-Q4_K_M.gguf"),
    "glm53-reap50-q4km-allfour": Path("/media/knight2/EDS2/models/llm/glm53-reap50-q4km/GLM-5.3-Flash-REAP50-Q4_K_M.gguf"),
}

PROFILES = {
    "single-v100": {"visible": "1", "physical": [1], "split_mode": "none", "expected_gpu": "Tesla V100-SXM2-32GB"},
    "single-v100-2": {"visible": "2", "physical": [2], "split_mode": "none", "expected_gpu": "Tesla V100-SXM2-32GB"},
    "dual-layer": {"visible": "1,2", "physical": [1, 2], "split_mode": "layer", "expected_gpu": "Tesla V100-SXM2-32GB"},
    "dual-tensor": {"visible": "1,2", "physical": [1, 2], "split_mode": "tensor", "expected_gpu": "Tesla V100-SXM2-32GB"},
    # 2026-09-19: GPU0 (RTX A4000, Ampere) cleared for benchmarking (was the live
    # production server for qwen36-27b-iq3; user granted use once that moved).
    # Single-card, no split needed -- for candidates small enough to fit 15GB alone.
    "single-a4000": {"visible": "0", "physical": [0], "split_mode": "none", "expected_gpu": "NVIDIA RTX A4000"},
    "single-ada": {"visible": "3", "physical": [3], "split_mode": "none", "expected_gpu": "NVIDIA RTX 4000 Ada Generation"},
    # 2026-09-19: GPU0 (A4000, 15GB) + GPU3 (RTX 4000 Ada, 20GB) combined via
    # layer split (no NVLink between these two, unlike the V100 pair -- layer
    # split minimizes cross-card traffic vs tensor split). Mixed-architecture
    # pair (Ampere + Ada Lovelace), used for >15GB candidates that don't fit
    # on GPU0 alone. GPU3 stays the live desktop -- this profile shares it,
    # doesn't reserve it exclusively.
    "dual-a4000ada": {
        "visible": "0,3", "physical": [0, 3], "split_mode": "layer",
        "expected_gpu": None,  # mixed hardware, checked per-device below instead of a single match
        "allowed_gpus": ["NVIDIA RTX A4000", "NVIDIA RTX 4000 Ada Generation"],
    },
    "all-four-layer": {
        "visible": "0,1,2,3", "physical": [0, 1, 2, 3], "split_mode": "layer",
        "expected_gpu": None,
        "allowed_gpus": [
            "NVIDIA RTX A4000", "Tesla V100-SXM2-32GB",
            "Tesla V100-SXM2-32GB", "NVIDIA RTX 4000 Ada Generation",
        ],
    },
}

PERF_PROMPT = (
    "Write a continuous technical explanation of point-in-time validation for financial machine learning. "
    "Use complete sentences and keep writing until the token budget ends."
)


def request_json(path: str, payload: dict[str, Any] | None = None, timeout: int = 240) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode()
    req = Request(f"http://127.0.0.1:{PORT}{path}", data=body,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def wait_ready(proc: subprocess.Popen[Any], seconds: int = 180) -> None:
    deadline = time.monotonic() + seconds
    last: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {proc.returncode}")
        try:
            request_json("/health", timeout=3)
            return
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(1)
    raise RuntimeError(f"server not healthy after {seconds}s: {last}")


def gpu_sample(physical_gpus: tuple[int, ...] | list[int] = (1, 2)) -> dict[int, dict[str, float]]:
    cmd = ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu",
           "--format=csv,noheader,nounits"]
    text = subprocess.check_output(cmd, text=True)
    result: dict[int, dict[str, float]] = {}
    for line in text.splitlines():
        fields = [v.strip() for v in line.split(",")]
        if len(fields) != 5:
            continue
        idx = int(fields[0])
        if idx in physical_gpus:
            result[idx] = {"util_pct": float(fields[1]), "memory_mib": float(fields[2]),
                           "power_w": float(fields[3]), "temperature_c": float(fields[4])}
    return result


def monitor_gpu(stop: threading.Event, samples: list[dict[int, dict[str, float]]],
                physical_gpus: tuple[int, ...] | list[int] = (1, 2)) -> None:
    while not stop.is_set():
        try:
            samples.append(gpu_sample(physical_gpus))
        except Exception:
            pass
        stop.wait(0.25)


def telemetry_summary(samples: list[dict[int, dict[str, float]]],
                      physical_gpus: tuple[int, ...] | list[int] = (1, 2)) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for idx in physical_gpus:
        rows = [sample[idx] for sample in samples if idx in sample]
        out[str(idx)] = {
            "samples": len(rows),
            "max_util_pct": max((r["util_pct"] for r in rows), default=None),
            "mean_util_pct": round(sum(r["util_pct"] for r in rows) / len(rows), 2) if rows else None,
            "max_memory_mib": max((r["memory_mib"] for r in rows), default=None),
            "max_power_w": max((r["power_w"] for r in rows), default=None),
            "max_temperature_c": max((r["temperature_c"] for r in rows), default=None),
        }
    return out


def complete(model: str, prompt: str, max_tokens: int, ignore_eos: bool = False) -> dict[str, Any]:
    return request_json("/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 42,
        "max_tokens": max_tokens,
        "ignore_eos": ignore_eos,
        "chat_template_kwargs": {"enable_thinking": False},
    })


def sharded_model_bytes(model_path: Path) -> int:
    """Return all shard bytes for a GGUF entrypoint, or the single-file size."""
    stem = model_path.name
    if "-00001-of-" not in stem:
        return model_path.stat().st_size
    prefix, suffix = stem.split("-00001-of-", 1)
    total = suffix.split(".", 1)[0]
    shards = sorted(model_path.parent.glob(f"{prefix}-*-of-{total}.gguf"))
    return sum(path.stat().st_size for path in shards)


def benchmark(name: str, model_path: Path, profile_name: str, *, auto_fit: bool = False,
              ready_timeout: int = 180, request_timeout: int = 240) -> dict[str, Any]:
    profile = PROFILES[profile_name]
    log_path = Path(f"/tmp/{name}-{profile_name}.log")
    env = os.environ.copy()
    # CUDA's default FASTEST_FIRST order is not the nvidia-smi order on node2:
    # without this setting ordinal 1 is the RTX A4000 and silently contaminates
    # a nominal `1,2` run. PCI_BUS_ID makes 1,2 the two physical V100s.
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(profile["visible"])
    env["GGML_CUDA_P2P"] = "1"
    visible = subprocess.check_output([str(BENCH), "--list-devices"], env=env,
                                      text=True, stderr=subprocess.STDOUT)
    visible_rows = [line.strip() for line in visible.splitlines() if line.strip().startswith("CUDA")]
    expected_gpu = profile["expected_gpu"]
    allowed_gpus = profile.get("allowed_gpus")
    if expected_gpu is not None:
        device_ok = (len(visible_rows) == len(profile["physical"])
                     and all(expected_gpu in line for line in visible_rows))
    else:
        device_ok = (len(visible_rows) == len(profile["physical"])
                     and allowed_gpus is not None
                     and len(allowed_gpus) == len(visible_rows)
                     and all(expected in line for expected, line in zip(allowed_gpus, visible_rows)))
    if not device_ok:
        raise RuntimeError(f"device guard failed for {profile_name}:\n{visible}")
    command = [str(SERVER), "--model", str(model_path), "--alias", name,
               "--host", "127.0.0.1", "--port", str(PORT), "--ctx-size", "8192",
               "--parallel", "1", "--split-mode", str(profile["split_mode"]),
               "--main-gpu", "0", "--flash-attn", "on",
               "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--jinja"]
    if auto_fit:
        margins = ",".join("1024" for _ in profile["physical"])
        command.extend(["--fit", "on", "--fit-ctx", "4096", "--fit-target", margins])
    else:
        command.extend(["--gpu-layers", "99"])
    if name.startswith("glm53-flash-"):
        # GLM-5.3 ends assistant turns with <|user|> or <|observation|>, but
        # existing GGUF headers retain only <|endoftext|> as an EOG token.
        # Without these documented tokenizer-only overrides the model can
        # invent another user turn, corrupting both accuracy and throughput.
        command.extend([
            "--override-kv", (
                "tokenizer.ggml.eot_token_id=int:154827,"
                "tokenizer.ggml.eom_token_id=int:154829"
            ),
        ])
    if profile["split_mode"] == "tensor":
        command.extend(["--tensor-split", ",".join("1" for _ in profile["physical"])])
    started = time.time()
    with log_path.open("w") as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            wait_ready(proc, ready_timeout)
            loaded = gpu_sample(profile["physical"])
            benchmarks = eval_suite.run_tasks(
                lambda prompt, max_tokens: request_json("/v1/chat/completions", {
                    "model": name, "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0, "seed": 42, "max_tokens": max_tokens,
                    "chat_template_kwargs": {"enable_thinking": False},
                    **({"stop": ["<|user|>", "<|observation|>"]}
                       if name.startswith("glm53-flash-") else {}),
                }, timeout=request_timeout)
            )
            samples: list[dict[int, dict[str, float]]] = []
            stop = threading.Event()
            thread = threading.Thread(
                target=monitor_gpu, args=(stop, samples, profile["physical"]), daemon=True,
            )
            thread.start()
            try:
                perf = request_json("/v1/chat/completions", {
                    "model": name, "messages": [{"role": "user", "content": PERF_PROMPT}],
                    "temperature": 0, "seed": 42, "max_tokens": 256, "ignore_eos": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                }, timeout=request_timeout)
            finally:
                stop.set()
                thread.join(timeout=2)
            timings = perf.get("timings", {})
            gpu = telemetry_summary(samples, profile["physical"])
            selected_active = all((gpu[str(i)].get("max_util_pct") or 0) >= 10
                                  for i in profile["physical"])
            physical = profile["physical"]
            if len(physical) == 1:
                interconnect = "local GPU"
            elif physical == [1, 2]:
                interconnect = "NVLink"
            elif set(physical) == {0, 1, 2, 3}:
                interconnect = "mixed: NVLink between V100s, PCIe to RTX GPUs"
            else:
                interconnect = "PCIe"
            return {
                "model": name,
                "profile": profile_name,
                "path": str(model_path),
                "bytes": sharded_model_bytes(model_path),
                "auto_fit": auto_fit,
                "runtime": str(SERVER),
                "topology": (f"physical GPU {physical}, split={profile['split_mode']}, "
                             "CUDA_DEVICE_ORDER=PCI_BUS_ID, "
                             f"interconnect={interconnect}"),
                "visible_device_probe": visible,
                "loaded_gpu": loaded,
                "selected_gpus_active": selected_active,
                "selected_v100s_active": selected_active,  # legacy report field
                "both_v100_active": selected_active and len(profile["physical"]) == 2,
                "gpu_decode_telemetry": gpu,
                "benchmarks": benchmarks,
                "benchmark_mean_score": benchmarks["_mean_score"],
                "benchmark_task_count": len(eval_suite.TASKS),
                "completion_tokens": perf.get("usage", {}).get("completion_tokens"),
                "completion_tokens_per_second": timings.get("predicted_per_second"),
                "prompt_tokens_per_second": timings.get("prompt_per_second"),
                "elapsed_seconds_including_load_and_tests": round(time.time() - started, 2),
                "server_log": str(log_path),
            }
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", choices=MODELS, default=None)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=["dual-layer"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--auto-fit", action="store_true",
                        help="let llama.cpp offload the maximum safe number of layers per profile")
    parser.add_argument("--ready-timeout", type=int, default=180)
    parser.add_argument("--request-timeout", type=int, default=240)
    parser.add_argument("--resume", action="store_true",
                        help="retain successful model/profile rows already in --out")
    args = parser.parse_args()
    if not SERVER.exists():
        raise SystemExit(f"missing llama-server: {SERVER}")
    if not BENCH.exists():
        raise SystemExit(f"missing llama-bench: {BENCH}")
    selected = args.models or list(MODELS)
    missing = [str(MODELS[name]) for name in selected if not MODELS[name].exists()]
    if missing:
        raise SystemExit("missing model(s): " + ", ".join(missing))
    payload: dict[str, Any] = {
        "contract": f"same {len(eval_suite.TASKS)}-task eval_suite battery plus forced 256-token decode",
        "benchmark_task_ids": [task.id for task in eval_suite.TASKS],
        "physical_gpu_inventory": {
            "0": "NVIDIA RTX A4000", "1": "Tesla V100-SXM2-32GB",
            "2": "Tesla V100-SXM2-32GB", "3": "NVIDIA RTX 4000 Ada Generation",
        },
        "cuda_device_order": "PCI_BUS_ID",
        "profiles": args.profiles,
        "note": "each result row records its selected physical GPUs; compare throughput by profile",
        "results": [],
    }
    if args.resume and args.out.exists():
        previous = json.loads(args.out.read_text())
        payload["results"] = previous.get("results", [])
    completed = {(row.get("model"), row.get("profile")) for row in payload["results"]
                 if not row.get("error")}
    failures = 0
    for name in selected:
        for profile in args.profiles:
            if (name, profile) in completed:
                print(f"SKIP {name} [{profile}]: successful result already present", flush=True)
                continue
            print(f"START {name} [{profile}]", flush=True)
            try:
                result = benchmark(name, MODELS[name], profile, auto_fit=args.auto_fit,
                                   ready_timeout=args.ready_timeout,
                                   request_timeout=args.request_timeout)
                payload["results"] = [row for row in payload["results"]
                                      if (row.get("model"), row.get("profile")) != (name, profile)]
                payload["results"].append(result)
                print(f"DONE {name} [{profile}]: {result['completion_tokens_per_second']:.2f} t/s, "
                      f"benchmark_mean={result['benchmark_mean_score']:.2f} "
                      f"({result['benchmark_task_count']} tasks), "
                      f"selected_active={result['selected_v100s_active']}", flush=True)
            except Exception as exc:
                failures += 1
                payload["results"] = [row for row in payload["results"]
                                      if (row.get("model"), row.get("profile")) != (name, profile)]
                payload["results"].append({"model": name, "profile": profile,
                                           "path": str(MODELS[name]),
                                           "error": f"{type(exc).__name__}: {exc}"})
                print(f"FAIL {name} [{profile}]: {type(exc).__name__}: {exc}", flush=True)
            write_result(args.out, payload)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
