#!/usr/bin/env python3
"""Measure independent A4000 + RTX 4000 Ada serving, serial and concurrent.

The two production services each have one explicitly pinned physical GPU.  This
benchmark does not call their combined throughput an ensemble quality score: it
measures the capacity gain when two different LLMs serve independent requests
in parallel.  Quality remains reported by the well-known-suite/mixture rows.
"""
from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/additional_gpu_serving_matrix_20260921.json"
PROMPT = (
    "Write a continuous technical explanation of point-in-time validation for financial "
    "machine learning. Use complete sentences and keep writing until the token budget ends."
)
SERVICES = {
    "a4000_qwen36": {
        "url": "http://127.0.0.1:11435", "physical_gpu": 0,
        "gpu": "NVIDIA RTX A4000", "service": "llama-qwen.service",
        "expected_model": "qwen3.6-unsloth-a4000",
    },
    "ada_qwen38": {
        "url": "http://127.0.0.1:8011", "physical_gpu": 3,
        "gpu": "NVIDIA RTX 4000 Ada Generation", "service": "local-chat-qwen38.service",
        "expected_model": "qwen38-primary",
    },
}


def request_json(url: str, path: str, payload: dict[str, Any] | None = None,
                 timeout: int = 300) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode()
    req = Request(url + path, data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def physical_gpu_processes() -> dict[int, list[int]]:
    query = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    buses = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits"], text=True,
    )
    by_bus = {bus.strip().lower(): int(idx.strip()) for idx, bus in
              (line.split(",", 1) for line in buses.splitlines() if line.strip())}
    result: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
    for line in query.splitlines():
        if not line.strip():
            continue
        bus, pid = (part.strip() for part in line.split(",", 1))
        idx = by_bus.get(bus.lower())
        if idx is not None:
            result[idx].append(int(pid))
    return result


def completion(spec: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    response = request_json(spec["url"], "/v1/chat/completions", {
        "model": spec["expected_model"],
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0, "seed": 42, "max_tokens": 256, "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    wall = time.perf_counter() - started
    tokens = int((response.get("usage") or {}).get("completion_tokens") or 0)
    timings = response.get("timings") or {}
    return {
        "completion_tokens": tokens,
        "wall_seconds": wall,
        "wall_tokens_per_second": tokens / wall if wall else None,
        "server_tokens_per_second": timings.get("predicted_per_second"),
        "prompt_tokens_per_second": timings.get("prompt_per_second"),
    }


def mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
    return sum(values) / len(values) if values else None


def service_exec_start(service: str) -> str:
    return subprocess.check_output(
        ["systemctl", "--user", "show", service, "-p", "ExecStart", "--value"], text=True,
    ).strip()


def main() -> int:
    models = {}
    for name, spec in SERVICES.items():
        listing = request_json(spec["url"], "/v1/models")
        ids = [row.get("id") for row in listing.get("data", [])]
        if spec["expected_model"] not in ids:
            raise RuntimeError(f"{name}: expected {spec['expected_model']!r}, got {ids!r}")
        models[name] = ids
        completion(spec)  # wake/warm each on its own assigned GPU

    processes = physical_gpu_processes()
    if not processes[0] or not processes[3]:
        raise RuntimeError(f"expected active compute processes on physical GPU 0 and 3: {processes}")

    serial: dict[str, list[dict[str, Any]]] = {name: [] for name in SERVICES}
    for _ in range(3):
        for name, spec in SERVICES.items():
            serial[name].append(completion(spec))

    concurrent_runs: list[dict[str, Any]] = []
    for _ in range(3):
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {name: pool.submit(completion, spec) for name, spec in SERVICES.items()}
            per_model = {name: future.result() for name, future in futures.items()}
        wall = time.perf_counter() - started
        tokens = sum(row["completion_tokens"] for row in per_model.values())
        concurrent_runs.append({
            "wall_seconds": wall, "completion_tokens": tokens,
            "aggregate_wall_tokens_per_second": tokens / wall if wall else None,
            "per_model": per_model,
        })

    serial_sum = sum(mean(rows, "wall_tokens_per_second") or 0 for rows in serial.values())
    concurrent_tps = mean(concurrent_runs, "aggregate_wall_tokens_per_second")
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "independent dual-model parallel serving capacity; not an ensemble quality score",
        "physical_gpu_contract": {
            "a4000_qwen36": 0, "ada_qwen38": 3,
            "excluded_from_this_measurement": [1, 2],
            "cuda_device_order": "PCI_BUS_ID where configured by service",
        },
        "services": SERVICES,
        "service_exec_start": {
            name: service_exec_start(spec["service"]) for name, spec in SERVICES.items()
        },
        "autofit_contract": (
            "--fit on without an explicit --gpu-layers override, so sleep-idle reload can "
            "adapt to current free VRAM instead of aborting auto-fit"
        ),
        "visible_models": models,
        "compute_processes_by_physical_gpu": processes,
        "repetitions": 3,
        "max_tokens_per_request": 256,
        "serial": serial,
        "serial_mean": {
            name: {field: mean(rows, field) for field in
                   ("wall_tokens_per_second", "server_tokens_per_second", "prompt_tokens_per_second")}
            for name, rows in serial.items()
        },
        "serial_capacity_sum_tokens_per_second": serial_sum,
        "concurrent": concurrent_runs,
        "concurrent_aggregate_mean_tokens_per_second": concurrent_tps,
        "concurrent_efficiency_vs_serial_capacity_sum": (
            concurrent_tps / serial_sum if concurrent_tps is not None and serial_sum else None
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(OUT.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(OUT)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
