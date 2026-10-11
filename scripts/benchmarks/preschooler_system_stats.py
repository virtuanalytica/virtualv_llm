#!/usr/bin/env python3
"""Save machine-readable system stats while llmwatch displays vLLM live.

llmwatch is an upstream terminal UI, not an export API. This sidecar keeps a
bounded terminal capture and independently records the same source counters.
GPU power here is a sampled board reading, never total system energy.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pty
import select
import signal
import struct
import subprocess
import termios
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import psutil

DEFAULT_LLMWATCH = Path("/media/knight2/claude-data/knight1/third_party/llmwatch/target/release/llmwatch")
STOP = False


def stop(_signal, _frame) -> None:
    global STOP
    STOP = True


def sample(base: str) -> dict:
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,power.draw,temperature.gpu",
                                   "--format=csv,noheader,nounits"], text=True, timeout=10)
    gpu = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        gpu.append({"index": int(parts[0]), "vram_mib": float(parts[1]),
                    "utilization_pct": float(parts[2]), "board_watts": float(parts[3]),
                    "temperature_c": float(parts[4])})
    metrics = {}
    try:
        with urlopen(base + "/metrics", timeout=5) as response:
            for line in response.read().decode().splitlines():
                if line.startswith("#") or " " not in line:
                    continue
                name, value = line.rsplit(" ", 1)
                if any(token in name for token in ("generation_tokens_total", "prompt_tokens_total",
                                                  "kv_cache_usage", "gpu_cache_usage",
                                                  "time_to_first_token", "time_per_output_token",
                                                  "request_success_total", "num_requests_running")):
                    try:
                        metrics[name] = float(value)
                    except ValueError:
                        pass
    except OSError as exc:
        metrics["error"] = str(exc)
    mem = psutil.virtual_memory()
    disk = psutil.disk_io_counters(perdisk=True)
    return {"at": datetime.now(timezone.utc).isoformat(), "epoch": time.time(),
            "cpu_pct": psutil.cpu_percent(interval=None), "ram_used_bytes": mem.used,
            "ram_available_bytes": mem.available, "gpu": gpu,
            "disk_io": {name: {"read_bytes": row.read_bytes, "write_bytes": row.write_bytes}
                        for name, row in disk.items() if name.startswith(("nvme", "pmem"))},
            "vllm_metrics": metrics,
            "scope": "GPU-board readings plus host CPU/RAM/NVMe counters; no wall-power meter"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--llmwatch-bin", type=Path, default=DEFAULT_LLMWATCH)
    args = parser.parse_args()
    if args.interval < 0.5:
        parser.error("interval must be at least 0.5 seconds")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    master = slave = None
    proc = None
    capture = deque()
    capture_size = 0
    if args.llmwatch_bin.is_file():
        master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
        proc = subprocess.Popen([str(args.llmwatch_bin), "--url", args.base,
                                 "--ssh", "localhost", "--interval", str(args.interval)],
                                stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
        os.close(slave)
        slave = None
    next_sample = 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.out.open("w", buffering=1) as target:
            while not STOP:
                if time.monotonic() >= next_sample:
                    try:
                        target.write(json.dumps(sample(args.base)) + "\n")
                    except (OSError, subprocess.SubprocessError, ValueError) as exc:
                        target.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(),
                                                 "error": str(exc)}) + "\n")
                    next_sample = time.monotonic() + args.interval
                if master is not None:
                    ready, _, _ = select.select([master], [], [], 0.2)
                    if ready:
                        try:
                            data = os.read(master, 8192)
                        except OSError:
                            data = b""
                        if data:
                            capture.append(data)
                            capture_size += len(data)
                            while capture_size > 1_000_000:
                                capture_size -= len(capture.popleft())
                else:
                    time.sleep(0.2)
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        if master is not None:
            os.close(master)
        if slave is not None:
            os.close(slave)
        (args.out.parent / "llmwatch-last-screen.ansi").write_bytes(b"".join(capture))


if __name__ == "__main__":
    main()
