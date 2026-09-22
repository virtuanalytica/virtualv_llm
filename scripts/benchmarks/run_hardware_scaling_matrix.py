#!/usr/bin/env python3
"""Run an apples-to-apples Qwen3.6 matrix and always restore live services."""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/benchmarks/benchmark_local_gguf_tp2.py"
RENDER = ROOT / "scripts/benchmarks/build_dual_v100_html.py"
OUT = ROOT / "reports/hardware_scaling_qwen36_20260921.json"
OLD_TENSOR = ROOT / "reports/local_gguf_dual_v100_tensor_20260917.json"
SERVICES = {
    "llama-qwen.service": "http://127.0.0.1:11435/health",
    "local-chat-qwen38.service": "http://127.0.0.1:8011/health",
}
PROFILES = ["single-a4000", "single-ada", "dual-a4000ada", "single-v100", "dual-layer"]


def active(service: str) -> bool:
    return subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", service], check=False,
    ).returncode == 0


def wait_healthy(url: str, seconds: int = 180) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"service did not become healthy: {url}")


def merge_existing_tensor_result() -> None:
    if not OUT.exists() or not OLD_TENSOR.exists():
        return
    payload = json.loads(OUT.read_text())
    old = json.loads(OLD_TENSOR.read_text())
    tensor = next((row for row in old.get("results", [])
                   if row.get("model") == "qwen36-27b-iq3"
                   and row.get("profile") == "dual-tensor" and not row.get("error")), None)
    if tensor is None:
        return
    tensor = dict(tensor)
    tensor["reused_measurement"] = True
    tensor["reused_from"] = str(OLD_TENSOR.relative_to(ROOT))
    tensor["reuse_reason"] = (
        "Tensor split twice caused a driver-level V100 hang in later long suites; "
        "reuse the already completed same-model measurement instead of risking a third hang."
    )
    payload["results"] = [row for row in payload.get("results", [])
                          if row.get("profile") != "dual-tensor"] + [tensor]
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    payload["matrix_model"] = "qwen36-27b-iq3"
    payload["tensor_policy"] = "validated prior measurement reused; no unsafe rerun"
    temporary = OUT.with_suffix(OUT.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(OUT)


def main() -> int:
    initially_active = {service: active(service) for service in SERVICES}
    for service, was_active in initially_active.items():
        if was_active:
            subprocess.run(["systemctl", "--user", "stop", service], check=True)
    failed = False
    try:
        command = [
            "flock", "-n", "/tmp/v100_exclusive.lock", "python3", str(RUNNER),
            "qwen36-27b-iq3", "--profiles", *PROFILES, "--out", str(OUT),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        merge_existing_tensor_result()
        subprocess.run(["python3", str(RENDER)], cwd=ROOT, check=True)
    except Exception:
        failed = True
        raise
    finally:
        restore_errors = []
        for service, was_active in initially_active.items():
            if not was_active:
                continue
            try:
                subprocess.run(["systemctl", "--user", "start", service], check=True)
                wait_healthy(SERVICES[service])
            except Exception as exc:
                restore_errors.append(f"{service}: {exc}")
        if restore_errors and not failed:
            raise RuntimeError("; ".join(restore_errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
