#!/usr/bin/env python3
"""Benchmark public GLM-5.3-Flash AJ-IQ2_XXS across every local GPU topology.

The 753B GLM-5.3 release cannot fit on the available filesystem.  This matrix
therefore uses the 313.33B inference trunk of GLM-5.3-Flash (the unused
training-only MTP/NextN block is pruned), with CPU offload chosen independently
for every hardware profile.
Live local-model services are restored even when a profile fails.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/benchmarks/benchmark_local_gguf_tp2.py"
RENDER = ROOT / "scripts/benchmarks/build_dual_v100_html.py"
OUT = ROOT / "reports/hardware_scaling_glm53_20260921.json"
MODEL_ROOT = Path("/media/knight2/EDS2/models/llm/glm53-flash-aj-iq2xxs")
MODEL = MODEL_ROOT / "AJ-IQ2_XXS/GLM-5.3-Flash-AJ-IQ2_XXS-00001-of-00002.gguf"
GLM_LLAMA_ROOT = Path("/media/knight2/EDS2/tools/llama.cpp-glm53-pr27752")
MODEL_KEY = "glm53-flash-aj-iq2xxs"
SHARDS = {
    "AJ-IQ2_XXS/GLM-5.3-Flash-AJ-IQ2_XXS-00001-of-00002.gguf": {
        "bytes": 44_665_098_240,
        "sha256": "1fa0535ecddaee4ff127eb8ffd47d3855ca7efa1cc29c996c24da699c5fbe3f7",
    },
    "AJ-IQ2_XXS/GLM-5.3-Flash-AJ-IQ2_XXS-00002-of-00002.gguf": {
        "bytes": 42_680_908_320,
        "sha256": "ab8abbde061254a7f6b1a16b3627c4b77e3084f52e3d9739ebaa515b9365945c",
    },
}
EXPECTED_BYTES = sum(item["bytes"] for item in SHARDS.values())
SERVICES = {
    "llama-qwen.service": "http://127.0.0.1:11435/health",
    "local-chat-qwen38.service": "http://127.0.0.1:8011/health",
}
PROFILES = [
    # Highest-information configurations first so an interrupted long run still
    # answers the user's all-hardware and standalone-V100 questions.
    "all-four-layer",
    "single-v100",
    "single-v100-2",
    "dual-layer",
    "single-ada",
    "single-a4000",
    "dual-a4000ada",
]
PROVENANCE = {
    "requested_family": "GLM-5.3",
    "tested_model": "GLM-5.3-Flash",
    "parameter_count": "313.33B stored inference trunk / 17.3B active",
    "quantization": "AJ-IQ2_XXS hand-mixed GGUF, 2.23 bpw",
    "source_repo": "aj9o9/GLM-5.3-Flash-GGUF",
    "source_url": "https://huggingface.co/aj9o9/GLM-5.3-Flash-GGUF",
    "source_revision": "07c62fcdeaf1c05d22bd123c3da8058a1b1e63e2",
    "source_last_modified": "2026-09-01T20:10:54Z",
    "weight_bytes": EXPECTED_BYTES,
    "weight_shards": SHARDS,
    "runtime_pr": "https://github.com/ggml-org/llama.cpp/pull/27752",
    "runtime_commit": "1d0c76f3c6d030fdfc269aa27db6334ea2834cec",
    "official_base": "https://huggingface.co/zai-org/GLM-5.3-Flash",
    "selection_reason": (
        "The official 753B GLM-5.3 and higher-bit Flash artifacts do not fit the disk. "
        "This public 87.35 GB quant fits, documents its measured quality, and targets "
        "the pinned llama.cpp GLM-5.3-Flash implementation."
    ),
}


def active(service: str) -> bool:
    return subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", service], check=False,
    ).returncode == 0


def wait_healthy(url: str, seconds: int = 240) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"service did not become healthy: {url}")


def enrich_report() -> None:
    if not OUT.exists():
        return
    payload = json.loads(OUT.read_text())
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    payload["matrix_model"] = MODEL_KEY
    payload["provenance"] = PROVENANCE
    payload["profile_policy"] = (
        "Each row exposes only its named physical GPU(s); single-V100 rows expose "
        "GPU1 and GPU2 independently. Layer split is used for heterogeneous and "
        "NVLink combinations; unsafe long-running V100 tensor split is excluded."
    )
    temporary = OUT.with_suffix(OUT.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(OUT)


def prepare_report() -> None:
    """Drop incompatible-artifact attempts while preserving resumable AJ rows."""
    if not OUT.exists():
        return
    payload = json.loads(OUT.read_text())
    payload["results"] = [
        row for row in payload.get("results", []) if row.get("model") == MODEL_KEY
    ]
    temporary = OUT.with_suffix(OUT.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(OUT)


def main() -> int:
    missing = [
        str(MODEL_ROOT / relative) for relative, expected in SHARDS.items()
        if not (MODEL_ROOT / relative).exists()
        or (MODEL_ROOT / relative).stat().st_size != expected["bytes"]
    ]
    if missing:
        raise SystemExit("missing or incomplete GLM-5.3 shard(s): " + ", ".join(missing))
    prepare_report()
    initially_active = {service: active(service) for service in SERVICES}
    for service, was_active in initially_active.items():
        if was_active:
            subprocess.run(["systemctl", "--user", "stop", service], check=True)
    failed = False
    try:
        command = [
            "flock", "-n", "/tmp/v100_exclusive.lock", "python3", str(RUNNER),
            MODEL_KEY, "--profiles", *PROFILES,
            "--out", str(OUT), "--auto-fit", "--ready-timeout", "1200",
            "--request-timeout", "1800", "--resume",
        ]
        env = os.environ.copy()
        env["LLAMA_SERVER"] = str(GLM_LLAMA_ROOT / "build-v100/bin/llama-server")
        env["LLAMA_BENCH"] = str(GLM_LLAMA_ROOT / "build-v100/bin/llama-bench")
        run = subprocess.run(command, cwd=ROOT, check=False, env=env)
        enrich_report()
        subprocess.run(["python3", str(RENDER)], cwd=ROOT, check=True)
        if run.returncode:
            raise subprocess.CalledProcessError(run.returncode, command)
    except Exception:
        failed = True
        enrich_report()
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
