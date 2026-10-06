#!/usr/bin/env python3
"""Read-only Kimi K2.5 checkpoint and engine capability probe.

Registration is deliberately distinct from serving readiness. This script
never loads the 595 GB model or reserves GPUs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

DEFAULT_CHECKPOINT = Path("/media/knight2/eds1/Kimi-K2.5-Tower")
DEFAULT_COLIBRI = Path("/media/knight2/EDS2/tools/colibri")
DEFAULT_1CAT = Path("/media/knight2/EDS2/envs/1cat-vllm-1.5.0")


def checkpoint_report(path: Path) -> dict:
    config = json.loads((path / "config.json").read_text())
    index = json.loads((path / "model.safetensors.index.json").read_text())
    text = config.get("text_config", {})
    shards = sorted(set(index["weight_map"].values()))
    missing = [name for name in shards if not (path / name).is_file()]
    quant = text.get("quantization_config", {})
    weight = quant.get("config_groups", {}).get("group_0", {}).get("weights", {})
    valid = (config.get("model_type") == "kimi_k25" and
             text.get("model_type") == "kimi_k2" and
             text.get("n_routed_experts") == 384 and
             text.get("num_hidden_layers") == 61 and
             quant.get("quant_method") == "compressed-tensors" and
             weight.get("num_bits") == 4 and not missing)
    return {"valid": valid, "root_type": config.get("model_type"),
            "text_type": text.get("model_type"), "moe_layers": text.get("num_hidden_layers", 0) - 1,
            "experts_per_layer": text.get("n_routed_experts"),
            "quant": {"method": quant.get("quant_method"), "bits": weight.get("num_bits"),
                      "group_size": weight.get("group_size")},
            "shards": len(shards), "missing_shards": missing,
            "indexed_bytes": index.get("metadata", {}).get("total_size")}


def colibri_report(install: Path, model: Path) -> dict:
    cli = install / "coli"
    if not cli.is_file():
        return {"installed": False, "registered": False, "gpu_binary": False}
    info = subprocess.run([str(cli), "info", "--model", str(model)], text=True,
                          capture_output=True, timeout=30, check=False)
    binary = install / "colibri"
    linked = subprocess.run(["ldd", str(binary)], text=True, capture_output=True,
                            timeout=5, check=False) if binary.exists() else None
    gpu = bool(linked and ("libcudart" in linked.stdout or "libamdhip64" in linked.stdout))
    return {"installed": True, "registered": info.returncode == 0,
            "gpu_binary": gpu, "info_exit": info.returncode,
            "info_message": (info.stderr or info.stdout).strip()[-350:],
            "serve_ready": False}


def onecat_report(env: Path) -> dict:
    library = env / "lib" / "python3.12" / "site-packages" / "vllm"
    model = library / "model_executor" / "models" / "kimi_k25.py"
    registry = library / "model_executor" / "models" / "registry.py"
    registered = (model.is_file() and registry.is_file() and
                  "KimiK25ForConditionalGeneration" in registry.read_text())
    quant_root = library / "model_executor" / "layers" / "quantization"
    fused_root = library / "model_executor" / "layers" / "fused_moe"
    turbomind = quant_root / "sm70_turbomind.py"
    triton = fused_root / "experts" / "triton_moe.py"
    fused_layer = fused_root / "layer.py"
    return {"installed": library.is_dir(), "model_registered": registered,
            "sm70_turbomind_compressed_int4_present":
            turbomind.is_file() and "prepare_compressed_uint4_linear" in turbomind.read_text(),
            "triton_wna16_moe_present":
            triton.is_file() and "class TritonWNA16Experts" in triton.read_text(),
            "fused_moe_weight_loader_present":
            fused_layer.is_file() and "def weight_loader" in fused_layer.read_text(),
            "native_int4_cpu_offload_verified": False, "sm70_serve_verified": False,
            "fused_moe_tier_loader_verified": False,
            "serve_ready": False}


def report(checkpoint: Path, colibri: Path, onecat: Path) -> dict:
    return {"checkpoint": checkpoint_report(checkpoint),
            "colibri": colibri_report(colibri, checkpoint),
            "onecat_vllm": onecat_report(onecat),
            "note": "Model registration and weight presence are not a serving benchmark."}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--colibri", type=Path, default=DEFAULT_COLIBRI)
    ap.add_argument("--onecat-env", type=Path, default=DEFAULT_1CAT)
    args = ap.parse_args()
    print(json.dumps(report(args.checkpoint, args.colibri, args.onecat_env), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
