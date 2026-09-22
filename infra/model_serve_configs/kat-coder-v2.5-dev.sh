#!/usr/bin/env bash
# Serve KAT-Coder-V2.5-Dev (llama.cpp GGUF) on the two V100s.
# Rank #1 by composite score (0.883) in the 2026-09-18 well_known_suite sweep:
# gsm8k 0.92, humaneval pass@1 0.95, mmlu_sample 0.86, bbh 0.833 (approx from
# composite; see reports/well_known_suite_20260917.json for exact per-task values).
#
# Model: Kwaipilot/KAT-Coder-V2.5-Dev, 35B total / ~3B active MoE (arch
# qwen3_5_moe -- loads on llama.cpp b10917+ despite no matching `strings` hit
# in libllama-common.so; don't trust absence-of-string-match as proof of
# non-support, just try loading it).
# Quant: bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF, Q4_K_M, ~21.4GB.
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/media/knight2/EDS2/models/kat-coder-v2.5-dev}
MODEL_FILE="$MODEL_DIR/Kwaipilot_KAT-Coder-V2.5-Dev-Q4_K_M.gguf"
SERVER=/media/knight2/EDS2/tools/llama.cpp/build-v100/bin/llama-server
PORT=${PORT:-18011}

download() {
  mkdir -p "$MODEL_DIR"
  hf download bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF \
    --include "*Q4_K_M*" --local-dir "$MODEL_DIR"
}

if [[ "${1:-serve}" == download ]]; then
  download
  exit 0
fi

test -f "$MODEL_FILE" || { echo "model not found at $MODEL_FILE -- run: $0 download" >&2; exit 1; }

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2
export GGML_CUDA_P2P=1

exec "$SERVER" \
  --model "$MODEL_FILE" --alias x --host 127.0.0.1 --port "$PORT" \
  --ctx-size 8192 --parallel 1 --gpu-layers 99 --split-mode layer --main-gpu 0 \
  --flash-attn on --reasoning off --cache-type-k q8_0 --cache-type-v q8_0 --jinja \
  --tensor-split 1,1
