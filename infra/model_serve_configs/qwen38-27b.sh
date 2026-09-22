#!/usr/bin/env bash
# Serve Qwen3.8-27B (llama.cpp GGUF) on the two V100s.
# Rank #3 by composite score (0.851) -- this is the ORIGINAL reference model
# for the whole benchmark sweep. Still resident on disk as of 2026-09-18
# (production baseline, not deleted by the disk-management rule).
#
# Model: Qwen/Qwen3.8-27B, 27B dense.
# Quant: unsloth/Qwen3.8-27B-GGUF, Q4_K_M.
set -euo pipefail

MODEL_FILE=${MODEL_FILE:-/media/knight2/EDS2/models/qwen3.8-27b-q4_k_m.gguf}
SERVER=/media/knight2/EDS2/tools/llama.cpp/build-v100/bin/llama-server
PORT=${PORT:-18011}

download() {
  hf download unsloth/Qwen3.8-27B-GGUF \
    --include "*Q4_K_M*" --local-dir "$(dirname "$MODEL_FILE")"
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
