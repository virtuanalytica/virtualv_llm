#!/usr/bin/env bash
# Serve Granite-4.2-30B (llama.cpp GGUF) on the two V100s.
# Rank #2 by composite score (0.858), strongest bbh (0.854) of all models
# tested in the 2026-09-18 well_known_suite sweep.
#
# Model: ibm-granite/granite-4.2-30b, 30B dense.
# Quant: bartowski/granite-4.2-30b-GGUF, Q4_K_M, ~18GB.
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/media/knight2/EDS2/models/granite-4.2-30b}
MODEL_FILE="$MODEL_DIR/granite-4.2-30b-Q4_K_M.gguf"
SERVER=/media/knight2/EDS2/tools/llama.cpp/build-v100/bin/llama-server
PORT=${PORT:-18011}

download() {
  mkdir -p "$MODEL_DIR"
  hf download bartowski/granite-4.2-30b-GGUF \
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
