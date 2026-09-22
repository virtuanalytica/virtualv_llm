#!/usr/bin/env bash
# Serve NVIDIA-Nemotron-3.5-Lightning-30B-A3B (llama.cpp GGUF) on the two V100s.
# Rank #5 by composite score (~0.816), fastest GGUF model tested (127.8 tok/s,
# 3.8x the Qwen3.8-27B reference) in the 2026-09-18 well_known_suite sweep.
# Weaker on gsm8k (0.78) than the top-4.
#
# Model: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B, 30B total / ~3B active,
# hybrid Mamba2+MoE+Attention (arch nemotron_h_moe -- confirmed present in
# libllama-common.so).
# Quant: UD-Q4_K_XL (unsloth or bartowski), ~25.5GB.
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/media/knight2/EDS2/models/nemotron35-lightning-30b-a3b}
MODEL_FILE="$MODEL_DIR/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_XL.gguf"
SERVER=/media/knight2/EDS2/tools/llama.cpp/build-v100/bin/llama-server
PORT=${PORT:-18011}

download() {
  mkdir -p "$MODEL_DIR"
  hf download unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF \
    --include "*UD-Q4_K_XL*" --local-dir "$MODEL_DIR"
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
