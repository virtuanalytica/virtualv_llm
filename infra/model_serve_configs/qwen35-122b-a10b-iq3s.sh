#!/usr/bin/env bash
# Serve the retained large-model winner on physical GPU1+2 only.
# Full v4 suite (2026-09-21): composite 0.89855, GSM8K .84, BBH .9167,
# MMLU .8625, HumanEval .975, TruthfulQA .40; measured decode 49.19 t/s.
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/media/knight2/EDS2/models/llm/qwen35-122b-a10b-iq3s}
MODEL_FILE="$MODEL_DIR/Qwen3.5-122B-A10B-UD-IQ3_S.gguf"
SERVER=/media/knight2/EDS2/tools/llama.cpp/build-v100/bin/llama-server
PORT=${PORT:-18011}
REVISION=51eab4d59d53f573fb9206cb3ce613f1d0aa392b

download() {
  mkdir -p "$MODEL_DIR"
  hf download unsloth/Qwen3.5-122B-A10B-GGUF \
    --revision "$REVISION" \
    --include Qwen3.5-122B-A10B-UD-IQ3_S.gguf \
    --local-dir "$MODEL_DIR"
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
  --model "$MODEL_FILE" --alias qwen35-122b-a10b --host 127.0.0.1 --port "$PORT" \
  --ctx-size 8192 --parallel 1 --gpu-layers 99 --split-mode layer --main-gpu 0 \
  --flash-attn on --reasoning off --cache-type-k q8_0 --cache-type-v q8_0 --jinja \
  --tensor-split 1,1
