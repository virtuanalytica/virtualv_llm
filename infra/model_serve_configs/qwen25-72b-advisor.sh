#!/usr/bin/env bash
# Serve Qwen2.5-72B-Instruct (llama.cpp GGUF) as a dedicated "advisor" role on
# the V100 NVLink pair -- 2026-09-19 user decision: a heavier/slower
# consultation model, separate from whichever fast model (kat-coder-v2.5-dev,
# the mixture-of-models ensemble, etc.) handles routine virtualpc agent
# traffic on the shared port 18011 slot. Runs on its OWN port (18016) so it
# stays persistently available regardless of what's active on 18011.
#
# Composite 0.753 (gsm8k 0.72, humaneval 0.95, mmlu 0.863 -- the best mmlu of
# any tested model, fitting for a "harder decisions" role, bbh 0.479),
# 14.9 tok/s on the dual-V100 profile. Best already-validated 70B-class
# candidate: llama31-70b-instruct-q4 scores lower (0.598) but that's dragged
# down by a known humaneval=0 bug, never re-verified -- don't substitute it
# here without re-testing.
# Model: Qwen/Qwen2.5-72B-Instruct, dense 72B.
# Quant: Q4_K_M, 12 shards, ~44GB total.
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/media/knight2/EDS2/models/llm/qwen25-72b-q4km}
MODEL_FILE="$MODEL_DIR/qwen2.5-72b-instruct-q4_k_m-00001-of-00012.gguf"
SERVER=/media/knight2/EDS2/tools/llama.cpp/build-v100/bin/llama-server
PORT=${PORT:-18016}

download() {
  mkdir -p "$MODEL_DIR"
  hf download Qwen/Qwen2.5-72B-Instruct-GGUF \
    --include "*q4_k_m*" --local-dir "$MODEL_DIR"
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
  --model "$MODEL_FILE" --alias advisor --host 127.0.0.1 --port "$PORT" \
  --ctx-size 8192 --parallel 1 --gpu-layers 99 --split-mode layer --main-gpu 0 \
  --flash-attn on --reasoning off --cache-type-k q8_0 --cache-type-v q8_0 --jinja \
  --tensor-split 1,1
