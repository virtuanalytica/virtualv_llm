#!/usr/bin/env bash
# Serve the pinned Qwen3.8-27B QUASAR-NVFP4 target across configurable GPU
# sets, to compare hardware for the same checkpoint (2026-09-19 user request:
# 2xV100 vs RTX4000Ada alone vs A4000 alone vs Ada+A4000 vs all 4 GPUs).
#
# The original serve_1cat_qwen38_tp2.sh hardcodes CUDA_VISIBLE_DEVICES=1,2 and
# --attention-backend FLASH_ATTN_V100 -- both V100/SM70-specific. This variant
# takes the device list and tensor-parallel size as parameters and omits the
# forced attention backend (letting vLLM auto-select per detected
# architecture) unless FORCE_ATTN_BACKEND is set, since FLASH_ATTN_V100 is a
# 1Cat SM70-only kernel and would be wrong (or refuse to start) on
# Ampere/Ada. Does NOT touch /tmp/v100_exclusive.lock when GPU 1/2 aren't
# both in the device list, so it doesn't falsely contend with real V100 jobs.
set -euo pipefail

DEVICES=${1:?usage: $0 <cuda_visible_devices e.g. 3 or 0,3 or 0,1,2,3> <tensor_parallel_size> [port]}
TP=${2:?usage: $0 <cuda_visible_devices> <tensor_parallel_size> [port]}
PORT=${3:-18012}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e5m2}
ROOT=/media/knight2/EDS2
ENV_DIR="$ROOT/envs/1cat-vllm-1.5.0"
TARGET="$ROOT/models/1cat-vllm/Qwen3.8-27B-QUASAR-NVFP4"
MODEL_NAME=${MODEL_NAME:-qwen38-1cat-target}

test -x "$ENV_DIR/bin/vllm"
test -f "$TARGET/model.safetensors.index.json"

# Only take the shared V100 lock when this run actually uses GPU 1 and/or 2 --
# a pure-Ada/A4000 run has no business contending for it.
if [[ ",$DEVICES," == *",1,"* || ",$DEVICES," == *",2,"* ]]; then
  exec 9>/tmp/v100_exclusive.lock
  if ! flock -n 9; then
    echo "V100 capacity is reserved by another DCC/LLM job" >&2
    exit 75
  fi
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$DEVICES"
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$ENV_DIR/bin:$PATH"
export HF_HOME="$ROOT/cache/huggingface-1cat"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$ROOT/cache/xdg-1cat"
export VLLM_CACHE_ROOT="$ROOT/cache/vllm-1cat"
export TMPDIR="$ROOT/tmp/vllm-1cat"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" "$ROOT/logs/vllm-1cat"

ARGS=(
  serve "$TARGET"
  --served-model-name "$MODEL_NAME"
  --trust-remote-code
  --dtype half
  --tensor-parallel-size "$TP"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization 0.85
  --max-num-batched-tokens 4096
  --max-num-seqs "$MAX_NUM_SEQS"
  --enable-prefix-caching
  --mamba-cache-mode align
  --reasoning-parser qwen3
  --default-chat-template-kwargs '{"enable_thinking":false}'
  --generation-config vllm
  --host 127.0.0.1
  --port "$PORT"
)

if [[ -n "${FORCE_ATTN_BACKEND:-}" ]]; then
  ARGS+=(--attention-backend "$FORCE_ATTN_BACKEND")
fi

exec numactl --interleave=all "$ENV_DIR/bin/vllm" "${ARGS[@]}"
