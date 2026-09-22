#!/usr/bin/env bash
# Serve the pinned Qwen3.8 NVFP4 target on the two V100s with 1Cat-vLLM 1.5.
#
# 2026-09-18: the low composite score first attributed to a missing
# --generation-config vllm (server silently falling back to the checkpoint's
# generation_config.json temperature=1.0/top_k=20/top_p=0.95) turned out NOT
# to be the real cause. Diagnostic curls to a HumanEval-style prompt at
# temperature=0 non-deterministically reproduce a repetition-loop failure
# ("The user is asking me to implement..." x3, finish_reason=stop) both WITH
# --generation-config vllm and independently of --kv-cache-dtype (tested
# fp8_e5m2 and auto/fp16, same failure both ways) -- ruling out KV-cache
# precision as the cause too. This points to a genuine TP=2 non-determinism
# / NVFP4-quantization-precision instability in this experimental SM70
# backend for this specific checkpoint, not a single wrong flag. Keep
# --generation-config vllm anyway (objectively more correct default
# regardless), but do not expect it to fix the quality gap -- see
# reports/well_known_suite_20260917.json's qwen38-1cat-vllm-target row and
# the corresponding note field for the full writeup.
set -euo pipefail

MODE=${1:-target}
PORT=${PORT:-18012}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e5m2}
ROOT=/media/knight2/EDS2
ENV_DIR="$ROOT/envs/1cat-vllm-1.5.0"
TARGET="$ROOT/models/1cat-vllm/Qwen3.8-27B-QUASAR-NVFP4"
DRAFT="$ROOT/models/1cat-vllm/Qwen3.8-27B-DFlash2"

case "$MODE" in
  target) DEFAULT_MODEL_NAME=qwen38-1cat-target ;;
  dflash2) DEFAULT_MODEL_NAME=qwen38-1cat-dflash2 ;;
  *) echo "usage: $0 [target|dflash2]" >&2; exit 2 ;;
esac
MODEL_NAME=${MODEL_NAME:-$DEFAULT_MODEL_NAME}

test -x "$ENV_DIR/bin/vllm"
test -f "$TARGET/model.safetensors.index.json"
test -f "$DRAFT/model.safetensors"

# One shared lock prevents DCC/AutoML and an LLM capacity run from contending
# for the same V100s. It remains held after exec because fd 9 is inherited.
exec 9>/tmp/v100_exclusive.lock
if ! flock -n 9; then
  echo "V100 capacity is reserved by another DCC/LLM job" >&2
  exit 75
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2
# The distro /usr/bin/nvcc is CUDA 12.0 and cannot compile TileLang's modern
# BF16 helpers. The installed CUDA 12.9 toolkit matches the wheel's CUDA-12.8
# ABI lane and the nvidia-cuda-nvcc-cu12 dependency installed with 1Cat.
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
  --tensor-parallel-size 2
  --attention-backend FLASH_ATTN_V100
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization 0.80
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

if [[ "$MODE" == dflash2 ]]; then
  ARGS+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"probabilistic\"}")
fi

exec numactl --interleave=all "$ENV_DIR/bin/vllm" "${ARGS[@]}"
