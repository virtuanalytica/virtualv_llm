#!/usr/bin/env bash
# Serve Qwen3.6-35B-A3B (1Cat-vLLM NVFP4) on the two V100s.
# Rank #4 by composite score (0.841), 114.8 tok/s (3.4x the Qwen3.8-27B GGUF
# reference) in the 2026-09-18 well_known_suite sweep.
#
# Model: nvidia/Qwen3.6-35B-A3B-NVFP4, 35B total / ~3B active MoE, native
# ModelOpt NVFP4 checkpoint (mixed FP8-dense + W4A16-NVFP4 routed/shared
# experts). Requires 1Cat-vLLM PR #270 (merged 2026-08-23) for SM70/V100
# support -- confirm that PR is present in the installed 1Cat-vLLM checkout
# before serving (`git log --oneline | grep -i "sm70\|v100"` in the 1Cat-vLLM
# source, or check `vllm --version` / release notes if installed from a wheel
# built after 2026-08-23).
# Size: ~23.5GB.
#
# Modeled on infra/vllm_v100/serve_1cat_qwen38_tp2.sh (same ENV_DIR / lock
# discipline / --generation-config vllm fix), model path swapped.
set -euo pipefail

MODE=${1:-target}
PORT=${PORT:-18012}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
ROOT=/media/knight2/EDS2
ENV_DIR="$ROOT/envs/1cat-vllm-1.5.0"
TARGET="$ROOT/models/1cat-vllm/Qwen3.6-35B-A3B-NVFP4"
MODEL_NAME=${MODEL_NAME:-qwen36-35b-a3b-1cat}

download() {
  mkdir -p "$TARGET"
  hf download nvidia/Qwen3.6-35B-A3B-NVFP4 --local-dir "$TARGET"
}

if [[ "$MODE" == download ]]; then
  download
  exit 0
fi

test -x "$ENV_DIR/bin/vllm"
test -f "$TARGET/model.safetensors.index.json" || { echo "model not found under $TARGET -- run: $0 download" >&2; exit 1; }

# Same shared lock as serve_1cat_qwen38_tp2.sh -- never wrap this script in an
# outer `flock -n /tmp/v100_exclusive.lock ...`, it manages the lock itself
# via this inherited fd 9 and a second flock on the same inode self-deadlocks.
exec 9>/tmp/v100_exclusive.lock
if ! flock -n 9; then
  echo "V100 capacity is reserved by another DCC/LLM job" >&2
  exit 75
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$ENV_DIR/bin:$PATH"
export HF_HOME="$ROOT/cache/huggingface-1cat"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$ROOT/cache/xdg-1cat"
export VLLM_CACHE_ROOT="$ROOT/cache/vllm-1cat"
export TMPDIR="$ROOT/tmp/vllm-1cat"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" "$ROOT/logs/vllm-1cat"

exec numactl --interleave=all "$ENV_DIR/bin/vllm" serve "$TARGET" \
  --served-model-name "$MODEL_NAME" \
  --trust-remote-code --dtype half --tensor-parallel-size 2 \
  --attention-backend FLASH_ATTN_V100 --kv-cache-dtype fp8_e5m2 \
  --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.80 \
  --max-num-batched-tokens 4096 --max-num-seqs "$MAX_NUM_SEQS" \
  --enable-prefix-caching --mamba-cache-mode align \
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking":false}' \
  --generation-config vllm \
  --host 127.0.0.1 --port "$PORT"
