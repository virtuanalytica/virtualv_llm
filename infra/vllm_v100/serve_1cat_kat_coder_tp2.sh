#!/usr/bin/env bash
# Serve KAT-Coder-V2.5-Dev (community NVFP4 conversion) on the two V100s with
# 1Cat-vLLM 1.5, modeled directly on serve_1cat_qwen38_tp2.sh.
#
# 2026-09-18 RESULT: FAILED, do not retry with this checkpoint. The target
# below (doth4580/Kwaipilot-KAT-Coder-V2.5-Dev-NVFP4-MIXED) OOMs on load --
# despite compressed-tensors metadata in its config.json, vLLM's FusedMoE
# layer falls back to the UNQUANTIZED weight-creation path for the MoE
# experts (confirmed in the traceback: unquantized_fused_moe_method.py),
# allocating full fp16 expert weights instead of the compressed ones. Root
# cause: this conversion (and the only other available one,
# sakamakismile/KAT-Coder-V2.5-Dev-NVFP4) is tagged/built for NVIDIA
# GB10/Blackwell hardware, not V100/SM70 -- 1Cat-vLLM's SM70 kernel path is
# hand-tuned per architecture (see the explicit merged PR #270 for
# nvidia/Qwen3.6-35B-A3B-NVFP4), and neither KAT-Coder NVFP4 conversion has
# that. See dual_v100_candidate_research_20260918.json for the full writeup.
# Kept for reference / in case a real SM70-targeted NVFP4 conversion of this
# model ever appears -- do not re-attempt with either of the two checkpoints
# named above.
#
# Original rationale (still valid background, just not achievable with
# currently-available checkpoints): user asked for additional 1Cat-vLLM/
# other-engine candidates that could reach the top of the composite ranking
# while staying at or above 25% of the best-composite model's tok/s
# (kat-coder-v2.5-dev GGUF is #1 at 0.883 composite / 95.7 tok/s -- floor
# ~23.9 tok/s). Treat any future quality number here as unverified until a
# full well_known_suite run completes -- qwen38's own NVFP4 conversion
# looked fine on a single sample and still turned out unreliable across the
# full suite, so don't declare
# victory on a quick probe alone.
set -euo pipefail

MODE=${1:-target}
PORT=${PORT:-18013}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e5m2}
ROOT=/media/knight2/EDS2
ENV_DIR="$ROOT/envs/1cat-vllm-1.5.0"
TARGET="$ROOT/models/1cat-vllm/KAT-Coder-V2.5-Dev-NVFP4"
DRAFT="$ROOT/models/1cat-vllm/KAT-Coder-V2.5-Dev-DFlash2"

case "$MODE" in
  target) DEFAULT_MODEL_NAME=kat-coder-1cat-target ;;
  dflash2) DEFAULT_MODEL_NAME=kat-coder-1cat-dflash2 ;;
  *) echo "usage: $0 [target|dflash2]" >&2; exit 2 ;;
esac
MODEL_NAME=${MODEL_NAME:-$DEFAULT_MODEL_NAME}

test -x "$ENV_DIR/bin/vllm"
test -f "$TARGET/config.json"
if [[ "$MODE" == dflash2 ]]; then
  test -f "$DRAFT/config.json"
fi

# Same shared lock as serve_1cat_qwen38_tp2.sh -- never wrap this script in
# an outer `flock -n /tmp/v100_exclusive.lock ...`, it manages the lock
# itself via fd 9 (self-deadlocks otherwise, hit this exact bug once already
# this session).
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
