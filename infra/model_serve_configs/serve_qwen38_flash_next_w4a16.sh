#!/usr/bin/env bash
# Serve one local Qwen3.8 Flash-Next W4A16 checkpoint through the audited
# 1Cat-vLLM build.  The caller owns lifecycle and benchmark result handling.
set -euo pipefail

DEVICES=${1:?usage: $0 <physical-gpus> <tp> <model-dir> [port] [cpu-offload-gb]}
TP=${2:?usage: $0 <physical-gpus> <tp> <model-dir> [port] [cpu-offload-gb]}
MODEL_DIR=${3:?usage: $0 <physical-gpus> <tp> <model-dir> [port] [cpu-offload-gb]}
PORT=${4:-18021}
CPU_OFFLOAD_GB=${5:-112}
ROOT=/media/knight2/EDS2
ENV_DIR="$ROOT/envs/1cat-vllm-1.5.0"

test -x "$ENV_DIR/bin/vllm"
test -f "$MODEL_DIR/config.json"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$DEVICES"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME="$ROOT/cache/huggingface-1cat"
export XDG_CACHE_HOME="$ROOT/cache/xdg-1cat"
export VLLM_CACHE_ROOT="$ROOT/cache/vllm-1cat"
export TMPDIR="$ROOT/tmp/vllm-1cat"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" "$ROOT/logs/vllm-1cat"

# The V100 pair has 64 GiB VRAM; the PLE table and remaining layers use the
# explicitly reported host-offload budget.  Keep a short context for a fair
# decode benchmark rather than claiming the vendor's long-context capacity.
exec numactl --interleave=all "$ENV_DIR/bin/vllm" serve "$MODEL_DIR" \
  --served-model-name qwen38-flash-next \
  --trust-remote-code --language-model-only --dtype half --tensor-parallel-size "$TP" \
  --cpu-offload-gb "$CPU_OFFLOAD_GB" --gpu-memory-utilization 0.88 \
  --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 4096 \
  --generation-config vllm --host 127.0.0.1 --port "$PORT"
