#!/usr/bin/env bash
# Start a verified SM70-compatible model on the two NVLinked V100s.
set -euo pipefail

MODEL_ID=${1:?"Pass a preflight-approved model id or local model directory"}
ROOT=/media/knight2/EDS2
VENV="$ROOT/envs/vllm-v100-cu124"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2
export HF_HOME="$ROOT/models/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$ROOT/cache/xdg"
export VLLM_CACHE_ROOT="$ROOT/cache/vllm"
export TMPDIR="$ROOT/tmp/vllm"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" "$ROOT/logs/vllm"

"$VENV/bin/python" "$(dirname "$0")/preflight.py" --model "$MODEL_ID" --tensor-parallel-size 2

# Both V100s are attached to NUMA node 1.  Interleaving lets vLLM use available
# DDR4 capacity on both nodes instead of failing because node 1 is currently
# pressured; it does not reserve RAM merely to inflate a utilization metric.
exec numactl --interleave=all "$VENV/bin/vllm" serve "$MODEL_ID" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 8192 \
  --host 127.0.0.1 --port 8011 \
  2>&1 | tee -a "$ROOT/logs/vllm/server.log"
