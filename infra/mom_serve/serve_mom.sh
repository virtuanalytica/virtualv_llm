#!/usr/bin/env bash
# Serve the default local mixture-of-models on this machine, behind the OpenAI-compatible
# endpoint that the omp "toddler" profile already points at (http://127.0.0.1:8030/v1, model mom-live).
#
#   serve_mom.sh up      start the default model if nothing answers on the port (idempotent)
#   serve_mom.sh down    stop it
#   serve_mom.sh status  show what answers on the port and which GPUs it uses
#
# `up` also starts the thinking router (scripts/benchmarks/mixture_proxy.py --thinking-router) on
# :8031 as model mom-router, in front of mom-live. It switches reasoning on only for calculation and
# code questions: measured on a fresh procedural pack 0.60 -> 0.98 at 28.6x the average latency
# (PR #65). MOM_ROUTER=0 skips it; mom-live on :8030 stays reachable directly.
#
# The default is chosen from measured rows (see docs/MOM_SERVE_DEFAULT.md). It is a single model,
# not an ensemble: on this box every measured ensemble either scored below its best member or
# dropped below the 50 tokens/s floor. Change MOM_DEFAULT to switch.
set -euo pipefail

PORT=${MOM_PORT:-8030}
ALIAS=mom-live
SERVER=/media/knight2/EDS2/tools/llama.cpp/build-v100-new/bin/llama-server
PIDFILE=${MOM_PIDFILE:-$HOME/.cache/mom-live.pid}
LOG=${MOM_LOG:-$HOME/.cache/mom-live.log}
MOM_DEFAULT=${MOM_DEFAULT:-qwen36}
VLLM_ENV=/media/knight2/EDS2/envs/1cat-vllm-1.5.0
# Weights: the eds1 copy (via the 1cat-vllm symlink) or, when eds1 is not mounted, the EDS2 copy.
VLLM_MODEL=${MOM_VLLM_MODEL:-/media/knight2/EDS2/models/1cat-vllm/Qwen3.6-35B-A3B-NVFP4}
[ -f "$VLLM_MODEL/config.json" ] || VLLM_MODEL=/media/knight2/EDS2/models/qwen36-35b-a3b-nvfp4
V100_LOCK=/tmp/v100_exclusive.lock
ROUTER_PORT=${MOM_ROUTER_PORT:-8031}
ROUTER_PIDFILE=${MOM_ROUTER_PIDFILE:-$HOME/.cache/mom-router.pid}
ROUTER_LOG=${MOM_ROUTER_LOG:-$HOME/.cache/mom-router.log}
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)

# vLLM answers /health with an empty 200, llama.cpp with {"status":"ok"}: check the status code only
answers() { [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")" = 200 ]; }
served() { curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" | grep -q "\"$ALIAS\""; }
router_up() { curl -s -m 3 "http://127.0.0.1:$ROUTER_PORT/v1/models" | grep -q '"mom-router"'; }

start_router() {
  [ "${MOM_ROUTER:-1}" = "0" ] && return 0
  router_up && { echo "mom-router already answers on :$ROUTER_PORT"; return 0; }
  nohup python3 "$REPO_ROOT/scripts/benchmarks/mixture_proxy.py" --port "$ROUTER_PORT" --name mom-router \
    --aggregator "mom=http://127.0.0.1:$PORT/v1#$ALIAS" --thinking-router > "$ROUTER_LOG" 2>&1 &
  echo $! > "$ROUTER_PIDFILE"
  for _ in $(seq 1 20); do router_up && { echo "mom-router up on :$ROUTER_PORT (thinking for calculation, code)"; return 0; }; sleep 0.5; done
  echo "mom-router did not start; see $ROUTER_LOG" >&2; return 1
}

# Default: Qwen3.6-35B-A3B NVFP4 over the NVLink V100 pair (TP2), vLLM 1Cat. One server for interactive use,
# the thinking router and the batch queue: TP1 on one V100 left the other V100 idle or split the work.
# Measured: TP1 103.7 t/s single stream (2026-10-10); TP2 111.0 t/s single stream, 288 t/s at 4 streams
# (2026-10-06 record harness), 8-task battery 0.854 in both layouts. MOM_TP=1 keeps the old one-V100 layout.
start_qwen36() {
  exec 9>"$V100_LOCK"
  flock -n 9 || { echo "V100 capacity is reserved by another job (lock $V100_LOCK)" >&2; exit 75; }
  local tp=${MOM_TP:-2}
  export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$([ "$tp" = 2 ] && echo 3,4 || echo 3) CUDA_HOME=/usr/local/cuda-12.9
  export PATH="$CUDA_HOME/bin:$VLLM_ENV/bin:$PATH" VLLM_WORKER_MULTIPROC_METHOD=spawn
  export HF_HOME=/media/knight2/EDS2/cache/huggingface-1cat HUGGINGFACE_HUB_CACHE=/media/knight2/EDS2/cache/huggingface-1cat/hub
  export XDG_CACHE_HOME=/media/knight2/EDS2/cache/xdg-1cat VLLM_CACHE_ROOT=/media/knight2/EDS2/cache/vllm-1cat TMPDIR=/media/knight2/EDS2/tmp/vllm-1cat
  mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR"
  exec numactl --interleave=all "$VLLM_ENV/bin/vllm" serve "$VLLM_MODEL" \
    --served-model-name "$ALIAS" --trust-remote-code --dtype half --tensor-parallel-size "$tp" \
    --attention-backend FLASH_ATTN_V100 --kv-cache-dtype fp8_e5m2 --max-model-len 32768 \
    --gpu-memory-utilization 0.80 --max-num-batched-tokens 4096 --max-num-seqs "$([ "$tp" = 2 ] && echo 4 || echo 1)" \
    --enable-prefix-caching --mamba-cache-mode align --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking":false}' --generation-config vllm \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --host 127.0.0.1 --port "$PORT"
}

case "${1:-status}" in
  up)
    if answers && served; then echo "mom-live already answers on :$PORT"; start_router; exit 0; fi
    mkdir -p "$(dirname "$PIDFILE")"
    case "$MOM_DEFAULT" in
      qwen36)
        ( start_qwen36 ) > "$LOG" 2>&1 &
        echo $! > "$PIDFILE"
        for _ in $(seq 1 240); do served && { echo "mom-live up on :$PORT (qwen36 NVFP4, TP${MOM_TP:-2})"; start_router; exit 0; }; sleep 3; done
        echo "mom-live did not become ready; see $LOG" >&2; exit 1 ;;
      kat-coder)
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 nohup "$SERVER" \
          -m /media/knight2/EDS2/models/kat-coder-v2.5-dev/Kwaipilot_KAT-Coder-V2.5-Dev-Q4_K_M.gguf \
          --alias "$ALIAS" --host 127.0.0.1 --port "$PORT" --ctx-size 8192 --parallel 1 \
          --n-gpu-layers 99 --flash-attn on --jinja --reasoning off --split-mode layer --tensor-split 1,1 \
          > "$LOG" 2>&1 &
        echo $! > "$PIDFILE"
        for _ in $(seq 1 90); do answers && { echo "mom-live up on :$PORT (kat-coder, GPUs 1,2)"; start_router; exit 0; }; sleep 2; done
        echo "mom-live did not become healthy; see $LOG" >&2; exit 1 ;;
      *) echo "unknown MOM_DEFAULT=$MOM_DEFAULT (qwen36|kat-coder)" >&2; exit 2 ;;
    esac ;;
  down)
    if [ -f "$ROUTER_PIDFILE" ]; then kill "$(cat "$ROUTER_PIDFILE")" 2>/dev/null || true; rm -f "$ROUTER_PIDFILE"; fi
    if [ -f "$PIDFILE" ]; then kill "$(cat "$PIDFILE")" 2>/dev/null || true; rm -f "$PIDFILE"; fi
    # vLLM runs workers as children; stop anything still serving the alias on this port
    for p in $(ps -eo pid,args | grep "[v]llm serve" | grep -- "--port $PORT" | awk '{print $1}'); do kill "$p" 2>/dev/null || true; done
    for p in $(ps -eo pid,args | grep "[l]lama-server" | grep -- "--port $PORT" | awk '{print $1}'); do kill "$p" 2>/dev/null || true; done
    # vLLM workers keep the V100 lock open for a few seconds after the server exits; wait so that an
    # immediate `up` does not see the lock as held by another job
    for _ in $(seq 1 60); do
      held=0
      for fd in /proc/[0-9]*/fd/*; do [ "$(readlink "$fd" 2>/dev/null)" = "$V100_LOCK" ] && { held=1; break; }; done
      [ "$held" = 0 ] && break; sleep 1
    done
    echo "mom-live stopped" ;;
  status)
    if answers; then echo "mom-live answers on :$PORT"; else echo "mom-live is down on :$PORT"; fi
    if router_up; then echo "mom-router answers on :$ROUTER_PORT"; else echo "mom-router is down on :$ROUTER_PORT"; fi
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader ;;
  *) echo "usage: $0 up|down|status" >&2; exit 2 ;;
esac
