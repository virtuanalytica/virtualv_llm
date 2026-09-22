#!/usr/bin/env bash
# Compare target-only and DFlash2 under one local TP=2 contract.
set -euo pipefail

ROOT=/media/knight2/EDS2/projects/numerai-signals
LOG_ROOT=/media/knight2/EDS2/logs/vllm-1cat
PORT=18012
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
PROFILE_TAG=${PROFILE_TAG:-8k}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
CONCURRENCY=${CONCURRENCY:-1}
SERVER_PID=

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  systemctl --user start local-chat-qwen38.service local-chat-deepseek-r1.service || true
}
trap cleanup EXIT INT TERM

mkdir -p "$LOG_ROOT" "$ROOT/reports"
systemctl --user stop local-chat-qwen38.service local-chat-deepseek-r1.service
# The desktop may remain on GPU 0, but no model process is allowed there.
systemctl --user stop llama-qwen.service ollama.service || true

for MODE in target dflash2; do
  LOG="$LOG_ROOT/${MODE}_tp2_${PROFILE_TAG}_20260917.log"
  PORT=$PORT MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS setsid "$ROOT/infra/vllm_v100/serve_1cat_qwen38_tp2.sh" "$MODE" >"$LOG" 2>&1 &
  SERVER_PID=$!
  READY=0
  for _ in $(seq 1 120); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      break
    fi
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      READY=1; break
    fi
    sleep 5
  done
  if [[ "$READY" != 1 ]]; then
    echo "$MODE failed to become healthy; inspect $LOG" >&2
    tail -80 "$LOG" >&2 || true
    exit 1
  fi
  /usr/bin/python3 "$ROOT/infra/vllm_v100/benchmark_1cat_tp2.py" \
    --mode "$MODE" --port "$PORT" --concurrency "$CONCURRENCY" \
    --out "$ROOT/reports/1cat_${MODE}_tp2_${PROFILE_TAG}_20260917.json"
  kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=
  sleep 5
done
