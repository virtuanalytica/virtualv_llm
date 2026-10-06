#!/usr/bin/env bash
# Live mixture-of-models "mom-live" (scripts/benchmarks/mixture_proxy.py), all six GPUs in use:
#
#   aggregator qwen38-27b-q4  2 replicas  V100 (CUDA 3) :8021, V100 (CUDA 4) :8026
#   proposer   devstral-24b   1 replica   RTX 4000 Ada (CUDA 0) :8022
#   proposer   qwen35-27b     2 replicas  RTX 4000 Ada (CUDA 1) :8023, (CUDA 5) :8025
#   proposer   gemma4-26b-a4b 1 replica   RTX 4000 Ada (CUDA 2) :8024
#   proxy      mom-live       :8030 (OpenAI-compatible; OMP profile "toddler", well_known_suite --external-url)
#
# Replicas are round-robined per request by the proxy. Row split over the NVLinked V100 pair is not
# possible with this llama.cpp build ("device CUDA0 does not support split buffers"), so each V100
# serves its own aggregator replica. Weights: /media/knight2/claude-data/knight1/eds1/models/llm/gguf.
# Every unit is a systemd --user unit: stop with `$0 stop`, inspect with `journalctl --user -u mom-*`.
set -euo pipefail

G=${MODEL_ROOT:-/media/knight2/claude-data/knight1/eds1/models/llm/gguf}
SERVER=/media/knight2/EDS2/tools/llama.cpp/build-v100/bin/llama-server
PROXY="$(cd "$(dirname "$0")/../.." && pwd)/scripts/benchmarks/mixture_proxy.py"
UNITS=(mom-qwen38 mom-qwen38b mom-devstral mom-qwen35 mom-qwen35b mom-gemma4 mom-proxy)
MOM_ARGS=()
RUN_ENV=()
if [[ -n "${MOM_KNOWLEDGE_BUNDLE:-}" ]]; then
  [[ -r "$MOM_KNOWLEDGE_BUNDLE" ]] || { echo "missing knowledge bundle: $MOM_KNOWLEDGE_BUNDLE" >&2; exit 2; }
  MOM_ARGS+=(--knowledge-bundle "$MOM_KNOWLEDGE_BUNDLE")
fi
if [[ -n "${MOM_JEV_URL:-}" ]]; then
  [[ -n "${MOM_JEV_API_KEY:-}" ]] || { echo "MOM_JEV_API_KEY is required with MOM_JEV_URL" >&2; exit 2; }
  MOM_ARGS+=(--jev-url "$MOM_JEV_URL")
  RUN_ENV+=(-E MOM_JEV_API_KEY)
fi
if [[ -n "${MOM_SPECIALIST_URL:-}" ]]; then
  MOM_ARGS+=(--specialist "kimi=${MOM_SPECIALIST_URL}")
  MOM_ARGS+=(--specialist-task coding --specialist-task complex_reasoning --specialist-task long_context)
fi
if [[ -n "${MOM_EVENTS_OUT:-}" ]]; then
  MOM_ARGS+=(--events-out "$MOM_EVENTS_OUT")
fi

download() {
  hf download lmstudio-community/Devstral-Small-2-24B-Instruct-2512-GGUF Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf \
    --local-dir "$G/Devstral-Small-2-24B-Instruct-2512-GGUF"
  hf download lmstudio-community/Qwen3.5-27B-GGUF Qwen3.5-27B-Q4_K_M.gguf --local-dir "$G/Qwen3.5-27B-GGUF"
  hf download lmstudio-community/gemma-4-26B-A4B-it-GGUF gemma-4-26B-A4B-it-Q4_K_M.gguf --local-dir "$G/gemma-4-26B-A4B-it-GGUF"
  hf download unsloth/Qwen3.8-27B-GGUF --include "*UD-Q4_K_M*" --local-dir "$G/qwen38-27b"
}

serve() {  # unit cuda_device port model_file alias ctx
  systemd-run --user --unit="$1" --collect -E CUDA_DEVICE_ORDER=PCI_BUS_ID -E CUDA_VISIBLE_DEVICES="$2" \
    "$SERVER" --model "$4" --alias "$5" --host 127.0.0.1 --port "$3" --ctx-size "$6" --parallel 2 \
    --gpu-layers 99 --flash-attn on --reasoning off --cache-type-k q8_0 --cache-type-v q8_0 --jinja
}

case "${1:-start}" in
  download) download ;;
  stop) systemctl --user stop "${UNITS[@]}" || true ;;
  start)
    bash /media/knight2/EDS2/projects/numerai-signals/signals-repo/scripts/gpu_preflight.sh
    serve mom-qwen38   3 8021 "$G/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf" qwen38 65536
    serve mom-qwen38b  4 8026 "$G/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf" qwen38 65536
    serve mom-devstral 0 8022 "$G/Devstral-Small-2-24B-Instruct-2512-GGUF/Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf" devstral 32768
    serve mom-qwen35   1 8023 "$G/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf" qwen35 32768
    serve mom-qwen35b  5 8025 "$G/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf" qwen35 32768
    serve mom-gemma4   2 8024 "$G/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf" gemma4 32768
    timeout 600 bash -c 'for p in 8021 8022 8023 8024 8025 8026; do until curl -sf -m2 localhost:$p/health >/dev/null; do sleep 5; done; done'
    systemd-run --user --unit=mom-proxy --collect "${RUN_ENV[@]}" python3 "$PROXY" --port 8030 --name mom-live \
      "--aggregator=qwen38=http://127.0.0.1:8021/v1|http://127.0.0.1:8026/v1" \
      --proposer devstral=http://127.0.0.1:8022/v1 \
      "--proposer=qwen35=http://127.0.0.1:8023/v1|http://127.0.0.1:8025/v1" \
      --proposer gemma4=http://127.0.0.1:8024/v1 \
      "${MOM_ARGS[@]}"
    ;;
  *) echo "usage: $0 [start|stop|download]" >&2; exit 2 ;;
esac
