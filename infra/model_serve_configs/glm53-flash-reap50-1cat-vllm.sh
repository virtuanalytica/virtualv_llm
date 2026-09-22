#!/usr/bin/env bash
# DO NOT RUN -- BLOCKED, confirmed non-viable on this hardware, 2026-09-21.
#
# GLM-5.3-Flash-REAP50-NVFP4 (compressed-tensors, safetensors, 103.02GB) hits
# a final, unfixable blocker on every GPU in this box:
#   NotImplementedError: No NvFp4 MoE backend supports the deployment configuration
# NVFP4 MoE compute kernels require native FP4 tensor cores (Blackwell-only);
# none of A4000 (Ampere)/2xV100 (Volta)/RTX4000Ada (Ada) have them, and this
# vLLM fork has no software fallback for them (unlike its sparse-attention
# indexer, which does have a real SM70 fallback -- see below). This checkpoint
# was deleted after confirming the failure; this script is kept only for its
# investigation history in case glm5next NVFP4 support is revisited after a
# future hardware change (Blackwell GPU) or a fork update.
#
# Full history (2026-09-21, in response to a "be more creative, check
# Colibri/vLLM/smaller quants" follow-up after the GGUF path below was
# blocked):
#
# 1) GGUF quant (IQ3_M, 72.13GB) via llama.cpp: llama.cpp doesn't support
#    glm5next at all (unmerged upstream, ggml-org/llama.cpp #27752/#27754/#27773).
#    vLLM's own GGUF loader also failed independently: it routes GGUF files
#    through transformers' generic load_gguf_checkpoint(), which has its OWN,
#    separate, smaller architecture list and doesn't know "glm5-next" either --
#    ValueError: "GGUF model with architecture glm5-next is not supported yet."
#    A transformers-library limitation, not a vLLM or hardware one.
# 2) Switched to this safetensors NVFP4 checkpoint (bypasses the GGUF bridge
#    entirely -- config.json declares architectures=["Glm5NextForConditionalGeneration"],
#    quant_method="compressed-tensors", both match vLLM's registry directly).
#    This box's 1Cat-vLLM 1.5.0 DOES have a complete native glm5next
#    implementation (vllm/models/glm5next/, with a dedicated sm70/ subpackage --
#    same V100-specific-support pattern as Qwen3.6-35B-A3B-NVFP4's PR #270), so
#    this path genuinely progressed further than GGUF:
#    a) Default attention backend, all 4 GPUs: crashed -- Sparse Attention
#       Indexer requires DeepGEMM (Hopper/Blackwell-only).
#    b) --attention-backend GLM5_SM70_SPARSE (a real SM70-compatible sparse-MLA
#       backend, vllm/models/glm5next/sm70/sparse.py), still all 4 GPUs: SAME
#       crash -- the DeepGEMM bypass is a PER-WORKER check
#       (_is_exact_sm70_cuda()), true for the 2 V100 workers but false for
#       A4000/RTX4000Ada, and one failed worker kills the whole engine.
#    c) Same SM70 backend, V100 PAIR ONLY (tensor-parallel-size 2,
#       CUDA_VISIBLE_DEVICES=1,2) so every worker is genuinely SM70: indexer
#       crash GONE (log confirmed "Applied 1Cat SM70 serving defaults"),
#       engine init started -- real progress.
#    d) NEW, final, unfixable blocker: the NvFp4 MoE kernel NotImplementedError
#       above. Matches the same root cause independently confirmed earlier
#       this session for the unrelated Qwen3.8-target NVFP4 checkpoint on
#       Ada+A4000.
#
# Not attempted (out of scope for a bounded benchmark): writing custom SM70
# NVFP4 MoE kernels. Open, untested lead if the GLM line is revisited:
# GLM-4.5-Air's smaller quants (Q2_K/UD-Q2_K_XL/IQ1/IQ2, 38-47GB) would fit the
# V100 pair's 64GB VRAM with little/no CPU offload via plain llama.cpp GGUF
# (glm4moe architecture IS supported there).
#
# Full writeup: reports/dual_v100_candidate_research_20260918.json's
# GLM-5.3-Flash-REAP50 entries.
set -euo pipefail
exit 1  # blocked -- see header. Remove this guard only after a genuine fix (new hardware, fork update).

PORT=${PORT:-18017}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-40}
CUDA_VIS=${CUDA_VIS:-1,2}
TP_SIZE=${TP_SIZE:-2}
ROOT=/media/knight2/EDS2
ENV_DIR="$ROOT/envs/1cat-vllm-1.5.0"
TARGET="$ROOT/models/1cat-vllm/GLM-5.3-Flash-REAP50-NVFP4"
MODEL_NAME=${MODEL_NAME:-glm53-flash-reap50}

test -x "$ENV_DIR/bin/vllm"
test -f "$TARGET/model.safetensors.index.json" || { echo "model not found under $TARGET" >&2; exit 1; }

# Uses all 4 GPUs -- does NOT take the V100-exclusive lock (that lock is for
# jobs that need the V100 pair specifically free of other tenants; this job
# spans beyond it). Confirm no other job is using A4000/RTX4000Ada before
# running this manually.

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$CUDA_VIS"
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$ENV_DIR/bin:$PATH"
export HF_HOME="$ROOT/cache/huggingface-1cat"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$ROOT/cache/xdg-1cat"
export VLLM_CACHE_ROOT="$ROOT/cache/vllm-1cat"
export TMPDIR="$ROOT/tmp/vllm-1cat"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" "$ROOT/logs/vllm-1cat"

# 2026-09-21 history:
# Attempt 1 (default attention backend, all 4 GPUs): crashed --
#   "RuntimeError: Sparse Attention Indexer CUDA op requires DeepGEMM to be
#   installed" -- DeepGEMM only supports Hopper (SM90)/Blackwell (SM100).
# Attempt 2 (--attention-backend GLM5_SM70_SPARSE, still all 4 GPUs): SAME
#   crash. Root cause found in sparse_attn_indexer_kpool.py's own source:
#   the DeepGEMM requirement is skipped only when
#   `_is_exact_sm70_cuda()` is True for THAT WORKER's own GPU (a per-worker
#   compute-capability check, not a backend selection) -- true for the V100
#   workers, but the A4000 (SM86) and RTX4000Ada (SM89) workers in this 4-GPU
#   TP group correctly fail the check and crash, which kills the whole
#   engine (one failed worker takes down the group).
# Attempt 3 (this one): V100 pair ONLY (CUDA_VISIBLE_DEVICES=1,2,
#   tensor-parallel-size 2) so EVERY worker is genuinely SM70 and the
#   DeepGEMM bypass applies everywhere. Trades VRAM headroom for
#   compute-capability homogeneity -- only 64GB combined vs the model's
#   ~96-103GB, so a much larger --cpu-offload-gb is needed (accept whatever
#   tok/s this yields per this session's "5 tok/s floor, accept anything
#   above it" guidance -- don't assume it'll be fast).
exec numactl --interleave=all "$ENV_DIR/bin/vllm" serve "$TARGET" \
  --served-model-name "$MODEL_NAME" \
  --trust-remote-code --dtype half --tensor-parallel-size "$TP_SIZE" \
  --cpu-offload-gb "$CPU_OFFLOAD_GB" \
  --attention-backend GLM5_SM70_SPARSE \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.95 \
  --max-num-batched-tokens 4096 --max-num-seqs "$MAX_NUM_SEQS" \
  --enable-prefix-caching \
  --generation-config vllm \
  --host 127.0.0.1 --port "$PORT"
