# Two-V100 Open-Weight Model Selection

## Decision

The best deployable near-term design is not an attempt to CPU-offload
DeepSeek-V4 Flash.  It is a two-service, V100-native pair:

1. **Primary service — Qwen3.8-27B, Unsloth UD-Q4_K_M GGUF** on V100 1.
2. **Reasoning/verifier service — DeepSeek-R1-Distill-Qwen-32B, a tested 4-bit GGUF** on V100 2.

The primary service handles normal analysis, retrieval-grounded prompts and
tool requests.  The R1 service is invoked only for high-value mathematical,
code-review or disagreement cases.  This uses both V100s without tensor
parallel synchronization on every token and provides a real independent
second opinion.  The Qwen3.8 decision is based on the local compatibility and
throughput test below, not an unverified claim that it universally outranks
Qwen3.5 or Qwen3-30B.

For maximum concurrent throughput rather than maximum answer quality, run two
identical Qwen3-30B-A3B instances, one per V100, behind a least-queue router.
Two copies improve concurrency, not the quality of an individual completion.

## Hardware and serving constraints

| Constraint | Consequence |
| --- | --- |
| 2 × Tesla V100-SXM2, 32 GiB each, NVLink, SM70 | FP16 and 4-bit GGUF are viable; FP8, FP4 and MXFP4 native execution are not. |
| 64 GiB aggregate VRAM | Dense 24–27B FP16 models can fit when sharded but leave modest KV-cache headroom; 30B+ FP16 is too tight for useful serving. |
| 314.5 GiB DDR4, about 285 GiB currently available | RAM should be used for data/cache only when it improves a measured result; it cannot make unsupported FP8 kernels runnable. |
| 194 GiB free EDS2 space | Enough for several 4-bit models and benchmarks, not the released DeepSeek-V4 Flash checkpoint. |

DeepSeek-V4-Flash-0731 is ~304B parameters with FP8/DSpark-oriented weights.
FP8 weights alone have a ~283 GiB floor before runtime memory and KV cache, and
V100 has no native FP8/FP4 path.  It is therefore rejected for this host rather
than downloaded speculatively.^1

The installed V100-compatible vLLM environment is vLLM 0.6.5 with PyTorch
2.5.1 CUDA 12.4 and `sm_70` kernels.  It is appropriate for compatible
standard architectures.  Modern vLLM DSpark support is newer, while the
current supplied newer CUDA wheel lacks V100 kernels.  Therefore DSpark is not
a credible serving route on this hardware.^2

## Candidate matrix

| Candidate | Producer-reported quality signals | Practical two-V100 format | Recommendation |
| --- | --- | --- | --- |
| **Qwen3.8-27B** | No vendor score was used as a cross-model ranking here; exact Q4 local smoke and concurrent router tests passed. | Unsloth `UD-Q4_K_M` GGUF uses 15.7 GiB resident weights on one V100; the Qwen3.8 hybrid/DeltaNet engine path was verified with current llama.cpp.^15 | **Implemented primary.** Use for normal chat; compare with Qwen3-30B later on the sealed suite. |
| **Qwen3-30B-A3B-Instruct-2507** | MMLU-Pro 78.4, GPQA 70.4, AIME25 61.3, LiveCodeBench 43.2; strong instruction-following scores.^3 | Unsloth reports 17.5 GiB for Qwen3-30B-A3B; use a Dynamic 4-bit GGUF per V100.^4 | Strong **comparison candidate**; test it against implemented Qwen3.8 on the sealed suite. |
| **DeepSeek-R1-Distill-Qwen-32B** | AIME24 72.6, MATH-500 94.3, GPQA-Diamond 62.1, LiveCodeBench 57.2 in DeepSeek's distill table.^5 | Proven ecosystem of 4-bit GGUF/GPTQ variants; choose a tested GGUF and hold one V100 for it. | **Specialist choice.** Use for deliberative reasoning/code verification, not every low-latency request. |
| **Mistral Small 3.2 24B** | MMLU-Pro 69.1, MBPP+ pass@5 78.3, HumanEval+ pass@5 92.9; Apache 2.0 and robust function-calling update.^6 | Full FP16 repository is 96.1 GB, so use 4-bit GGUF per V100 or shard with a deliberately small context. | Good single-model fallback when tool calling and language coverage matter more than R1-style reasoning. |
| **Qwen3.5-27B** | Newer producer table reports MMLU-Pro 86.1, GPQA Diamond 85.5 and SWE-bench Verified 72.4.^7 | A 4-bit checkpoint is plausibly small enough, but its newest multimodal architecture needs a V100 GGUF/engine smoke test first. | **Research candidate, not deployment default** until the precise quantization and engine pass local tests. |
| GPT-oss-20B | Apache 2.0 and official vLLM recipe; model is released as MXFP4-oriented weights.^8 | Native MXFP4 is unsuitable for V100; conversion quality and kernel support would need validation. | Exclude for now. |
| Llama 4 Scout 109B MoE | 17B active / 109B total; custom Llama licence, not open source in the usual sense.^9 | Even 4-bit weights leave little useful KV headroom in 64 GiB. | Exclude. |
| DeepSeek-V4 Flash / DSpark | 304B FP8-oriented weights. | Cannot fit or execute natively on V100. | Exclude. |

All benchmark scores above are vendor/model-card values under differing prompts,
tools and sampling settings.  They are useful for pre-screening, not a
cross-model proof.  A deployment decision must rely on the local evaluation
plan below.

## Serving engines

### Recommended: llama.cpp CUDA + GGUF

Use llama.cpp for the first V100 deployment.  It supports multi-GPU layer
splitting by default and an experimental tensor mode; its documentation
explicitly supports CUDA peer-to-peer and NCCL for cross-GPU reductions.^10
The V100s are NVLinked, so benchmark both `layer` and `tensor` modes, but use
the stable layer mode initially for Qwen3 MoE.  Tensor mode is not implemented
for several MoE architectures, so it must not be assumed to work.^10

Use `CUDA_VISIBLE_DEVICES=1,2`, `GGML_CUDA_P2P=1`, `-ngl all`, and a fixed
4–8K context for the first fit test.  Prefer **one model per V100** for the
router/verifier design; this avoids per-token inter-GPU synchronization and
keeps both cards useful.  Use a split model only for an FP16 24B fallback or a
model that genuinely does not fit on one V100.

### vLLM

Keep vLLM as the API/continuous-batching candidate for a compatible standard
checkpoint.  Its PagedAttention paper reports 2–4× higher throughput than
earlier serving systems under its evaluation settings, especially for larger
and longer-sequence workloads.^11  The installed V100 build must be validated
per model; do not substitute a newer FP8-only wheel merely to obtain DSpark.

### Colibrì

Colibrì is an interesting research alternative, not the default production
engine.  It treats VRAM, RAM and SSD as a hierarchy and streams MoE experts
from disk.^12  Its own public hardware reports are still experimental and
hardware/model specific; an example reports ~2.25 tok/s, while independent
reporting has shown 0.05–0.1 tok/s for a huge streamed model on modest
hardware.^13  It is worthwhile only as a separately sandboxed experiment for
very large MoE models, never as the daily feature/forecast assistant path.

## Unsloth use

Unsloth Dynamic quantizations are good candidates for obtaining Qwen3 GGUFs;
Unsloth reports that Qwen3-30B-A3B fits in 17.5 GiB and evaluates its Dynamic
2.0 approach using MMLU and KL-divergence tests.^4  Treat that as a useful
memory/quantization claim, not an independent quality ranking.  Preserve the
exact repository revision, quantization level and chat template with every
benchmark result.  For the proposed R1 verifier, an independent reproduction
reports Q4_K_M llama.cpp scores close to its local BF16 run across AIME, MATH,
GPQA and LiveCodeBench, which makes Q4 a defensible *test* starting point
rather than an assumption.^14

## Required local benchmark before adoption

1. Run each candidate at Q4 with a 4K context, one request, then 4 concurrent
   requests. Record load time, TTFT, output tok/s, aggregate tok/s, GPU memory,
   temperature and wattage.
2. Run the same 100–200 prompt sealed suite: financial data interpretation,
   Python/SQL, tool-call JSON validity, Dutch/English instructions and
   adversarial factuality prompts. Score correctness and formatting blind.
3. Test the two-model router: Qwen-only versus Qwen→R1 only on uncertain tasks.
   Report quality uplift, added latency and fraction routed. Keep the verifier
   only if its benefit exceeds the latency/energy cost.
4. Pin the winning weights to EDS2 and only then run the 30-minute steady-state
   benchmark. No tokens/s claim is valid until it comes from this machine.

## Local implementation result — 2026-09-11

The selected files are pinned on EDS2: `Qwen3.8-27B-UD-Q4_K_M.gguf`
(16,464,440,224 bytes) and `DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf`
(19,851,335,840 bytes).  Current llama.cpp CUDA build `8ea2902`, compiled for
SM70 and SM86, loaded both successfully with 8K context and Q8 KV cache.

The loopback-only OpenAI-compatible router at `http://127.0.0.1:8010/v1`
routes normal chat to Qwen3.8 on physical V100 1 and analysis/verifier requests
to R1 on physical V100 2, while allowing explicit pinning and queue-aware
fallback.  In the concurrent measured smoke test, Qwen completed a normal
response at 18.806 wall output-tokens/s and R1 completed the reasoning response
at 25.730 wall output-tokens/s.  GPU1 peaked at 97% utilisation, 195.62 W and
49 C; GPU2 peaked at 99%, 202.24 W and 52 C.  The 200 W requested/current
power limits were confirmed by NVML; short 1–2% telemetry excursions are sensor
sampling tolerance, not a changed power-limit setting.  The raw one-second
telemetry and responses are in
`/media/knight2/EDS2/logs/local_chat/router_benchmark_20260911T225611.json`.

This is a functional/throughput validation, not the sealed quality suite in
the preceding section.  The two Q4 models remain running independently, so
their VRAM is not available to a simultaneous GPU training job; the launcher
refuses to start on an already occupied V100.

## Sources

1. DeepSeek-AI. [DeepSeek-V4-Flash-0731 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).
2. vLLM. [DSpark speculator documentation](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dspark/); vLLM. [installation hardware baseline](https://docs.vllm.ai/en/v0.6.5/getting_started/installation.html).
3. Qwen. [Qwen3-30B-A3B-Instruct-2507 model card](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507).
4. Unsloth. [Run & fine-tune Qwen3](https://unsloth.ai/blog/qwen3).
5. DeepSeek-AI. [DeepSeek-R1-Distill-Qwen-32B model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B).
6. Mistral AI. [Mistral Small 3.2 24B model card](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506).
7. Qwen. [Qwen3.5-27B model card](https://huggingface.co/Qwen/Qwen3.5-27B).
8. OpenAI. [gpt-oss-20b model card](https://huggingface.co/openai/gpt-oss-20b).
9. Meta. [Llama 4 Scout model card](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Original).
10. llama.cpp. [Multi-GPU documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md).
11. Kwon et al. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180).
12. Colibrì. [project repository](https://github.com/JustVugg/colibri).
13. Colibrì. [published hardware datapoint](https://github.com/JustVugg/colibri/issues/215); Tom's Hardware. [independent report](https://www.tomshardware.com/tech-industry/colibri-proof-of-concept-gains-frontier-level-1-5-tb-ai-model-novel-approach-runs-on-only-25gb-of-ram-and-shows-promise-for-local-ai-setups).
14. UnicomAI. [DeepSeek-Eval quantization results](https://github.com/UnicomAI/DeepSeek-Eval).
15. Unsloth. [Qwen3.8-27B GGUF release](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF).
