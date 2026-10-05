# VirtualV LLM model test roadmap

Status date: 2026-09-24. This document separates measured local evidence from
upstream reference numbers and from untested hypotheses.

## 2026-09-24 decisions

- MiMo-V2.6-Pro q4 (MXFP4) and q3 (BPW3.0/BPW3.5) are not planned. They do
  not fit this disk or these V100s, and they are the wrong next step.
- A usable rate under 3 tokens/second is not acceptable on this machine.
  Mixture throughput is the sequential rate, `1 / sum(1/member_t/s)`, not
  the slowest member's own rate.
- The live MiMo-V2.6-Pro BPW2.5 suite stays. Its measured decode is above
  that floor. Mixture rows already use the sequential rate, and the search
  keeps combinations at or above 3 tok/s. The next work is the ordered queue
  below, after the disk gate, not another MiMo quant.

## Current local baselines

| Goal | Local baseline | Result |
|---|---|---|
| Fast, balanced model | Qwen3.8 Flash-Next AP-IQ2_S, all four GPUs | 40.79 tok/s; public composite 0.9232 |
| Fastest measured Qwen row | Qwen3.8 Flash-Next AP-IQ2_S, 2x V100 | 42.79 tok/s; public composite 0.9113 |
| Quality candidate | DeepSeek-V4-Flash-0731 UD-IQ3_XXS | 13.89 tok/s; public composite 0.9103 |
| Agentic candidate | GLM-5.3-Flash AJ-IQ2_XXS | 21.49 tok/s; reasoning-effort fix still requires a clean re-test |

These values are read from `reports/well_known_suite_20260917.json`. Different
engines, contexts and topologies remain separate rows.

## Disk gate before the queue

Checked 2026-09-24 11:24 CEST. `/media/knight2/EDS2/models/llm` contains only
`mimo-v26-pro-bpw2.5` (319,743,531,488 bytes, suite still running) and
`qwen38-27b` (live chat on GPU 3, port 8011). `df` showed about 74 GB free.
A download starts only when free space exceeds the artifact size plus the
24 GB reserve, so nothing of about 50 GB or larger fits while the MiMo shards
stay. Do not delete those shards until `mimo-v26-pro-bpw25.service` has exited
and `/tmp/v100_exclusive.lock` is free. The Qwen chat weights stay.

## Ordered experiment queue

1. **GLM AJ-IQ2_XXS re-test, after a re-download.** The weights are not on
   disk. Recorded size is 87,346,006,560 bytes (two shards in
   `scripts/benchmarks/run_glm53_hardware_matrix.py`, repo
   `aj9o9/GLM-5.3-Flash-GGUF`, revision
   `07c62fcdeaf1c05d22bd123c3da8058a1b1e63e2`). Re-run only after the disk gate
   and the V100 lock both pass. Preserve the old result as superseded evidence
   and use a new run identifier. `glm53-reap50-retest.service` is inactive, not
   waiting on the lock. Its IQ3_M weights (72,132,392,352 bytes) are also gone,
   so starting that unit would download before it can score. Leave it stopped
   until the same gate passes.
2. **DeepSeek-V4-Flash-0731 placement matrix, after a re-download.** The
   UD-IQ3_XXS directory is gone. The benchmark registry describes four shards,
   about 104 GiB. Same disk and lock gates. Once the weights are back, use the
   measured 13.89 tok/s row as baseline. Test V100 layer split and the supported
   all-four layer split with identical context and prompts. Treat “V100 experts,
   RTX attention/KV, CPU remainder” as a hypothesis until a runtime exposes and
   verifies tensor-class placement; never infer it from aggregate VRAM use.
3. **Qwen quant improvement.** AP-Q4_K_XL (101,142,769,536 bytes) was removed
   on 2026-09-24; the restore command is in
   `/media/knight2/EDS2/models/ARCHIVED_MODELS_MANIFEST.md` section 3. Download
   only when free space exceeds that size plus the 24 GB reserve. The measured
   rows stay valid. Promote either a speed champion (at least 42.79 tok/s and
   composite at least 0.91) or a balanced champion (composite above 0.9232 and
   at least 35 tok/s).
4. **1Cat-vLLM baseline before branch work.** Pin release `v1.5.0`, run its
   SM70 preflight in an isolated environment and establish target-only quality
   before MTP. Upstream reports 80.732 tok/s target-only and 138.26 tok/s MTP4
   for Flash-Next under its own recorded contract; those are not local claims.
5. **1Cat branch trials.** Evaluate only branches tied to Qwen Flash-Next,
   DeepSeek V4 or GLM 5.3. Record branch commit, wheel hash and build log. Promote
   a branch only if it passes API, determinism, quality and throughput regression
   gates against `v1.5.0`; never merge an experimental fork directly into the
   benchmark controller.
6. **DS4-specific DeepSeek quant.** A DS4 Q2/mixed checkpoint requires its own
   pinned runtime and a new download. Start only after the disk gate passes and
   keep its score separate from generic llama.cpp GGUF results.

## Sources and non-portable reference results

- [1Cat-vLLM 1.5.0](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.0)
  documents SM70 paths for Qwen3.8 Flash-Next, DeepSeek-V4-Flash and GLM-5.3,
  with explicit warnings that results are workload/topology specific.
- [Qwen3.8 Flash-Next AP-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-AP-GGUF)
  is the source of the locally tested AP quant family.
- [Qwen3.8 Flash-Next NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
  is a mixed-format checkpoint; support must be demonstrated by the selected
  runtime rather than inferred from the model name.
- [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
  is the official quality target. [DS4 model guidance](https://github.com/antirez/ds4/blob/main/docs/MODELS.md)
  applies only to DS4-compatible files and runtime revisions.
- [NVIDIA GLM-5.3-Flash NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)
  remains disk-gated; the existing GGUF re-test comes first.

## Promotion and evidence rules

- Smoke gate: pinned source hash, engine health, correct model identity, no OOM,
  and a deterministic short output check.
- Full gate: current protocol GSM8K, BBH, MMLU and HumanEval plus specialist
  holdout; public-suite leader and private/holdout leader are separate titles.
- Throughput gate: report prompt and decode separately, including context,
  batch/concurrency, warm/cold state, physical GPU IDs and telemetry.
- MTP/speculative rows never replace target-only rows. Failed or unsupported
  profiles remain visible without a numeric score.
- Do not delete a model until its result, source revision, SHA-256 and raw logs
  are present in durable evidence and a retained winner is identified.
