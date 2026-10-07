# VirtualV LLM model test roadmap

Status date: 2026-10-07 (sections dated 2026-09-24 are kept as written). This document separates measured local evidence from
upstream reference numbers and from untested hypotheses.

## 2026-10-07 update — EDSQ-Volta takeover cycle (fieldintelligence)


**Completed 5–7 Oct.** Tier-balanced MoE judges live and measured: K2.5 1.24,
DeepSeek-V4-Flash-0731 2.30, GLM-5.3-Flash 2.41–6.20, K2.7-Code 0.68 tok/s
(E-config flags, 6-GPU attention; full evidence in
`fieldintelligence/EDSQ-Volta` → `evidence/`). MoM v2 (K2.5 judge)
19m37s/question → MoM v3 (DSv4 judge) 5m21s at equal verdict quality
(portfolio 8.2 % vs wrong drafts on all three proposers). PMem 100 Optane
4×512 G tier live (App Direct/fsdax/DAX): 10.1–10.5 GB/s per mount local
socket, remote-socket 0.4–1.0 → NUMA-local placement mandatory. 6.4 TB NVMe
model home (`eds1`, mount by-UUID). Day/night profile live via
`~/bin/vllm-profile` (cron 07:30/19:00): day = 1Cat TP2 NVFP4 target
46.6–49.0 t/s (meets the 50 t/s goal), night = judges + queue worker.

**Baseline discrepancy to resolve:** the DSv4-0731 re-test measured 2.30 t/s
(UD-IQ4_XS, E-config) against the Sept row of 13.89 t/s (UD-IQ3_XXS), and
GLM-5.3-Flash 2.41–6.20 t/s (UD-Q4) against 21.49 t/s (AJ-IQ2_XXS). Different
quant, flags and tier state — re-test under the Sept configuration before
treating either number as canonical.

**Backlog (ordered):**
1. P1 — RESOLVED 2026-10-07. The HTTP 404 was not the gsm8k step (GSM8K
   and TruthfulQA had finished). The batteries were started with
   `--external-url http://host:port/v1`, and the suite appends `/v1/...`
   itself, so its first direct request went to `/v1/v1/chat/completions`;
   llama-server answers 200 on `/v1/chat/completions` and 404 on the doubled
   path (checked on the running :8011 server). `well_known_suite.py` now
   refuses such a URL at start. Still to do: re-run the GLM-5.3-Flash and
   DSv4-Flash-0731 batteries with the server root; their rows stay failed
   until then.
2. K2.7-Code UD-Q3_K_XL (432 G) downloaded — serve + compare vs Q4_K_XL
   (0.43–0.68 t/s baseline; Q3 fits the page cache without a PMem tail).
   Test chain pattern: `evidence/microbench/kimi_k27_code_tier_shard_20261006.md`.
3. K3 UD-IQ2_XXS (662 G) auto-chain armed: download → symlink tier-split
   (~450 G NVMe + ~212 G PMem) → serve :18022 → portfolio probe. Log:
   `/tmp/k3_test.log`. 2.5-bit class; projected ceiling ~1–1.5 t/s
   (bandwidth-bound) — quality verdict decides adoption.
4. MoM v3.1 day lane: judges are night-profile by design; evaluate the
   1Cat NVFP4 lane for scored daytime MoM after item 7's quality gate.
5. GLM-5.3-Flash REAP50 GGUF conversion bug (missing
   `glm5-next.attention.indexer.kpool` tensor) — report upstream; the
   official unsloth conversion works.
6. Colibri source checkout (`coli build` needs a clone) + qwen38 A/B
   against llama.cpp — decides whether the CPU expert tier switches engines.
7. NVFP4-TP2 repetition-loop instability: quality gate (8-task battery) on
   the day lane before scored daytime outputs are trusted.
8. JEV SystemOne adapter for well_known_suite: stateful protocol
   (`POST /v1/systemone`, state + questions), key `virtualv-mom-local`
   (hash-only in `~/.config/toddler-jev/keys.json`) — needs a small
   adapter, not a suite flag.

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

## 2026-10-07: re-baseline on the current hardware

The machine now holds four RTX 4000 Ada 20 GB (CUDA 0, 1, 2, 5) and two Tesla
V100-SXM2 32 GB on NVLink (CUDA 3, 4); the RTX A4000 is gone. The Qwen3.8
Flash-Next AP-IQ2_S baselines above were measured on the old four-card set
(A4000 + 2x V100 + one Ada), so they are historical. They are re-measured
first, under the unchanged protocol `v4-mmlu-fewshot-20260918`:

| Row to measure | Topology | Old reference | Result |
|---|---|---|---|
| Qwen3.8 Flash-Next AP-IQ2_S, six GPUs | 4x Ada 20 GB + 2x V100 32 GB, layer split | 40.79 tok/s, composite 0.9232 (old all-four) | pending |
| Qwen3.8 Flash-Next AP-IQ2_S, four Ada | 4x RTX 4000 Ada 20 GB, layer split | none | **40.61 tok/s, composite 0.9180** (`qwen38-flash-next-ap-iq2s-ada4`, 4K context, about 53 GB in VRAM); inside the margin of the old 0.9232 |
| Qwen3.8 Flash-Next AP-IQ2_S, V100 pair | 2x V100 32 GB NVLink, layer split | 42.79 tok/s, composite 0.9113 | pending |

This table is updated in place when a row is measured: the value, the result
row's model identifier and the pull request that published it.

## 2026-10-07: five variants expected to beat the baseline on both axes

Target to beat: 40.79 tok/s and composite 0.9232 together. Each line is a
hypothesis with the measurement it rests on; none is a result yet.

| # | Variant | Why it should be faster | Why composite should hold or rise | Result |
|---|---|---|---|---|
| 1 | Qwen3.8 Flash-Next NVFP4, 1Cat-vLLM TP2 target-only, V100 pair | Upstream reports 80.7 tok/s; locally the same engine gives 111 tok/s on Qwen3.6-35B-A3B NVFP4 | 4-bit weights versus IQ2_S. The existing `qwen38-1cat-vllm-target` row scores 0.34, which points at a template or scoring fault to fix first | pending |
| 2 | Variant 1 with DFlash2/MTP speculative decoding | 81.59 tok/s measured at B1 on this pair | Speculation is verified against the target, so quality equals variant 1; reported as its own row | pending |
| 3 | Qwen3.8 Flash-Next AP-Q4_K_XL, six GPUs, fully in VRAM | 34.59 tok/s on the V100 pair with the remainder outside VRAM; 144 GB now holds all 101 GB | Highest measured local composite, 0.9272 | pending |
| 4 | Qwen3.8 Flash-Next AP-IQ2_S, V100 pair, tensor split | Tensor split is the fastest llama.cpp topology on this NVLink pair (44.87 tok/s for the 27B model versus 33 with layer split) | Same weights as the baseline | pending |
| 5 | Qwen3.8 Flash-Next AP-Q4_K_M, six GPUs | 40.00 tok/s on the old set, where the A4000 was the slowest card | 0.9175 measured, inside the baseline's margin, at 4-bit | pending |

Reading the result: the composite has a 95% margin of about ±3.8 points on
these rows (`reports/score_confidence.json`), and all measured Flash-Next rows
are tied. A variant counts as faster on a measured tok/s difference. It counts
as better on composite only if a confirmation run on a larger sample separates
it; until then "not lower" is the honest claim. Variants that are faster but
clearly lower (Qwen3.6-35B-A3B NVFP4 at 111 tok/s and 0.841, Kat-Coder v2.5 at
91.7 tok/s and 0.907) stay in the speed lane and are not listed here.

Execution starts after the runs that are in flight on 2026-10-07 have ended
and the V100 lock is free.

## 2026-10-07: cloud providers through the omp CLI

`omp` (18.6.1) fronts several providers with one print mode and one
accounting format, so each goes through the same suite as a local model
(`run_external_provider_cascade.py --provider omp-…`, no tools, no local
context). Throughput comes from `omp bench` as a distribution
(`omp_throughput.py`, median and 95th percentile), not from one completion.

| Provider key | Models | Smoke test | Full suite |
|---|---|---|---|
| `omp-zai` | glm-5.3-flash, glm-5.3 | pass | re-queued (first runs lost to a missing HumanEval file) |
| `omp-openai-codex` | gpt-6-luna, gpt-6-astra, gpt-6-sol | pass | gpt-6-sol: composite 0.9505 (`cloud-omp-openai-codex-gpt-6-sol`); luna and astra re-queued |
| `omp-google` | gemini-3.8-flash | pass | done (`cloud-omp-google-gemini-3.8-flash`) |
| github-copilot | all tried | 400 "model not supported" | not registered |
| google-antigravity | gemini-3.8-flash | omp: unhandled API mapping | not registered |
| grok-build | grok-4.5 | no answer (not signed in) | not registered |

Decode rate over 10 chat requests each (`reports/cloud_cli_throughput.json`),
median with the observed range: Gemini 3.8 Flash 94 tok/s (83 to 702; the top
value is a burst on a very short decode window, which is why the median is
published), GPT-6 Luna 83 (53 to 104), GLM-5.3 60 (47 to 72), GLM-5.3-Flash
46 (43 to 53), GPT-6 Sol 44 (33 to 74), GPT-6 Astra 21 (18 to 28). Time to
first token is 2.0 to 4.5 s at the median.

These rows run without the per-request output cap local rows get (the CLI
cannot truncate), which the row records in `output_budget`.

## 2026-10-07: tool experts for the FQ and video lanes

`specialist_experts.py` scores the bare model and the same model with one
tool on identical items (FQ: forward-kinematics simulator; video: FFmpeg
render and repair, at most three rounds). Results go to
`reports/specialist_experts.json`. First row, GLM-5.3-Flash through omp: FQ
0 of 2 bare to 2 of 2 with the simulator, video 0 of 3 to 1 of 3 with the
repair loop. The bare FQ answers failed as unparseable JSON, so part of the
gain is format repair. A 73-item parametric FQ pack
(`fq_v2_parametric.csv`) is being measured to replace the 2-item reading.

## 2026-10-07: disk freed for the Flash-Next runs

Removed from the model disk after the retention check, with restore commands
in `ARCHIVED_MODELS_MANIFEST.md`: Qwen3.5-397B-A17B UD-Q4_K_M (244 GB, never
measured, sizes matched the Hub) and MiMo-V2.6-Pro BPW2.5 (298 GB, complete
row kept, SHA-256 per shard recorded). The model disk went from 111 GB to
570 GB free. Kimi K2.7 Q3 and Q4 are kept on purpose. Flash-Next AP-IQ2_S,
AP-Q4_K_M and AP-Q4_K_XL (81.6, 94.5 and 101.1 GB, pinned revision) are on
disk for the re-baseline and for variants 3 and 5; 378 GB remains free.

## 2026-10-07: related plans

- Specialist mixture of experts with the self-healing and self-learning
  loops: [SPECIALIST_MIXTURE_OF_EXPERTS_PLAN.md](SPECIALIST_MIXTURE_OF_EXPERTS_PLAN.md).
- Agent CLIs as benchmark rows: see the omp section below. The direct ZCode
  backend in `external_cli_agent_proxy.py` still waits on a model provider in
  the headless CLI's own config; Z.ai's GLM is reached through omp instead.
- Lessons from professional suites and what was adopted:
  [PROFESSIONAL_SUITE_LESSONS.md](PROFESSIONAL_SUITE_LESSONS.md).

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
