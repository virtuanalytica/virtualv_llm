# Changelog

## v1.0.2 — 2026-10-07

Measurement integrity: error bars, contamination-resistant composites, a
strict contribution protocol, and cloud models measured through one CLI.

### Added

- **Contamination-resistant composites** next to the public one: the mean of
  the post-cutoff holdout and option-reordered MMLU, and a specialist
  composite over the eight text lanes, each with a 95% interval. Canary
  recall and form sensitivity are shown as signals. On current evidence the
  order changes: MiMo-V2.6-Pro drops from 0.79 public to 0.03, and the three
  hosted Gemini rows carry a canary-recall signal.
- **Error bars on every ranked score** (`score_confidence.py`): the public
  composite rests on 298 items and has a margin of about ±3 to ±5 points.
  A paired test shows HumanEval-40 is saturated at the top of the table.
- **Strict contribution protocol**: append-only result submissions under
  `submissions/`, a validator, and a guard that runs from the base branch.
- **Cloud models through the omp CLI**, same suite as local models: GPT-6
  Astra 0.9802, Gemini 3.8 Flash 0.9658, GPT-6 Sol 0.9505, GLM-5.3-Flash
  0.9466, GLM-5.3 0.9436, GPT-6 Luna 0.9257, with decode-rate distributions
  from `omp bench` (`reports/cloud_cli_throughput.json`).
- **Re-baseline on the six-GPU machine**: Qwen3.8 Flash-Next AP-IQ2_S on four
  RTX 4000 Ada cards, 40.61 tok/s, composite 0.9180. Cascade profiles
  `ada4`, `v100pair` and `six`; the Q4_K_M and Q4_K_XL quants are candidates
  again.
- **Tool experts for the FQ and video lanes** and a 73-item parametric FQ
  pack: GLM-5.3-Flash goes from 35 of 73 bare to 65 of 73 with the simulator
  loop (v2 pack; v3 states the duration limit in the task text).

### Fixed

- **The "gsm8k 404" of v1.0.1** was an `--external-url` ending in `/v1`
  while the suite appends `/v1/...` itself. The suite now refuses that URL.
- The suite checks for the HumanEval data before starting; six runs were
  lost on the last task in worktrees without it.
- The contribution guard decides maintainership from repository permission;
  the event label reports the owner as an outside contributor.

### Known issues

- The contamination-resistant composite rests on 27 items and the holdout
  pack is public in this repository; there is no rotating live lane yet.
- Video scores 0 for every bare model; no lane has the 73 items needed to
  support a "95% correct" claim except FQ.
- The V100-pair and six-GPU Flash-Next rows and variants 3 and 5 are still
  being measured.

## v1.0.1 — 2026-10-07

Hardware-tier expansion and the first EDSQ-Volta contribution cycle. This
release adds three storage/compute tiers to the benchmark envelope and new
insights for advancing the suite:

### Added

- **4× RTX 4000 Ada results.** Full battery rows and dedicated cascade
  runners for the Ada/A4000 lanes: `mistral-small4-119b-q4-adaa4000`,
  `deepseek-v4-flash-reap150b-q2k-adaa4000`, plus
  `run_qwen38_flash_next_gguf_cascade.py`,
  `run_qwen38_flash_next_vllm_cascade.py` and
  `run_glm53_reap50_cascade.py`.
- **6.4 TB NVMe tier as the model home** (`eds1`). Provisioned after the
  chassis power cycle renamed the NVMe devices; mount by-UUID (fstab), not
  by device name — the rename silently unmounted both data volumes.
- **PMem 100 Optane 4×512 GB tier** (App Direct, per-socket interleaved,
  fsdax/ext4 with `dax=always`). Measured contributions and new insights:
  - parallel reads 10.1–10.5 GB/s per mount on the local socket,
    saturating at 4 readers; remote-socket access collapses to
    0.4–1.0 GB/s → NUMA-local expert placement is mandatory;
  - DAX bypasses the page cache: every PMem-resident expert read is real
    Optane traffic — tier sizing must budget for it;
  - symlink tier-sharding serves >RAM models (K2.7-Code, 544 GB) as one
    GGUF spanning NVMe-page-cache + PMem-DAX.
- **EDSQ-Volta section on the dashboard** (Oct 2026): tier-balanced MoE
  judges — Kimi K2.5 (1.24 t/s), DeepSeek-V4-Flash-0731 (2.30 t/s),
  GLM-5.3-Flash (up to 6.2 t/s), K2.7-Code (0.68 t/s) — and mixture
  latency v2→v3 (19m37s → 5m21s per question at equal verdict quality;
  the judge corrected all three wrong proposer drafts).
- **Thinking-budget rule** (suite-relevant): llama.cpp separates thinking
  into `reasoning_content`, but `max_tokens` counts both — budget
  `thinking + visible answer` or scored runs truncate mid-reasoning.

### Known issues

- The gsm8k lm-eval step returns HTTP 404 against llama.cpp external
  servers right after the MMLU block (reproduced on GLM-5.3-Flash and
  DeepSeek-V4-Flash-0731; BBH + MMLU complete normally). Full batteries
  for these members are pending this fix.
- NVFP4-TP2 non-determinism/repetition-loop instability on the 1Cat
  target lane is documented in the serve script header and remains open.

### Credits

- 4× RTX 4000 Ada battery results, 6.4 TB NVMe tier and PMem 100 Optane
  4×512 GB tier: contributions and new insights from the
  fieldintelligence lab (node2) for advancing the benchmark suite.
- Tier-balanced MoE judges and evidence: EDSQ-Volta
  (github.com/fieldintelligence/EDSQ-Volta), porting the optimization
  playbook of [1CatAI/1Cat-vLLM-Gaudi](https://github.com/1CatAI/1Cat-vLLM-Gaudi).
