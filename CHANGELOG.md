# Changelog

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
