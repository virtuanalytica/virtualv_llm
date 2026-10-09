# Changelog

## v1.0.3 — 2026-10-09

A working day lane, the Claude line from 4.5 upward, a complete dashboard
and the first Kimi numbers on the V100 pair.

### Added

- **Claude models from 4.5 to current, Fable excluded**: Opus 5 0.9820,
  Opus 5.5 0.9818, Opus 4.8 0.9700, Sonnet 5 0.9643, Sonnet 5.5 0.9542.
  Opus 4.7, 4.6, 4.5 and Sonnet 4.6, 4.5 are not measured yet: the Claude
  CLI returned 502 on three attempts.
- **DeepSeek-V4-Flash-0731 UD-IQ4_XS** on the V100 pair with experts in RAM:
  composite 0.9362 at 3.16 tok/s (`dsv4-flash-0731`).
- **Kimi K2.7-Code Q3 vs Q4 and Kimi K3 IQ2_XXS** on the V100 pair, as fixed
  probes rather than suite rows (under 1 to 3 tok/s): Q3 1.04 tok/s on a
  1200-token code task against 0.68 for Q4, K3 0.14 to 0.22 tok/s. Raw
  evidence is in fieldintelligence/EDSQ-Volta.
- **The main table lists every measured row** (72 to 99 rows), with statuses
  for superseded, retired, unsupported and speed-excluded rows.
- **Publication dates** with a source URL on 42 rows, and a **BetterBench
  self-assessment** section on the dashboard.

### Changed

- **The day lane is Qwen3.8 Flash-Next AP-IQ2_S** (composite 0.915 at 40.8
  tok/s). The 1Cat NVFP4 lane it replaces scored 0.34 on the same suite.
- The Fable 5.1 row is removed; Fable is out of scope.

### Fixed

- The agent-CLI proxy retries transient provider errors.
- The suite takes `--lm-eval-timeout` and `--request-timeout` for slow
  models, and `VIRTUALV_SKIP_DASHBOARD=1` skips the dashboard rebuild.

### Known issues

- Roadmap variants 1, 2 and 4 cannot run on this hardware (Flash-Next NVFP4
  does not load on two V100s; llama.cpp has no tensor split for qwen4exp).
- GLM-5.3-Flash UD-Q4_K_XL has no complete row: at about 1 tok/s the suite
  needs some 15 hours and two attempts were cut short.
- DeepSeek-V4 and GLM need `--reasoning off` on llama-server; with reasoning
  on, 78% of GSM8K answers came back empty.
- Kimi K3 on six GPUs is being measured in a separate run.

## v1.0.2 — 2026-10-07

Measurement integrity: error bars, contamination-resistant composites, a
strict contribution protocol, and cloud models measured through one CLI.

### Added

- **Contamination-resistant composites** next to the public one: the mean of
  the post-cutoff holdout and option-reordered MMLU, and a specialist
  composite over the eight text lanes, each with a 95% interval. Canary
  recall and form sensitivity are shown as signals. On current evidence
  MiMo-V2.6-Pro drops from 0.79 public to 0.08 and the local GLM-5.3 REAP50
  quant from 0.89 to 0.72, while seven of 30 audited rows carry a
  canary-recall signal (Claude Sonnet 5.5 continues 9 of 10 held-out GSM8K
  questions verbatim, Claude Opus 5.5 8, GPT-6 Astra 7). At the
  top the resistant composite is saturated: twelve rows score 0.95 or more
  on 26 items.
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
- **Six-GPU and V100-pair rows**: AP-IQ2_S 0.9148 at 40.80 tok/s on six
  cards and 0.9113 at 40.77 on the V100 pair; AP-Q4_K_M 0.9342 at 38.52 and
  AP-Q4_K_XL 0.9193 at 38.31 on six cards. More cards do not speed this model
  up under layer split, and neither Q4 variant beats the baseline on both
  axes.
- **Claude through the `claude -p` CLI**: Opus 5.5 0.9818, the new leader
  and tied with GPT-6 Astra (canary recall 0.8); Sonnet 5.5 0.9542 (canary
  recall 0.9). Haiku 4.5 is listed with a caveat (MMLU artifact); Fable 5.1 is
  blocked on usage credits.
- **Tool experts for the FQ and video lanes** and a 73-item parametric FQ
  pack: with the duration limit stated in the task (v3), GLM-5.3-Flash goes
  from 31 of 73 bare to 70 of 73 with the simulator loop (v2, limit unstated:
  35 to 65).

### Fixed

- **Post-cutoff holdout answer key.** Three keys were wrong and one item had
  no correct option; nearly every model gave the same "wrong" answer and the
  arithmetic confirmed the models. Keys corrected, the unsolvable item
  disabled, all 24 audit rows re-scored from their stored predictions, and a
  test now recomputes every arithmetic key.

- **The "gsm8k 404" of v1.0.1** was an `--external-url` ending in `/v1`
  while the suite appends `/v1/...` itself. The suite now refuses that URL.
- The suite checks for the HumanEval data before starting; six runs were
  lost on the last task in worktrees without it.
- The contribution guard decides maintainership from repository permission;
  the event label reports the owner as an outside contributor.

### Known issues

- The contamination-resistant composite rests on 26 items and the holdout
  pack is public in this repository; there is no rotating live lane yet.
- Video scores 0 for every bare model; no lane has the 73 items needed to
  support a "95% correct" claim except FQ.
- Few-shot prompts reach agent CLIs as one message; Claude Haiku 4.5 then
  reviews all questions instead of answering the last (MMLU artifact).
- Roadmap variants 1, 2 and 4 (1Cat-vLLM, tensor split) are not measured yet.

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
