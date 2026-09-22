# LLM benchmark operations

## Active scheduling

| Schedule (Europe/Amsterdam) | Lock | Entry point | Purpose |
|---|---|---|---|
| Every 15 minutes | `/tmp/llm_bench_sweep.lock` | `scripts/benchmarks/run_next_benchmark.py` | Selects one locally present model without a current general-suite result, then runs it. It never downloads a model. |
| At minute 17, 32, 47 and 57 | `/tmp/llm_bench_html.lock` | `scripts/benchmarks/build_dual_v100_html.py` | Rebuilds `reports/dual_v100_nvlink_benchmark.html` from JSON artifacts. |

Both append to `reports/lm_eval_runs/cron_sweep.log`. The current crontab is the operational source of truth; this document deliberately does not install or alter a schedule.

## Runners and artifacts

| Script | Role | Result artifact |
|---|---|---|
| `well_known_suite.py` | GSM8K, BBH, MMLU sample, TruthfulQA and HumanEval; starts an isolated llama.cpp server or attaches to an external vLLM endpoint | `reports/well_known_suite_20260917.json` |
| `run_next_benchmark.py` | Cron-safe queue selector for models already on disk | delegates to `well_known_suite.py` |
| `specialist_suite.py` | Optional Chemistry, Physics, Vision and Video specialist runner | `reports/specialist_suite_20260922.json` |
| `run_qwen38_flash_next_vllm_cascade.py` | Resumable Qwen Flash-Next vLLM hardware/quant cascade, with GPU-idleness preflight | `reports/qwen38_flash_next_vllm_cascade_20260921.json` |
| `run_glm53_reap50_cascade.py`, `run_glm53_hardware_matrix.py` | GLM quant/hardware cascades | GLM cascade/matrix JSON reports |
| `run_hardware_scaling_matrix.py`, `benchmark_additional_gpu_serving.py` | Single V100, dual V100, A4000, Ada and combined-serving measurements | hardware-serving JSON reports |
| `mixture_of_models.py`, `optimize_model_mixture.py`, `materialize_mixture_frontier.py` | Post-hoc and routed mixtures from retained per-question logs | well-known-suite mixture rows |
| `build_dual_v100_html.py` | Dashboard renderer; does no inference | `reports/dual_v100_nvlink_benchmark.html` |

## Specialist suite

The tracked starter packs are `config/benchmark_specialist_holdout_v1.csv` (Chemistry and Physics) and `config/specialist_benchmarks/{vision,video,iq,eq,fq,qq}.csv`. Before a real comparison, copy each selected pack to the ignored private path `data/eval_cache/specialist_packs/<specialist>.csv`, replace/rotate its questions, and use that private file thereafter. The runner automatically prefers private packs. Multiple-choice lanes expose only prompt and options to the model; `answer` stays in the scorer. Vision rows additionally name a local image asset. FQ rows contain robot geometry and target-pose fields; the model returns actuator JSON and the scorer runs the kinematic simulation.

Use a new filename when rotating a pack, keep the prior CSV read-only, and do not commit a future private pack to a public repository. Each result stores the selected filename and SHA-256, so changing a pack cannot silently alter a historical score. The checked-in v1 pack is a starter/audit template, not a permanent blind test after it has been published.

```bash
# Complete specialist suite (default for future general-suite invocations)
python3 scripts/benchmarks/well_known_suite.py MODEL --specialists all

# Time-saving choices
python3 scripts/benchmarks/well_known_suite.py MODEL --specialists chemistry,physics,iq,eq,qq
python3 scripts/benchmarks/well_known_suite.py MODEL --specialists vision,video,fq
python3 scripts/benchmarks/well_known_suite.py MODEL --specialists none
```

For a temporal contamination comparison, attach an independently verifiable model-release date to the result:

```bash
python3 scripts/benchmarks/well_known_suite.py MODEL \
  --model-release-date YYYY-MM-DD --model-release-source https://primary-release-note.example/
```

The dashboard does not treat an absent date, a public/private score gap, or a canary result as proof of contamination. It records those as audit evidence that requires review.

## Tool-access profiles

Every benchmark result belongs to exactly one profile and is displayed in its own table:

| Profile | Permitted capability | What it measures |
|---|---|---|
| `sandbox` | No network and no filesystem tool | Pure model inference/reasoning. |
| `disk` | Search/read only within a purpose-built, read-only benchmark archive | Agentic retrieval, source selection and synthesis. The archive may deliberately contain answer-bearing artifacts when that is the task. |
| `internet_disk` | The same archive plus a logged, allow-listed web retrieval tool | Research-agent performance, not pure model intelligence. |

Never mount the real home directory, git checkout, chat export, wallet store, private answer packs, browser profile or credentials into either tool profile. If answer discovery is intentionally measured, place only synthetic/redacted answer-bearing documents in the benchmark archive and label the score `disk-assisted`. The current raw llama.cpp/vLLM chat endpoint is sandbox-only; a tool-agent wrapper is required before a `disk` or `internet_disk` score may be written.

The expensive three-profile matrix is gated by `config/benchmark_execution_policy.json`. The GLM-5.3 sandbox gate has passed for `glm53-flash-aj-iq2xxs`; the matrix is therefore available, but must still use a dedicated tool-agent wrapper for the two tool profiles. Verify the gate with `python3 scripts/benchmarks/benchmark_phase_gate.py`; a non-zero exit is an intentional block, not a benchmark failure.

### Tensor topology safety

`llama.cpp` tensor split and vLLM tensor parallel are different runtimes. Two historical *long llama.cpp GGUF tensor-split* decode suites left the V100 driver unable to provide a device handle. That is a driver-level failure signature, not proof of a particular kernel or of vLLM TP being faulty. Unattended long GGUF runs therefore use layer split. GLM-5.3 also uses an experimental runtime and automatic CPU offload, so its complete hardware matrix deliberately uses layer split.

The next requested candidate, `deepseek-ai/DeepSeek-V4.1-Flash` under vLLM TP=2 on physical V100s 1/2, is queued only behind `infra/vllm_v100/preflight.py`. At present the preflight rejects it: released FP8/FP4/DSpark weights exceed 64 GiB VRAM and cannot execute natively on V100 SM70. CPU offload increases capacity but does not create FP8/FP4 kernels. It must remain a documented blocked candidate rather than an invented benchmark or t/s result.

Vision is capability-gated: a text-only endpoint gets an explicit `unsupported` result rather than a fabricated score. Native diffusion-video is also separate; the initial video score evaluates a model-authored FFmpeg filtergraph rendered and checked locally within five minutes. It must not be compared to VBench results from a native text-to-video model.
