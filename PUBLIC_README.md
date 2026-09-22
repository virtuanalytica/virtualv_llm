# VirtualV LLM Testsuite

> **Mission:** Measure what a local model actually does on the hardware you
> actually have — not on someone else's H200 pod.
>
> **Approach:** every published score carries its protocol, engine, source
> revision, hardware topology, and raw logs. No score without evidence, no
> OOM presented as a benchmark result, no throughput number without its
> engine/quant/context alongside it.

VirtualV LLM Testsuite is a hardware-aware benchmark harness for evaluating
local LLMs on heterogeneous, non-datacenter GPU pools — the kind most
self-hosters actually have (mixed generations, no NVLink between every card,
PCIe 3.0, a Xeon that isn't the newest thing on the market). It runs GGUF
(llama.cpp) and NVFP4/AWQ (1Cat-vLLM) candidates through one shared quality
protocol, records post-hoc mixture/ensemble results from the same evaluation
samples, and renders everything into one dashboard.

A sibling project: [virtuanalytica/virtualpc](https://github.com/virtuanalytica/virtualpc) —
a self-hosted multi-agent operating system. This testsuite is what decides
which local models virtualpc-style agent rosters actually get to run.

## What's in the box

- **Quality suite** (`scripts/benchmarks/well_known_suite.py`) — GSM8K, BBH
  (6 sub-tasks), MMLU (8 subjects, 5-shot), TruthfulQA (context only), and
  HumanEval (40 problems, isolated subprocess execution). Deterministic
  generation, task-appropriate stop tokens, a frozen protocol ID so results
  under different prompt/scorer versions never silently compare as equal.
- **Hardware-aware cascades** (`scripts/benchmarks/run_*_cascade.py`) —
  resumable, systemd-friendly runners that hold an exclusive GPU lock, stop
  and restart competing local services around each measurement, and record
  every state transition atomically so a crash never loses a proven result.
- **Post-hoc mixture search** (`scripts/benchmarks/optimize_model_mixture.py`,
  `materialize_mixture_frontier.py`, `mixture_of_models.py`) — recombines
  already-measured per-question generations (majority vote for discrete
  tasks, any-pass for HumanEval) to find the best ensemble at a given size,
  without re-serving any model.
- **Dashboard** (`scripts/benchmarks/build_dual_v100_html.py`) — renders every
  current-protocol result, ranked, with failures shown as failures (never a
  fabricated zero score).

## Core principles

1. **Evidence before conclusion.** Every score references its protocol,
   runtime, model source, hardware profile, and raw output.
2. **Quality and speed stay separate.** Tokens/sec is context, not a
   substitute for task accuracy.
3. **Compare like with like.** Protocol version, prompt policy, context
   length, engine, and hardware topology travel with every score.
4. **No fabricated outcomes.** OOM, a missing kernel, a timeout, or an
   unsupported modality get an explicit status, never a fictional number.
5. **Resumable by design.** Long cascades run under systemd, write state
   atomically, and can resume after a crash without losing proven results.
6. **Keep what matters, not what's convenient.** Model weights are disposable
   once a result is durably recorded (hash + raw logs); the evidence, not the
   multi-GB file, is the asset worth keeping.

See [`docs/VIRTUALV_LLM_TESTSUITE_STANDARD.md`](docs/VIRTUALV_LLM_TESTSUITE_STANDARD.md)
for the full governance standard: term definitions, the complete evidence
contract, the operational runbook, and the operator instruction manual.

## Quick start

```bash
git clone https://github.com/virtuanalytica/virtualv_llm.git
cd virtualv_llm
```

Benchmark a model already present locally:

```bash
python3 scripts/benchmarks/well_known_suite.py MODEL_ID \
  --out reports/well_known_suite_latest.json
```

Benchmark an already-running external/local endpoint (hardware fields are
required so results stay comparable later):

```bash
python3 scripts/benchmarks/well_known_suite.py MODEL_ID \
  --external-url http://127.0.0.1:PORT \
  --external-model SERVED_MODEL \
  --physical-gpus 0,1 \
  --topology 'describe your topology' \
  --engine 'engine name' \
  --out reports/well_known_suite_latest.json
```

Rebuild the dashboard after any new result:

```bash
python3 scripts/benchmarks/build_dual_v100_html.py
```

### Prerequisites

- Python 3.12+
- A local llama.cpp build (for GGUF candidates) and/or a 1Cat-vLLM
  environment (for NVFP4/AWQ candidates)
- `lm-eval` on `PATH` for the quality-suite scorers

## Status

This is a live, actively-used internal harness, published so the methodology
and tooling are reusable. Results in `reports/well_known_suite_20260917.json`
reflect one specific host's GPU pool; treat absolute scores as illustrative
of the *method*, not as a general model leaderboard.
