# Preschooler 3–4B teacher matrix

Status 2026-10-11: **candidate acquisition and loader canaries**, not a completed quality comparison. The ten pinned candidates are in [`config/preschooler_teacher_roster.json`](../config/preschooler_teacher_roster.json). The associated Preschooler business concept lives in Toddler. New model releases enter by a new roster revision and repeat this protocol; no existing result row is silently replaced.

## Measurement order

1. Fetch each pinned snapshot to `claude-data` with `preschooler_download_roster.py`. It checks the upstream revision and model-card license, reserves at least 200 GB SSD headroom, and verifies every downloaded weight against upstream LFS SHA-256 where available. EDS1 and EDS2 currently lack sufficient free space for this batch. Do not copy weights or raw private questions into Git.
2. Run an isolated loader and three-prompt diagnostic canary on one **leased, otherwise unused Ada**. Record engine/version, physical GPU index, quant/dtype, context, decode settings, 256-token probe, server logs and failure reason. The canary gates only empty/garbled output and incomplete forced decoding; individual wrong answers are recorded for diagnosis and still proceed to the private quality run. A loader failure is a runtime result, not a zero quality score. MiniCPM3 and Phi-4-mini-flash require source/runtime review before custom model code is enabled.
3. Only after the currently owned benchmark slot is released, run the established private VirtualV endpoint battery on each passing model: eight fixed tasks, ten specialist lanes, three anti-contamination methods, 256-token forced decode, and GPU-board-Wh per answer. Use one protocol version and identical sealed pack hashes. The public full suite remains retired. Vision/video lanes that require a multimodal endpoint get a separate matched profile and are never imputed as zero for text-only endpoints.
4. Keep **two comparable hardware profiles** where feasible: one Ada BF16/FP8-native profile and a common 4-bit quantized profile. Report engine and quant in every row. A cross-engine ranking without a matched profile is exploratory. Do not claim an engine advantage from a model difference.
5. Enumerate all **1,013 subsets** of 2–10 models offline only after all ten have item-level outcomes on the same sealed pack. `preschooler_mixture_frontier.py` reports best fixed, uniform random and unattainable oracle bounds on the *same global item intersection*. It cannot supply a deployed mixture score. For top complementary 2–4-model subsets, run task router, cascade and verifier modes live on fresh sealed items; count every model call, load delay, latency, GPU-board energy and host energy. Reject a complex mixture if its lower confidence bound or cost per accepted answer does not beat the best fixed route.
6. Compare three trainable student bases (Qwen3-4B-Instruct, Phi-4-mini-Instruct, SmolLM3-3B) with equal data, steps, LoRA rank, evaluator and hardware. Distillation candidates come from the broader ten-model pool; the chosen base is recorded as ancestry. Future model releases can contribute new teachers after rights and marginal-information checks.

## llmwatch and machine-readable system statistics

[llmwatch](https://github.com/HindrikStegenga/llmwatch) is an MIT-licensed terminal UI for a vLLM `/metrics` endpoint plus GPU information over SSH. Pinned local checkout: upstream commit `c276a42f8f40b89a979f676eef987a7727fbf40a`. It has been built locally and connected to `mom-live:8030`; local SSH key access works. Its terminal output is **not** a metrics export. `preschooler_system_stats.py` runs it beside each vLLM canary, keeps a bounded ANSI capture, and saves `/metrics`, CPU, RAM, NVMe and GPU board readings to JSONL. This supplies runtime evidence; energy per answer still comes from the benchmark's interval-integrated GPU-board sampler. Whole-system kWh requires a calibrated wall power source and must not be inferred from board watts.

```bash
python3 scripts/benchmarks/preschooler_download_roster.py
ionice -c3 nice -n 15 python3 scripts/benchmarks/preschooler_download_roster.py --download

python3 scripts/benchmarks/preschooler_benchmark.py \
  --alias qwen3-4b-instruct --gpu-index 0 --port 8930 \
  --monitor scripts/benchmarks/preschooler_system_stats.py \
  --out /media/knight2/claude-data/knight1/benchmarks/preschooler/qwen3-4b-instruct
```

The full private option additionally requires `--full --private-runner <active-worktree>/scripts/benchmarks/private_endpoint_suite.py` and the benchmark owner's existing hold and private pack commitments. `preschooler_private_bridge.py` preserves the private scorer but integrates interpolated GPU board power for subsecond answers; the generic two-sample-in-window method would discard them. The runner refuses to mark the private composite complete if the hold, packs, lanes, audit or energy gate fails. The private endpoint runner remains in the active benchmark worktree pending its own integration; no public pack is substituted.

## Current runtime finding

The initial Qwen3-4B-Instruct loader canary on physical GPU 0 failed during CUDA initialization in 1Cat-vLLM 1.5.0. Repeating on 1.5.1 with Triton attention and eager execution also failed with `cudaErrorNoKernelImageForDevice`. The BF16 snapshot itself passed hash verification; **no vLLM answer or throughput score** was produced in those two attempts. The separate Transformers BF16 reference endpoint passed output-integrity canary and forced 256-token decode on GPU 0 at **20.969 wall tokens/s including prefill**; it answered 2/3 diagnostic prompts exactly (the Dutch variant “Goede morgen” differs from the expected “goedemorgen”). This is no private composite. llmwatch captured the compatible `/metrics` endpoint and GPU panel; machine-readable system samples were retained. Neither failed 1Cat attempt affects the active V100 day service.
