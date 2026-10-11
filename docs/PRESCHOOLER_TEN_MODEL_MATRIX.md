# Preschooler 3–4B teacher matrix

Status 2026-10-11: **two of ten private text profiles complete**, not a ten-model quality comparison. All ten pinned snapshots were downloaded and hash-verified on `claude-data`. The candidates are in [`config/preschooler_teacher_roster.json`](../config/preschooler_teacher_roster.json). The associated Preschooler business concept lives in Toddler. New model releases enter by a new roster revision and repeat this protocol; no existing result row is silently replaced.

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

The initial Qwen3-4B-Instruct loader canary on physical GPU 0 failed during CUDA initialization in 1Cat-vLLM 1.5.0. Repeating on 1.5.1 with Triton attention and eager execution also failed with `cudaErrorNoKernelImageForDevice`. The BF16 snapshot itself passed hash verification; **no vLLM answer or throughput score** was produced in those two attempts. The separate Transformers BF16 reference endpoint passed output-integrity canary and forced 256-token decode on GPU 0 at **20.969 wall tokens/s including prefill**; it answered 2/3 diagnostic prompts exactly (the Dutch variant “Goede morgen” differs from the expected “goedemorgen”). llmwatch captured the compatible `/metrics` endpoint and GPU panel; machine-readable system samples were retained. Neither failed 1Cat attempt affects the active V100 day service.

## First two complete private text profiles

The isolated Transformers BF16 endpoint on physical Ada GPU 0 completed the active `v3-private-specialist-anti-eight-gamedev-gamer-20261009` protocol on 11 October 2026. Model weights are pinned at `cdbee75f17c01a7cc42f958dc650907174af0554`; decoding used temperature 0, reasoning off and low effort. The raw private report remains outside Git, SHA-256 `8efe7b42cbcbe6f20084a5de1ec2b3ec286187c901285407e40a13b9e08d1cac`. It recorded:

| Measure | Result | Scope |
| --- | ---: | --- |
| Eight fixed tasks | 0.9792 mean | Eight scored tasks, not the specialist composite |
| Private specialist composite | 0.25; 95% interval 0.1485–0.3515 | 44 scored items across ten text-output lanes; image vision was unsupported and excluded |
| 256-token throughput probe | 25.722 output tokens/s | Transformers wall time including prefill; separate canary wall rate 25.581 tokens/s |
| GPU board energy | 0.091 Wh/answer; 12.737 Wh in 140 requests | Physical GPU 0 only; excludes CPU, DRAM, PMem, SSD and PSU |
| Verbatim canary recall | 0/10 | Small historical recall probe, not proof of no contamination |
| Reordered-option robustness | 0.8667 original → 0.6000 reordered | 15 questions; 26.67 percentage-point drop needs investigation |
| Fresh post-cutoff arithmetic | 8/11 = 0.7273 | Private local holdout, small sample |

The raw report's `timing_source: server decode` label is incorrect for this Transformers adapter: its `predicted_per_second` value was calculated around the entire `model.generate` call, including prefill. The numeric 25.722 t/s remains a valid request-wall rate. The adapter now exports an explicitly named wall rate and lets the private runner use its wall-clock fallback. This correction does not re-score quality or rewrite private evidence.

The Qwen result is marked `promotion_eligible: false`. Its strong eight-task mean and weak specialist performance measure different work; neither predicts a trained Preschooler or establishes superiority over another candidate. The post-cutoff and option-order samples are too small for a robust generalization claim. The video lane in this protocol asks for an FFmpeg filtergraph from text and scores the rendered artifact; it is not a native visual-generation test.

Phi-4-mini-Instruct then completed the **same** private protocol on one physical Ada GPU 0 with a Transformers BF16 endpoint, pinned revision `cfbefacb99257ffa30c83adab238a50856ac3083`, temperature 0, reasoning off and low effort. The specialist and holdout pack SHA-256 values match the Qwen run; its private raw report remains outside Git, SHA-256 `e8801d1b815099a7c5b3ee361978b4148d7994b6c3727a8b653333a19061baa1`.

| Measure | Qwen3-4B-Instruct | Phi-4-mini-Instruct | Scope |
| --- | ---: | ---: | --- |
| Eight fixed tasks | 0.9792 | 0.7292 | Same eight-task battery; this is a mean across heterogeneous tasks |
| Private specialist composite | 0.25 | 0.3583 | Same 44 text items and ten lanes; respective 95% intervals 0.1485–0.3515 and 0.2585–0.4581 overlap |
| 256-token throughput | 25.722 | 26.337 | Output tokens/s over request wall time including prefill |
| GPU-board energy per answer | 0.091 Wh | 0.062 Wh | 140 requests each on GPU 0; host power excluded |
| Verbatim canary recall | 0/10 | 0/10 | Small historical probe, no proof of general non-contamination |
| Reordered-option robustness | 0.8667 → 0.6000 | 0.8667 → 0.6000 | 15 items; option-order sensitivity remains |
| Fresh arithmetic holdout | 8/11 | 10/11 | Small sealed sample; not a general quality ranking |

Both raw reports have `promotion_eligible: false`. Phi's stronger specialist point estimate and lower GPU-board Wh are worth follow-up, while Qwen's eight-task score is higher. The specialist intervals overlap, the models have different architectures and tokenizer behavior, and the one-GPU BF16 profile alone cannot establish which base is better for a trained student. The remaining eight profiles and fresh paired holdouts are still required before mixture or training decisions.
