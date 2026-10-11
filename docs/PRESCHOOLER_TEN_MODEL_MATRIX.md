# Preschooler 3–4B teacher matrix

Status 2026-10-11: **five of ten private text profiles complete**, not a ten-model quality comparison. All ten pinned snapshots were downloaded and hash-verified on `claude-data`. The candidates are in [`config/preschooler_teacher_roster.json`](../config/preschooler_teacher_roster.json). The [scalar public summary](../reports/preschooler_public_profiles_20261011.json) and [partial offline oracle](../reports/preschooler_partial_oracle_20261011.json) omit raw private items. The associated Preschooler business concept lives in Toddler. New model releases enter by a new roster revision and repeat this protocol; no existing result row is silently replaced.

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

## First three complete private text profiles

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

The Qwen and Phi raw reports have `promotion_eligible: false`. Phi's stronger specialist point estimate and lower GPU-board Wh are worth follow-up, while Qwen's eight-task score is higher. The specialist intervals overlap, and the one-GPU BF16 profile alone cannot establish which base is better for a trained student.

SmolLM3-3B then completed the same private text protocol on GPU 0, pinned revision `a07cc9a04f16550a088caea529712d1d335b0ac1`. Its model card defaults to extended thinking; the first partial run was invalidated because the reference endpoint ignored `enable_thinking=false`. The adapter now passes that setting to the chat template, and the complete replacement run used `/no_think`. The invalid partial run is retained outside Git and excluded from this matrix. The replacement raw report is owner-only, SHA-256 `261cc1a0ce45cb13fd8a169ab35cae48002538be6c2230e966bf52adec7793bf`.

| Measure | Qwen3-4B-Instruct | Phi-4-mini-Instruct | SmolLM3-3B | Scope |
| --- | ---: | ---: | ---: | --- |
| Eight fixed tasks | 0.9792 | 0.7292 | 0.6250 | Same eight tasks |
| Private specialist composite | 0.2500 | 0.3583 | 0.4667 | 44 items, ten lanes; SmolLM3 95% interval 0.3649–0.5684 |
| 256-token output | 25.722 | 26.337 | 33.282 | Tokens/s over wall time including prefill |
| GPU-board energy per answer | 0.091 | 0.062 | 0.076 | Wh on GPU 0; 140 requests each; host energy excluded |
| Reordered-option robustness | 0.8667 → 0.6000 | 0.8667 → 0.6000 | 0.8000 → 0.8000 | 15 small-sample questions |
| Fresh arithmetic holdout | 8/11 | 10/11 | 9/11 | Small sealed sample, not a general ranking |

Phi-3.5-mini-Instruct completed the same pinned private text protocol on physical Ada GPU 0, revision `2fe192450127e6a83f7441aef6e3ca586c338b77`. The canary and 140-request battery passed. Its owner-only raw report SHA-256 is `c2b7d0b2a8c9220c9bdbdb773ade5788570860335aa6eca5fc5a451cdcb0d622`.

| Measure | Qwen3-4B-Instruct | Phi-4-mini-Instruct | SmolLM3-3B | Phi-3.5-mini-Instruct | Scope |
| --- | ---: | ---: | ---: | ---: | --- |
| Eight fixed tasks | 0.9792 | 0.7292 | 0.6250 | 0.6958 | Same eight-task battery |
| Private specialist composite | 0.2500 | 0.3583 | 0.4667 | 0.2333 | Same 44 items; Phi-3.5 95% interval 0.1309–0.3357 |
| 256-token output | 25.722 | 26.337 | 33.282 | 36.244 | Tokens/s over wall time including prefill |
| GPU-board energy per answer | 0.091 | 0.062 | 0.076 | 0.101 | Wh on GPU 0; 140 requests each; host energy excluded |
| Reordered-option robustness | 0.8667 → 0.6000 | 0.8667 → 0.6000 | 0.8000 → 0.8000 | 0.8667 → 0.6667 | 15 small-sample questions |
| Fresh arithmetic holdout | 8/11 | 10/11 | 9/11 | 9/11 | Small sealed sample, not a general ranking |

All four reports are `promotion_eligible: false`. The common item-level export shares **36 identical items across seven specialist lanes**. The best fixed model, SmolLM3, is correct on 20/36; the unattainable oracle over all four is correct on 22/36. Phi-3.5 supplies the two additional correct items, giving an **upper bound** of 5.56 percentage points on this small intersection. A real router may gain less or nothing after model calls, error detection, latency and energy. Four FQ/video items lack the hash-bearing item keys expected by the oracle tool, and four gamer episodes use a separate trace schema; they are excluded from this claim. This is exploratory reuse of one private pack, not a fresh router evaluation. Equal-budget student training has not begun.

Qwen3-4B-Thinking-2507 is **thinking-only** according to the pinned local model card (SHA-256 `6a608c737decd22569edf9609499d142021a65e942999890704a880bc804d163`). Its chat template renders identically for `enable_thinking=false` and `true`, opening `<think>` in both cases. The initial three-prompt canary passed, but the fixed reasoning-off battery could not apply: several early outputs reached 768 tokens before a usable final answer. The partial run was stopped, archived owner-only as `invalid_protocol`, and contributes no score or oracle item. The reference endpoint now rejects a no-thinking template that still opens `<think>`, and the roster blocks this checkpoint from the fixed profile. A later **separate thinking-on profile** must preregister the reasoning-token budget, answer extraction, wall throughput and energy accounting before running this model. Five other roster candidates remain to be evaluated in the current profile; the thinking-only checkpoint remains a distinct candidate, not an inferred zero.

Ministral-3-3B-Instruct needs its pinned `tekken.json` as well as `tokenizer.json`; the snapshot downloader now retains and hash-checks both. Its first 1Cat-vLLM 1.5.1 canary failed before weight loading: vLLM imported `PixtralRotaryEmbedding`, removed in Transformers 5.17. The [upstream vLLM issue](https://github.com/vllm-project/vllm/issues/58755) documents that mismatch. An isolated Transformers 5.16.1 overlay resolved that import, but FP8 weight requantization then failed with `CUDA error: no kernel image is available for execution on the device` on the RTX 4000 Ada. Neither attempt is a quality result. The model card's BF16 dequantization route works in the reference endpoint with Transformers 5.16.1, Accelerate 1.12.0 and MistralCommonBackend 1.12.0. Its text-only canary passed 3/3 diagnostics and the forced 256-token probe ran at 37.345 output tokens/s including prefill. The full private text run is complete; its separate package versions and BF16 conversion remain explicit in every comparison. To reproduce the compatibility overlay in the isolated 1Cat environment, install `transformers==5.16.1` and `accelerate==1.12.0` with `pip --target` into `compat-transformers-5.16.1`; do not replace the shared 5.19 installation.


The complete Ministral3 private report is owner-only, SHA-256 `31a4ae4263bb9ec68769e2b5d523eae0898703207475a5241f8ad3a6864a7926`. It used the same protocol and pack commitments as the other four profiles: eight fixed tasks 0.7375; specialist composite 0.3667 over 44 items (95% interval 0.2664–0.4669); 256-token wall rate 37.609 output tokens/s including prefill; GPU-board energy 0.057 Wh per answer over 140 requests. The option-order probe changed from 1.0000 to 0.6667 over 15 items; the fresh arithmetic sample was 11/11. These small probes do not establish robustness or general superiority. The row is `promotion_eligible: false`.

Across **36 identical specialist items in seven lanes**, the best fixed model remains SmolLM3 at 20/36. The unattainable item oracle over all five reaches 24/36, a ceiling of 11.11 percentage points. Its smallest top subset here is Ministral3 + Phi-3.5 + SmolLM3. A live router must identify those complementary cases without seeing the answer key and must pay for every model call; no live mixture score is claimed. The five-model search yields 26 candidate subsets. The BF16 reference engines are broadly comparable, but the Ministral checkpoint came from FP8 dequantization under Transformers 5.16.1 rather than the 5.12.1 runtime of the other four. A same-runtime hardware replication is needed before attributing speed or energy differences to the model.
