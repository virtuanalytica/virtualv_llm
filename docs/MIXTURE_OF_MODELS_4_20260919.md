# Four-model mixture benchmark — 2026-09-19

The search reused real per-question logs from
`reports/well_known_suite_20260917.json` under evaluation protocol
`v4-mmlu-fewshot-20260918`. No synthetic scores or new model inference were
used. `optimize_model_mixture.py` evaluated all 1,820 four-member combinations
available from 16 eligible models; member order was fixed by solo composite so
even-vote tie-breaking was deterministic.

## Results

| Mixture | GSM8K | HumanEval | MMLU | BBH | Core mean | TruthfulQA | Diagnostic 5-task mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Quality winner: KAT-Coder + Granite + DeepSeek-V4-Flash + Mistral Small 4 | .980 | .975 | .8438 | .8958 | **.923650** | .575 (KAT route) | **.853920** |
| Gemma routed: KAT-Coder + DeepSeek-V4-Flash + Llama 3.1 + Gemma 4 | .940 | .975 | .8688 | .8750 | **.914700** | **.5917** (Gemma route) | **.850093** |
| Best direct four-way vote containing Gemma: KAT-Coder + Granite + DeepSeek-V4-Flash + Gemma 4 | .960 | .975 | .8375 | .8750 | **.911875** | **.5917** (Gemma route) | **.847833** |
| Previous three-model winner: KAT-Coder + Granite + Qwen3.8 | .960 | .975 | .8625 | .8125 | .902500 | not scored | — |

The direct experiment that merely appended Gemma to the previous three-model
winner scored `.887075`, below `.902500`. Gemma therefore does not improve that
mixture as an equal voter. It is useful as a TruthfulQA specialist: routing the
free-text task to Gemma raises TruthfulQA from the best non-Gemma member's
`.5750` to `.5917`, while an odd three-member subgroup handles discrete votes.

The quality-optimized four-model mixture is the new core leader at `.923650`,
an absolute gain of `.021150` over the old three-model ensemble and `.040725`
over the best solo model (`.882925`). It does not contain Gemma; the strongest
Gemma-aware routed alternative reaches `.914700`.

## Reproduce

```bash
python3 scripts/benchmarks/optimize_model_mixture.py --size 4 --top 10
python3 scripts/benchmarks/optimize_model_mixture.py \
  --size 4 --top 10 --require-member gemma4-26b-a4b-q4

python3 scripts/benchmarks/mixture_of_models.py \
  kat-coder-v2.5-dev granite-4.2-30b \
  deepseek-v4-flash-reap150b-q2k-adaa4000 \
  mistral-small4-119b-q4-adaa4000 \
  --name mixture-of-models-4-quality \
  --truthfulqa-specialist kat-coder-v2.5-dev
```

## Interpretation limits

This is post-hoc model selection and scoring on the same sample set. The
ranking is a selection-set estimate, not a held-out generalization claim. A
fresh held-out run is required before production promotion. Runtime is also a
sequential ensemble: quality improves, but latency/compute is approximately
the sum of its member calls. TruthfulQA remains outside the official four-task
leaderboard composite; the five-task mean above is diagnostic only.
