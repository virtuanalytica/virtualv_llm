# VirtualV LLM Testsuite

Local, hardware-aware LLM benchmark suite: model selection, evaluation, governance
standard, and results for VirtualV's on-prem GPU pool (RTX A4000 + 2×Tesla V100-SXM2
NVLink + RTX 4000 Ada).

## Start here

- `docs/VIRTUALV_LLM_TESTSUITE_STANDARD.md` — the governance standard: principles,
  hardware/runtime profiles, quality suite protocol, evidence/provenance contract,
  operational process, and the operator instruction manual (Annex A).
- `docs/QWEN38_FLASH_NEXT_EXECUTION_PLAN.md` — the durable, chat-independent execution
  plan for the active Qwen3.8 Flash-Next GGUF cascade.
- `docs/LLM_BENCHMARK_OPERATIONS.md`, `docs/MIXTURE_OF_MODELS_4_20260919.md` — earlier
  operational notes and the mixture-of-models design.
- `reports/well_known_suite_20260917.json` — primary evidence file (all benchmark
  result rows). `reports/dual_v100_nvlink_benchmark.html` is the rendered dashboard.
- `output/pdf/` — the business-readable standard+results PDF and a weight-free code
  backup (`VirtualV_llm_suite_v1.zip`).

## Provenance

Migrated from `numerai-signals` (private working repo) on 2026-09-22, at the request
of the repo owner, after `scripts/reporting/build_virtualv_llm_suite_backup.py` had
already produced a vetted, weight-free file manifest. This is a **copy**, not a move:
the originals remain in `numerai-signals` for now. The active benchmark service
(`qwen38-flash-next-gguf-cascade.service`) still runs and writes there; do not point
it at this repo until the cascade completes and a deliberate cutover is done (see
`docs/QWEN38_FLASH_NEXT_EXECUTION_PLAN.md`).

Excluded by design: model weights, private evaluation packs (`data/eval_cache/
specialist_packs`), raw per-model logs (`reports/lm_eval_runs/`, `reports/llama_logs/`),
secrets, and API keys.
