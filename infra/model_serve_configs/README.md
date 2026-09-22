# Model serve configs

Archived serving configurations for models that reached the top-5 by composite
score in the dual-V100 well_known_suite sweep (`reports/well_known_suite_20260917.json`,
dashboard: `reports/dual_v100_nvlink_benchmark.html`). Weights are NOT kept locally
(disk-management rule: delete after recording a result) — each script here has
the HF repo id, quantization, a download command, and the exact serve flags used
during benchmarking, so a virtualpc consultant can reinstall and run any of these
if chosen for production.

This is an append-only archive: a script stays here even if its model later drops
out of the top-5. Ranking snapshot after the completed cascade and BBH log
normalization (2026-09-21, composite =
unweighted mean of gsm8k/humaneval/mmlu/bbh, see `scripts/benchmarks/rank_models.py`):

| Rank | Model | Composite | Engine | Script |
|---|---|---|---|---|
| 1 | Qwen3.5-122B-A10B | 0.899 | llama.cpp GGUF | `qwen35-122b-a10b-iq3s.sh` |
| 2 | KAT-Coder-V2.5-Dev | 0.883 | llama.cpp GGUF | `kat-coder-v2.5-dev.sh` |
| 3 | Llama-3-70B-Instruct | 0.860 | llama.cpp GGUF | reinstall command in cascade JSON |
| 4 | Granite-4.2-30B | 0.858 | llama.cpp GGUF | `granite-4.2-30b.sh` |
| 5 | Qwen3.8-27B (reference) | 0.851 | llama.cpp GGUF | `qwen38-27b.sh` |

The post-hoc four-model mixture scores 0.924, above every individual model,
but is a router/ensemble result rather than one separately installable weight
file. The disk cascade therefore retains Qwen3.5-122B-A10B as the single large
individual winner; the production Qwen3.8 reference remains separately
protected.

Re-check this table after every new candidate lands; add a new script for any
model that newly enters the top-5, don't delete scripts for ones that drop out.
