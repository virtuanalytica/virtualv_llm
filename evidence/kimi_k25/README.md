# Kimi K2.5 placement inputs

`unsloth_imatrix_expert_counts.jsonl` and `pmem_dax_parallel_read_20261006.json`
are copied unchanged from branch `evidence/pmem-parallel-read-20261006` of
`/home/knight2/repos/1Cat-vLLM-Volta` (2026-10-06). The former comes from the
Unsloth Kimi K2.5 GGUF imatrix calibration, 50 chunks, 60 MoE layers and 384
experts per layer. It is third-party calibration, not a holdout benchmark.

The PMem microbenchmark measured a parallel DAX read on this machine. Its
bandwidth does not predict Kimi decode speed; the prior Q3 GGUF run was much
slower than a bandwidth-only ceiling. `infra/kimi_k25/placement.py` therefore
reports traffic shares and an **offline** placement plan, not token/s.

The source is attributed to `unsloth/Kimi-K2.5-GGUF`; this repository does not
claim ownership of the model weights or the calibration dataset.
