# Qwen3.8 Flash-Next execution plan

This is the durable, chat-independent execution plan.  The active service is
`qwen38-flash-next-gguf-cascade.service`; state and every transition are
written atomically to `reports/qwen38_flash_next_gguf_cascade_20260922.json`.

1. Record the earlier W4A16 candidates as rejected by their *tested*
   compressed-tensors backend on V100 SM70 (it reported SM75 minimum).  This
   is not a blanket 1Cat-vLLM limitation: 1Cat 1.5.0 documents SM70 W4A16
   routes through TurboMind, Marlin, or explicit emulation.  Re-evaluate only
   after the GGUF cascade with a pinned 1Cat runtime, a known backend, and a
   smoke/quality gate; do not infer a t/s result from the old failure.
2. Run `AP-IQ4_XS` through a 2×V100 layer-split suite at 4K context and an
   all-four-GPU layer-split suite at 4K.  GPU telemetry and physical device IDs
   are stored with each row.
3. Run `AP-Q4_K_M` the same way, beginning its 2×V100 run at 2K context due to
   its 61.22-GiB published VRAM requirement.  A smoke-test OOM becomes an
   explicit unsupported profile, never a fabricated score.
4. When both V100 suites are complete, calculate the unweighted mean of GSM8K,
   BBH, MMLU and HumanEval; keep the higher-scoring GGUF and delete only the
   lower-scoring completed quant.  The original result and source hash remain.
5. Build/run MTP only for that winner with an explicitly pinned qwen4exp MTP
   runtime and matching draft sidecars.  Store MTP t/s as a separate row; it
   must not replace normal decode t/s in the model ranking.

## Deferred 1Cat-vLLM 1.5.0 work (after this cascade)

The GGUF service has exclusive ownership of the V100 pair until its two
candidate/profile runs have durable outcome rows and its winner/prune decision
has been recorded.  No 1Cat process may be started before that point.

The next eligible experiment is `RadixArk/Qwen3.8-Flash-Next-NVFP4`,
target-only first, with a pinned source revision, a hash/size check, a smoke
request, and a frozen quality gate before any throughput result is published.
It must use an exact four-V100 TP4 host to reproduce the upstream 8K/512
no-MTP contract (80.732 tok/s) and the matched MTP4 contract (138.26 tok/s).
This machine has only two V100s; GPU0 is an RTX A4000 and GPU3 an RTX 4000
Ada.  Its mixed four-GPU group is therefore not an acceptable substitute for
that contract.  MTP4 remains pending an actual four-V100 allocation and a
matching standalone drafter; it is not a fallback benchmark for GGUF.

`nvidia/Qwen3.6-35B-A3B-NVFP4` is eligible on the local V100 pair via
`infra/model_serve_configs/qwen36-35b-a3b-1cat-vllm.sh`, but already has a
recorded local result; rerun it only for a protocol change or regression gate.

The 1Cat DeepSeek-V4-Flash result cited in release 1.5.0 is PP2 x TP4.  It
requires eight compatible GPU ranks and is not schedulable on this four-GPU
host.  The GLM-5.3 Flash NVFP4 lane is separately blocked here because its
MoE backend requires native FP4 tensor cores.  Flash-Next W4A16 is a deferred
1Cat candidate, not an SM70-wide exclusion: select and record the concrete
TurboMind, Marlin, or emulation backend before attempting it.

References: 1Cat-vLLM 1.5.0 release; Qwen Flash-Next target-only PR #415 and
MTP4 PR #389.  The reported tok/s values are workload-specific gates, never
generic performance claims.

Operational checks:

- `systemctl --user status qwen38-flash-next-gguf-cascade.service`
- `journalctl --user -u qwen38-flash-next-gguf-cascade.service -f`
- `python3 scripts/benchmarks/build_dual_v100_html.py`

The service holds `/tmp/v100_exclusive.lock`, stops/restarts known local chat
services per profile, and refuses to benchmark if a competing compute context
remains.  Stop it safely with `systemctl --user stop
qwen38-flash-next-gguf-cascade.service`; Hugging Face download resumption and
the JSON state make a later restart safe.

## Repository migration (2026-09-22)

The suite's code, config, and evidence have been **copied** (not moved) to a new
dedicated repo: `git@github.com:virtuanalytica/virtualv_llm.git` (private), commit
`783fa5e`. The originals here are unchanged and remain authoritative until a
deliberate cutover: the active `qwen38-flash-next-gguf-cascade.service` still
points at this repo's absolute paths (`WorkingDirectory=`, `ExecStart=` in
`infra/systemd/qwen38-flash-next-gguf-cascade.service`), so **do not** stop/relocate
it mid-run. Cutover sequence for a future session: (1) let the current cascade reach
a durable stopping point (winner selected, pruning done), (2) re-run the
`virtualv_llm` repo's own `build_virtualv_llm_suite_backup.py`/result-sync to pick
up the final rows, (3) repoint the systemd unit's `WorkingDirectory` at the new
repo's checkout path, (4) only then remove the duplicated benchmark code from
`numerai-signals` (never before the new repo's copy is verified running).

## Engine/hardware selection strategy for the next candidates (not yet executed)

Captured from a 2026-09-22 hardware-fit analysis, for use when selecting the next
model+engine+placement combination to test. Disk headroom is currently tight
(81GB free of 1.1TB at last check, 93% used) -- do not start a new multi-GiB model
download without confirming free space first; the GGUF cascade above still has
priority on the V100 pair via `/tmp/v100_exclusive.lock`.

- **This is an interaction problem, not three independent factors.** The best
  model on the wrong engine for this exact heterogeneous topology loses to a
  mediocre model that fits well. Sweep as a combination grid (model x engine x
  placement), not per-factor.
- **Engine candidates for the heterogeneous 4-GPU pool** (16/20/32/32 GiB, three
  generations, PCIe 3.0, no NVLink to CPU, Cascade Lake host):
  - `llama.cpp` (current baseline): per-tensor placement (`--override-tensor` /
    `-ot "exps=CUDA2,CUDA3"`), `--n-cpu-moe`, `--tensor-split`, KV-cache
    quantization, mixed sm_70/86/89 support in one process.
  - `ik_llama.cpp` (fork): purpose-built hybrid CPU+GPU MoE offload kernels,
    reportedly 1.5-2x faster than mainline llama.cpp for DeepSeek-style
    architectures in exactly this CPU+GPU-expert-offload scenario. Verify the
    fork actually builds for sm_70 before relying on it -- architecture support
    tends to lag mainline.
  - `KTransformers`: **not promising on this host.** Its main advantage (AMX
    kernels for expert GEMMs on CPU) requires Sapphire Rapids+; this host's Xeon
    8259CL is Cascade Lake (AVX-512, no AMX), so only a fraction of the claimed
    speedup would apply, and it places attention on a single GPU only.
  - `vLLM`/`SGLang`: already ruled out (sm_80+ and homogeneous TP pools required).
  - **MTP (multi-token prediction)** is a real multiplier (1.5-2x t/s) in hybrid
    setups specifically because verification amortizes over the bandwidth-bound
    CPU-expert path too. Qwen3.8, GLM-5.3, and DeepSeek-V4 each have an MTP head;
    whether the GGUF stack already supports it per model needs to be tested, not
    assumed.
- **Model-architecture fit on this memory hierarchy:** MLA (DeepSeek) minimizes
  KV-cache, freeing more V100 HBM for experts; GLM-5.3's hybrid sparse+linear
  attention has an even smaller cache but a much younger GGUF implementation
  (immature kernels can erase a theoretical advantage). Active-params x
  quant-bits-per-token is the literal CPU-pool bandwidth load. On paper,
  DeepSeek-V4-Flash (13B active, MLA, mature GGUF support) maps best onto this
  hierarchy; GLM-5.3-Flash is the higher-uncertainty candidate.
- **Hardware-specific placement rules for this exact host:** attention+KV on the
  Ada card, helper layers on the A4000, expert tensors on the V100 pair (HBM2
  ~1100 GB/s matters far more for MoE lookups than V100 compute weakness), rest
  on CPU. Everything crosses PCIe 3.0 (~16 GB/s, no NVLink to CPU) -- pipeline
  parallelism does not work here, so per-tensor placement (minimizing cross-PCIe
  traffic per layer) is the right strategy, not tensor/pipeline parallelism.
  **NUMA-bind CPU threads** (`--numa` in llama.cpp / `numactl`): with 2 sockets,
  non-NUMA-aware expert placement can halve effective bandwidth -- flagged as the
  single biggest free win not yet applied on this host. KV at `q8_0` frees HBM
  for more expert tensors; the large page cache means CPU-side weights are read
  from RAM, not re-read from SSD, once warm.
- **Suggested next sweep** (after the current GGUF cascade reaches a stopping
  point and disk space is confirmed): {Qwen3.8-Flash-Next IQ4_XS on 2xV100
  (current baseline winner), DeepSeek-V4-Flash 0731 IQ3 hybrid, GLM-5.3-Flash
  IQ3 hybrid} x {llama.cpp, ik_llama.cpp} x {2 tensor-placement configurations},
  recording TG t/s, PP t/s, and quality per cell, with engine version and exact
  flags pinned per row (a single `-ot` change can reorder the ranking).
- **DeepSeek-V4-Flash 0731 status (checked 2026-09-22, incomplete):** the
  304B-parameter `deepseek-ai/DeepSeek-V4-Flash-0731` repo is confirmed to exist
  publicly with DSpark weights and stronger agentic scores than the prior
  checkpoint, but no published GGUF/quant artifact implementing the specific
  "3-bit hybrid, V100s-for-experts + Ada/A4000-for-attention-KV, rest CPU"
  placement was found before this investigation was interrupted. This placement
  is a **hypothesis to prove, not an assumed ~15-25 tok/s result** -- the
  already-measured local DeepSeek-V4-Flash-REAP-150B Q2_K result (5.85 tok/s
  under heavy CPU offload) is the only real local data point so far. Needs a
  from-scratch GGUF-availability check before attempting.

## DeepSeek-V4-Flash 0731 GGUF availability (checked 2026-09-22, web research only, no download)

Confirmed GGUF quants of `deepseek-ai/DeepSeek-V4-Flash-0731` (284B, 13B active,
MLA) exist from multiple publishers:

- `unsloth/DeepSeek-V4-Flash-0731-GGUF`: UD-IQ3_XXS (~104 GiB), UD-IQ3_S
  (~116 GiB), UD-Q3_K_M/UD-Q3_K_XL (~128 GiB).
- `bullerwins/DeepSeek-V4-Flash-0731-GGUF`: expert-focused quantizations,
  Pareto-pruned by size/KLD.
- `ox-ox/DeepSeek-V4-Flash-0731-gguf-ds4`: IQ2_XXS variant unpacking the
  model's native FP4/FP8 tensors before re-quantizing.

**Disk gate:** at last check (2026-09-22) this host has 81GiB free of 1.1TiB
(93% used) -- even the smallest available quant (104GiB) does not currently
fit. It should fit once the active GGUF cascade prunes its lower-scoring Qwen
quant (frees ~84-88GiB, per `run_qwen38_flash_next_gguf_cascade.py`'s own
prune step) -- do not attempt a DeepSeek-V4-Flash download before that
happens and free space is re-verified.

**Engine gate:** V100 is SM70; native FP8 needs SM89+ (Ada/Hopper/Blackwell),
so V100 would use llama.cpp's software-emulated FP8 path automatically for any
FP8-native tensors -- expect this to be slower than a card with native FP8.
`ik_llama.cpp` (checked: `github.com/RodriMora/ik_llama.cpp` fork) claims
better hybrid CPU/GPU MoE performance via MLA/FlashMLA/fused-MoE/tensor
overrides, but its sm_70 build support was not confirmed by this search --
verify by attempting a local build before relying on it. Also found
`antirez/llama.cpp-deepseek-v4-flash`, a fork with dedicated DeepSeek-V4-Flash
architecture support (mainline llama.cpp's own V4 support is still WIP per
`ggml-org/llama.cpp` discussion #22376) -- worth checking as the actual engine
to use for this architecture rather than assuming mainline or the MoE-offload
fork alone covers it.

Still unproven either way: no local run, no measured tok/s, no quality gate.
This section only establishes that the artifacts and candidate engines exist;
the "V100s for experts / Ada+A4000 for attention+KV / rest CPU" 3-bit hybrid
placement itself remains an untested hypothesis.
