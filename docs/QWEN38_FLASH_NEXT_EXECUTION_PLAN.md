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

## GLM-5.3-Flash-NVFP4 via 1Cat-vLLM SM70 (re-checked 2026-09-22, correction)

**Correction to an earlier same-day finding in this doc**: the first check
only looked in `vllm/model_executor/models/` (the legacy per-model-file
location) and concluded GLM-5.3 support was entirely absent from the
installed `1cat-vllm==1.5.0` (`/media/knight2/EDS2/envs/1cat-vllm-1.5.0/`).
That was incomplete -- this release ships GLM-5.3-Flash support in a
*different* location, `vllm/models/glm5next/` (registered in
`vllm/model_executor/models/registry.py` lines 129/418-420/645 as
`Glm5NextForCausalLM` / `Glm5NextForConditionalGeneration` / `Glm5NextMTP`),
which the first pass's directory listing missed entirely.

Verified directly by reading the source, not assumed:
- `vllm/models/glm5next/nvidia/attention.py:371-374` has explicit runtime
  SM70 detection (`current_platform.is_device_capability((7, 0))`) selecting
  an `sm70_fp16_indexer` code path.
- `vllm/models/glm5next/nvidia/kda.py` imports `vllm._sm70_ops as sm70_ops`
  and calls SM70-specific custom kernels (`sm70_glm53_fp16_gemv_out`,
  `sm70_glm_kda_fg_b_out`) for the KDA (gated delta attention) path, each
  guarded by `hasattr(torch.ops._C, "sm70_...")` with a generic fallback if
  absent.
- A dedicated `vllm/models/glm5next/sm70/` package exists with `fp8_kv.py`
  (packed E4M3 KV-cache) and `sparse.py` (sparse MLA) -- the two other
  SM70-specific components PR #341's description names -- though neither is
  currently imported from `nvidia/model.py`/`attention.py`, so they read as
  present-but-not-yet-wired-in rather than active.
- **But the compiled kernels themselves are missing**: a `CUDA_VISIBLE_
  DEVICES="" python3 -c "import torch, vllm; hasattr(torch.ops._C,
  'sm70_glm53_fp16_gemv_out')"` check against the installed venv's
  `torch.ops._C` returned `False` for both tested ops. The Python
  integration is real; the C++/CUDA extension backing it was not compiled
  into this wheel, so at runtime every SM70 KDA path would silently fall
  back to the generic kernel (the `hasattr` guards exist precisely for this
  fallback) -- functional, not SM70-optimized.
- Still **unverified**: whether NVFP4 *weight* quantization/loading itself
  (as opposed to the KDA attention kernels checked above) works on SM70 at
  all in this release -- no `nvfp4`/`NVFP4` string appears anywhere under
  `vllm/models/glm5next/`, and the actual quantization backend lives
  elsewhere in `vllm/model_executor/layers/quantization/`. This is the same
  class of question as the earlier corrected W4A16/SM75 claim (see the
  "hallucination correction" note elsewhere in this doc) -- do not assume
  either way without checking that code path specifically before a large
  download.

Net effect: running GLM-5.3-Flash-NVFP4 here would **not** need a build
against a newer `main` (the model class and SM70 runtime-detection scaffold
already ship in the installed 1.5.0); it would run, just without the SM70
KDA kernel speedups until those are compiled in. Whether NVFP4 weight
loading itself works on SM70 is still open.

Separately, PR #341's own test plan targets **8xV100 TP4/PP2** with a
**181GiB** checkpoint -- this host has only 2xV100 (+2 unrelated GPUs) and,
even after clearing every disposable model, a realistic max disk budget well
under 181GiB. The upstream validation doesn't map onto this topology at all;
a working port here (if even possible with 2 V100s instead of 8) would need
real engineering, not just following the PR's own recipe.

Not pursued further tonight -- flagged for an explicit user decision on
whether the engineering cost (verify NVFP4 weight loading on SM70, possibly
compile the missing `sm70_*` kernel extension, then adapt the TP4/PP2 8-GPU
path down to whatever this 4-GPU host can actually do) is worth it before
spending more time on it, rather than assumed.

## Disk budget for the rest of Fase 1/2 (checked 2026-09-22, 72GB free)

Current `/media/knight2/EDS2` usage (94% full, 72GB free): `deepseek-v4-
flash-0731-iq3xxs` 98GB (benchmark in progress), `qwen38-flash-next-ap-
iq4xs` 85GB (proven winner, 91.9% -- keep), `glm53-reap50-iq3m` 68GB
(re-downloaded for the retry, keep until that completes), `qwen38-27b`
16GB (low priority, disposable if needed). `ap-q4km` is already gone from
disk (matches its removal from `CANDIDATES` in commit `ae2e0624`).

The `ap-iq2s` cascade candidate needs ~76GB -- more than the 72GB
currently free. Planned sequencing to avoid a repeat of tonight's
near-miss: once DeepSeek's benchmark finishes, prune its 98GB (matching
the existing disposable-weights pattern -- the result stays in the JSON)
before starting anything else; that alone clears ~170GB, enough for the
GLM retry (already on disk, no download needed) and, once GLM's own
benchmark also completes and its 68GB is pruned, comfortably enough for
`ap-iq2s`'s download without ever dropping near the disk-full mark again.

## DeepSeek-V4-Flash-0731-IQ3_XXS: real CUDA crash on Ada, not a benchmark failure

The all-four-layer run got much further than GLM's attempts -- model
loaded, and the suite completed dozens of GSM8K/HumanEval-style tasks
successfully (task 10361 through 10436, normal ~18 t/s decode) before
crashing at task 11463. The result JSON only recorded `URLError:
<urlopen error [Errno 111] Connection refused>` (the lm-eval client's
view once the server died) -- same class of misleading-generic-error
problem as GLM's `RuntimeError: llama-server exited with code 1`. The
real cause is in the server log:
```
ggml-cuda.cu:108: CUDA error
CUDA error: invalid argument
current device: 3, in function ggml_cuda_kernel_launch at common.cuh:1710
cudaGetLastError()
```
Device 3 in the all-four-layer topology is the RTX 4000 Ada (physical
GPU 3), not either V100. GDB backtrace confirms it's a genuine CUDA
kernel-launch failure inside `ggml_cuda_mul_mat_vec_q` (the fused
mat-vec-times-quant kernel), not an OOM or a Python-side bug. Plausible
cause: a kernel-launch parameter (grid/block dims or the fused
multi-device args struct visible in the backtrace) that's valid on the
other three devices' architectures but invalid on Ada (SM89) specifically
for this IQ3_XXS quant, triggered only by whatever batch shape task 11463
happened to produce -- not reproducible from the first ~11000 tasks.

Not retried yet. Next attempt should try `--profile dual-layer` (V100
pair only, no Ada/A4000 in the mix) to isolate whether this is
Ada-specific or a heterogeneous-4-GPU-topology issue; --fit will offload
more to CPU without Ada/A4000 in the split, so expect a slower run but
one that either reproduces the same crash on the V100s (real IQ3_XXS/
SM70 kernel bug) or completes cleanly (confirms it's Ada/heterogeneous-
specific).

## Naming correction for the Fase 2 mixture command

The approved plan's `optimize_model_mixture.py --require-member` example
(`/home/knight2/.claude/plans/onderzoek-eerst-nog-tussendoor-misty-sky.md`)
names the sixth member `qwen38-flash-next-ap-iq2xxs-v100`. That name is
stale: the cascade candidate was swapped from `ap-q4km` to `ap-iq2s` in
commit `ae2e0624` (`ap-q4km`'s v100 profile was structurally broken, see
that commit's message), so `model_id()` (`run_qwen38_flash_next_gguf_
cascade.py:88-89`) will actually produce `qwen38-flash-next-ap-iq2s-
v100`/`-allfour`. Use that name, not `-iq2xxs-`, when running the Fase 2
mixture command -- the plan's underlying intent (include the newest
low-bit Qwen3.8 quant candidate) is unchanged, only the literal key.

## GLM-5.3-REAP50-IQ3_M: RESOLVED -- root cause was a nested flock, not fit/tensor_split

Full chain of this bug across three misdiagnoses, for anyone reading the
history above: (1) first retry attempt was blamed on a `--fit`/
`--tensor-split` conflict (wrong -- that was reading a stale pre-fix log,
see the correction above); (2) an isolated `llama-fit-params` build
confirmed the fit logic itself was fine; (3) the real cause: `run_glm53_
reap50_cascade.py`'s `benchmark()` wrapped its own `well_known_suite.py`
subprocess call in `flock -n /tmp/v100_exclusive.lock`, but this script is
meant to run under an *outer* `flock /tmp/v100_exclusive.lock python3
run_glm53_reap50_cascade.py` already (per its own runbook usage and the
`glm53_reap50_iq3m_retry.log` invocation) -- a second, non-blocking flock
on the same lock file from a child process can never acquire it while the
parent already holds it. Every retry failed in well under a second with a
generic `subprocess.CalledProcessError...exit status 1`, before
`well_known_suite.py` ever started the server -- no fresh server log, no
result row, which is exactly what made this look like a silent/mysterious
failure across two earlier (wrong) diagnoses. Confirmed by running
`well_known_suite.py` directly (bypassing the cascade script): the model
loaded and began decoding immediately. Fixed in commit `9f5572a5` by
dropping the inner flock, matching `run_qwen38_flash_next_gguf_cascade.py`
(which never wrapped its subprocess in flock for the same reason). The
GLM-5.3-REAP50-IQ3_M v100-profile benchmark itself is running correctly
as of this fix.

**v100-profile result (completed 2026-09-22, ~93 min at 16.6 t/s)**:
gsm8k (flexible-extract) 0.66, mmlu_sample 0.6062, humaneval pass@1
**0.05** (notably low -- a real measured score, not a crash; not
investigated further tonight, flagged in case it recurs on the
allfour-profile run). Metadata (`source_repo`, `quantization`,
`hardware_profile`) was backfilled via `annotate()` after the fact,
since running `well_known_suite.py` directly (bypassing the cascade
wrapper, to avoid the nested-flock bug above) skips that step -- the
cascade's own `row_complete()` check will skip re-running this profile
and also skip `annotate()` for it, so this was done manually once.
allfour-profile still needs to run (via the now-fixed
`run_glm53_reap50_cascade.py --only iq3m`, which will pick up exactly
that remaining profile plus prune the weights once both are complete).

**Update: GLM-5.3-REAP50-IQ3_M cascade is now fully COMPLETE end-to-end.**
A second bug was found and fixed first -- `row_complete()`/`complete_rows()`
checked `isinstance(row.get("gsm8k"), (int, float))` directly, but `gsm8k`
is always a nested dict, never a bare number, so this check was `False` for
every possible result (including the just-completed v100 profile, which the
fixed cascade immediately tried to rerun from scratch). Fixed in commit
`514b874d` using `rank_models.gsm8k_score()`, same as every other script in
this directory. With both fixes in place the cascade ran cleanly: v100
profile correctly recognized as already complete and skipped, allfour
profile ran and passed (`gsm8k=0.68, bbh=0.75, mmlu=0.6, humaneval=0.125,
t/s=22.71, composite=0.539` -- notably better than the v100 profile's
0.496 composite, and almost 40% faster), then `prune_authorized`/`pruned`
fired automatically and removed the 68GB weight directory. Disk: 139GB
free after pruning (was 72GB at the start of tonight's disk-budget
concerns), comfortably enough for the `ap-iq2s` cascade's ~76GB without
needing to prune DeepSeek's weights first.

## DeepSeek-V4-Flash-0731-IQ3_XXS: dual-layer retry CONFIRMS Ada-specific crash

Retried with `--profile dual-layer` (V100 pair only, no Ada/A4000) per the
plan above. Completed cleanly, exit code 0, no crash: gsm8k (flexible)
0.96, bbh 0.875, mmlu_sample 0.8313, **humaneval 0.975**, composite
**0.9103**, 13.9 t/s. This confirms the earlier CUDA kernel-launch failure
(`invalid argument` in `ggml_cuda_mul_mat_vec_q` on device 3 = RTX 4000
Ada) was specific to that device/topology, not a general IQ3_XXS or SM70
problem -- the exact same weights ran ~11000+ tasks further on the V100
pair without incident. Composite (0.9103) is just under the current solo
leader (`qwen38-flash-next-ap-iq4xs`, 0.919) -- a strong new mixture
candidate for the Fase 2 frontier search, not (yet) a new solo leader.
The all-four-layer profile for this model remains unresolved/untried
again; not a priority to chase given dual-layer already gives a complete,
usable result.

## Fase 1 COMPLETE: Qwen3.8-Flash-Next AP-IQ2_S cascade -- NEW solo leader

Downloaded (77GB, `agentionai/Qwen3.8-Flash-Next-AP-GGUF`), both profiles
ran cleanly via `run_qwen38_flash_next_gguf_cascade.py --only ap-iq2s`
(no bugs hit this time -- `complete()` in that script never had the
gsm8k-dict bug the GLM cascade had):
- v100: gsm8k=0.92, bbh=0.875, mmlu=0.875, humaneval=0.975, **composite
  0.9113**, 42.8 t/s.
- allfour: gsm8k=0.92, bbh=0.9167, mmlu=0.8812, humaneval=0.975,
  **composite 0.9232**, 40.8 t/s.

**allfour is the new overall solo leader**, beating the prior champion
(`qwen38-flash-next-ap-iq4xs-v100`, 0.9191) despite using a lower-bit
quant (IQ2_S vs IQ4_XS) at comparable throughput. New top-6 solo ranking:
0.9232 (ap-iq2s-allfour) > 0.9191 (ap-iq4xs-v100) > 0.9177
(ap-iq4xs-allfour) > 0.9175 (ap-q4km-allfour) > 0.9113 (ap-iq2s-v100) >
0.9103 (deepseek-v4-flash-0731-iq3xxs). Weights were not auto-pruned
(the cascade only prunes a losing quant when comparing multiple quants
within one `--only` run; a single new model has nothing to compare
against) -- keep `qwen38-flash-next-ap-iq2s` on disk as the new champion,
same as `ap-iq4xs`. Disk: 62GB free (95% full) after this download; tight
but Fase 2 (mixture frontier search) is CPU-only/post-hoc and needs no
further downloads.

**This closes out Fase 1 of the approved plan** (GLM retry, DeepSeek
retry, new Qwen3.8 cascade candidate -- all three done). Next: Fase 2,
`materialize_mixture_frontier.py` re-run with the full refreshed
candidate pool (now including this new leader, both GLM profiles, and
DeepSeek) to refresh the stale `mixture-optimized-6` frontier, then the
explicit 6-member `--require-member` mixture attempt per the original
plan (using the correct `qwen38-flash-next-ap-iq2s-v100` name, not the
stale `-iq2xxs-` one -- see the naming-correction note above).

## Fase 2: mixture frontier refreshed -- NEW overall record 0.9586

`materialize_mixture_frontier.py` (CPU-only, post-hoc, no GPU lock
needed) ran against the full 31-candidate pool including everything from
tonight (both GLM profiles, DeepSeek, the new Qwen3.8 ap-iq2s champion).
`--size 6` (C(31,6) = 736,281 combinations, sample-cached so each
combination is pure in-memory scoring) took 16 minutes; sizes 2-5 were
fast. New frontier, all improved over the stale one:
- size 2: 0.9243 (`ap-iq2s-allfour` + `deepseek-v4-flash-0731-iq3xxs`)
- size 3: 0.9431 (+ `ap-iq2s-allfour`, `kat-coder-v2.5-dev`, `glm53-flash-aj-iq2xxs`)
- size 4: 0.9502 (`ap-iq2s-v100`, `qwen35-122b-a10b-iq3s`, `kat-coder-v2.5-dev`, `deepseek-v4-flash-reap150b-q2k-adaa4000`)
- size 5: **0.9586** (`ap-iq2s-allfour`, `qwen35-122b-a10b-iq3s`, `granite-4.2-30b`, `devstral-small2-24b-q4`, `deepseek-v4-flash-reap150b-q2k-adaa4000`)
- size 6: **0.9586** (tied with size 5 -- a 6th member, `ap-q4km-allfour` +
  `deepseek-v4-flash-0731-iq3xxs` + `nemotron35-lightning-30b-a3b` +
  `mixtral-8x22b-instruct-q3ks`, added no further gain over the best-5)

**New overall record: 0.9586** (`mixture-optimized-5`/`-6`), up from the
stale 0.9523. Confirms tonight's new Qwen3.8 leader genuinely strengthens
the frontier, not just the solo ranking. Dashboard rebuilt. Standard
selection-set caveat applies (see each row's `selection_warning`):
optimized on the same samples used to measure the members, not yet
validated on a fresh held-out run.

Next: the explicit 6-member `--require-member` attempt from the original
plan, using the corrected `qwen38-flash-next-ap-iq2s-v100` name (this
frontier search already picked `-allfour` for the automatic best-5/6, so
that attempt is now more of a comparison point than a blind shot at a
new record).
