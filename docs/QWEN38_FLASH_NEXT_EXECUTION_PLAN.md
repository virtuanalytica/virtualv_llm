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
