# Kimi K2.5 and knowledge-backed MoM, phase 2

## Current contract

The model is **Moonshot AI Kimi K2.5**. `Kimi-K2.5-Tower` is the local
checkpoint directory; a future EDSQ artifact would be our quantization of
Moonshot's model, not a new model. The checkpoint has 61 text layers, 60 MoE
layers, 384 experts per layer, and native compressed-tensors INT4 group32.

The live MoM proxy accepts a frozen knowledge bundle, explicit task labels,
an optional Kimi specialist endpoint, and an optional local JEV API. Exact,
reviewed fact/rule question-answer pairs can bypass all models. Other
retrieved facts and code symbols are quoted as evidence for the aggregator;
code symbols are eligible only for `coding`/`humaneval`. JEV supplies an
advisory unvalidated support probability for retrieved evidence, or a
fact/rule task classification if there is no evidence; its local backend
uses Qwen and consumes model tokens. It never overrides deterministic facts.
The JEV PIQA calibration is not used for other task families.

The Kimi specialist is invoked only when a working OpenAI-compatible
`MOM_SPECIALIST_URL` is set and the task is explicitly labelled `coding`,
`humaneval`, `complex_reasoning`, or `long_context`. The 2×V100-SXM2 NVLink
board is one 64 GB TP2 execution domain; the Ada members have separate
endpoints and send their proposals concurrently. A tiered Kimi endpoint
would put only hot execution state on TP2 and page expert weights from
DRAM/PMem. Kimi serving is **not validated**: installed Colibri v1.11.0
rejects `kimi_k25`. Installed 1Cat vLLM 1.5.0 has the model class, SM70
TurboMind compressed INT4 and Triton WNA16 MoE code, plus a FusedMoE weight
loader. Its exact native checkpoint dispatch and a 595 GB tier-loader are
unverified. Colibri is the primary tier-serving route; a 1Cat FusedMoE
tier-loader hook is a research route. Registration, kernels, an offline
placement plan and a microbenchmark are not an end-to-end result. Run
`python3 infra/kimi_k25/preflight.py`
to refresh this read-only status.

## Private local build

From the repository root, after indexing the exact committed revision with
the existing GitNexus/LightRAG builder, export reviewed records into a private
directory outside the Git checkout:

```bash
python3 scripts/benchmarks/import_knowledge_sources.py \
  --lightrag-facts /path/to/facts.jsonl \
  --gitnexus-symbols /path/to/symbols.jsonl \
  --repo https://github.com/virtuanalytica/virtualv_llm \
  --commit REVISION --include-prefix scripts/benchmarks/ \
  --out /path/to/private/imported.jsonl
python3 scripts/benchmarks/build_knowledge_bundle.py \
  /path/to/private/imported.jsonl docs/knowledge_public_seed.jsonl \
  --bundle-id virtualv-local-REVISION --scope private \
  --out /path/to/private/bundle.json
```

`import_knowledge_sources.py` marks every imported LightRAG/GitNexus record
private. `knowledge_public_seed.jsonl` contains two small, previously public
Apache-2.0 records with source revisions; the included status is a staging
decision, not a permission to upload the whole private bundle.

For the final Lens condition, derive private Lens rows from this exact bundle:

```bash
python3 scripts/benchmarks/lens_bridge.py /path/to/private/bundle.json \
  --out /path/to/private/lens-rows.json
```

`LensBridge` uses the installed Knitweb Lens `MappingRowsAdapter` and
`Retriever` in process. It rejects stale rows whose bundle digest or content
differs. Its ranked hits are added to the native retrieval context, with
record IDs, source citations and the same coding-task boundary. It does not
publish or query a public Lens service.

The existing `infra/model_serve_configs/mom-live.sh` accepts
`MOM_KNOWLEDGE_BUNDLE`, `MOM_JEV_URL`, `MOM_JEV_API_KEY`, `MOM_SPECIALIST_URL`
`MOM_LENS_ROWS` and `MOM_EVENTS_OUT`. `MOM_KIMI_TP2_URL` activates the
V100-pair layout: Kimi owns both V100s through a separately started TP2
endpoint; Qwen3.8 aggregator runs on Ada 5 with 8192 context/parallel 1,
and Devstral/Qwen3.5/Gemma4 on Ada 0/1/2. The launcher checks the Kimi
`/v1/models` route before starting Ada servers. Use `MOM_SPECIALIST_MODEL`
if the endpoint requires a model ID. Measure the Ada aggregator's fit and
quality at this shorter context before any promotion. These switches are
opt-in, so an existing benchmark run is not
changed. At `start`, the launcher loads the local mode-600
`~/.config/virtualv_llm/mom-jev.env` if present, enabling JEV for every MoM
launched through this script; set `MOM_DISABLE_JEV=1` for a controlled
ablation. The JEV server runs on localhost; use its own `jevserver` key store
and a shell environment with a secret key, never commit the key. The proxy
requires both URL and key and falls back to the aggregator if JEV fails.

Supply `X-MoM-Task: coding` or `?mom_task=coding` to select the codegraph and
specialist. `well_known_suite.py --external-url http://127.0.0.1:8030
--external-model mom-live --mom-task-labels` sends task labels for its direct
requests and lm-eval tasks. A separate raw engine run is required for model
decode tokens/s. The proxy event log can be summarized with
`scripts/benchmarks/summarize_mom_events.py`; it reports request latency and
direct-answer share and deliberately leaves model tokens/s unset.

## Offline expert placement

`infra/kimi_k25/placement.py` consumes the copied per-layer calibration in
`evidence/kimi_k25/`. Supply a measured native expert instance byte count and
explicit tier capacities; the output maps each `(layer, expert)` to GPU,
DRAM, either PMem socket, or disk. The plan preserves layer-specific skew and
balances estimated PMem traffic. It does not move checkpoint bytes or attach
to an engine. Do not infer a decode result from its traffic fractions.

## Public stage for Knitweb, Pulse, Lens and FieldIntelligence

```bash
python3 scripts/benchmarks/prepare_public_knowledge.py \
  /path/to/private/bundle.json --out-dir /path/to/private/public-stage \
  --knitweb-root /media/knight2/EDS2/projects/knitweb
```

The stage contains `public_bundle.json`, `lens_rows.json`,
`field_snapshot.json`, `knitweb_asset.json`, optional unsigned Synaptic
bytecode, and a checksum manifest. The stage is local, unsigned and
`published:false`. Only records explicitly marked `publish_allowed:true`
with HTTPS source, license and no private path/secret marker pass validation.
Review the exact staged bytes, license and source references before signing
or importing into public Knitweb/Pulse, Lens, or ClosedIntelligence. This
workflow does not make a network publication. Keep raw prompts, LightRAG
private facts, GitNexus databases and JEV rules/keys in the private tier.

## Measurement gate

Compare Q3 and Q4 on the same frozen holdout, context, decoding policy and
hardware. For each quant, retain four configurations: raw optimized engine;
that engine in the same MoM with JEV; MoM plus JEV and the frozen
LightRAG/GitNexus/codegraph bundle; finally the same bundle with Lens ranking
enabled. Keep the JEV service, other MoM members and task labels fixed across
conditions. Record answer quality, request latency p50/p95, direct-answer
share, JEV calls, Lens hits, specialist calls, and per-engine raw decode speed
separately. An instant answer from a stored fact is a zero-model
token answer; it must not inflate a Kimi or Qwen tokens/s claim. Publish a
model-speed figure only with engine, quant, hardware and raw timing artifact.
The Unsloth Q3 GGUF is present. A Kimi K2.5 Q4 GGUF was not found on eds1
when this plan was written, and eds1 had about 470 GB free. Confirm capacity
and storage-tier equivalence before acquiring Q4. The 595 GB native INT4
compressed-tensors checkpoint is a separate reference, not the Unsloth Q4
GGUF comparator.
