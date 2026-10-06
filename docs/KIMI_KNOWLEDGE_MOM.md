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
code symbols are eligible only for `coding`. JEV supplies an advisory
unvalidated support probability for retrieved evidence; its local backend
uses Qwen and consumes model tokens. It never overrides deterministic facts.
The JEV PIQA calibration is not used for other task families.

The Kimi specialist is invoked only when a working OpenAI-compatible
`MOM_SPECIALIST_URL` is set and the task is explicitly labelled `coding`,
`complex_reasoning`, or `long_context`. Kimi serving is **not validated**:
the installed Colibri v1.11.0 rejects `kimi_k25`, and the installed 1Cat vLLM
has a model registry entry but no verified native INT4 + PMem + SM70 serving
path. Registration, an offline placement plan, and a kernel microbenchmark
are not an end-to-end model result. Run `python3 infra/kimi_k25/preflight.py`
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

The existing `infra/model_serve_configs/mom-live.sh` accepts
`MOM_KNOWLEDGE_BUNDLE`, `MOM_JEV_URL`, `MOM_JEV_API_KEY`, `MOM_SPECIALIST_URL`
and `MOM_EVENTS_OUT`. These are opt-in, so an existing benchmark run is not
changed. The JEV server runs on localhost; use its own `jevserver` key store
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

Compare the same frozen holdout, context, decoding policy and hardware for
baseline MoM and knowledge-backed MoM. Record answer quality, request latency
p50/p95, direct-answer share, JEV calls, specialist calls, and per-engine raw
decode speed separately. An instant answer from a stored fact is a zero-model
token answer; it must not inflate a Kimi or Qwen tokens/s claim. Publish a
model-speed figure only with engine, quant, hardware and raw timing artifact.
