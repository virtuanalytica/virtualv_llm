# LLM benchmark operations

## Active scheduling

| Schedule (Europe/Amsterdam) | Lock | Entry point | Purpose |
|---|---|---|---|
| Disabled since 2026-10-09 | `/tmp/llm_bench_sweep.lock` | `scripts/benchmarks/run_next_benchmark.py` | Historical public full-suite runner; cron is commented out and the runner refuses future public full-suite starts. |
| At minute 17, 32, 47 and 57 | `/tmp/llm_bench_html.lock` | `scripts/benchmarks/build_dual_v100_html.py` | Rebuilds `reports/dual_v100_nvlink_benchmark.html` from JSON artifacts. |

The HTML rebuild still appends to `reports/lm_eval_runs/cron_sweep.log`; the public sweep no longer runs. The current crontab is the operational source of truth.

## Runners and artifacts

| Script | Role | Result artifact |
|---|---|---|
| `well_known_suite.py` | Retired public battery; CLI rejects new runs, while its functions remain importable for historical report rendering | `reports/well_known_suite_20260917.json` (archive) |
| `run_next_benchmark.py` | Retired cron selector; CLI rejects new runs | historical archive only |
| `specialist_suite.py` | Optional Chemistry, Physics, Vision and Video specialist runner | `reports/specialist_suite_20260922.json` |
| `run_qwen38_flash_next_vllm_cascade.py` | Resumable Qwen Flash-Next vLLM hardware/quant cascade, with GPU-idleness preflight | `reports/qwen38_flash_next_vllm_cascade_20260921.json` |
| `run_glm53_reap50_cascade.py`, `run_glm53_hardware_matrix.py` | GLM quant/hardware cascades | GLM cascade/matrix JSON reports |
| `run_hardware_scaling_matrix.py`, `benchmark_additional_gpu_serving.py` | Single V100, dual V100, A4000, Ada and combined-serving measurements | hardware-serving JSON reports |
| `mixture_of_models.py`, `optimize_model_mixture.py`, `materialize_mixture_frontier.py` | Post-hoc and routed mixtures from retained per-question logs | well-known-suite mixture rows |
| `build_dual_v100_html.py` | Dashboard renderer; does no inference | `reports/dual_v100_nvlink_benchmark.html` |

## Private software- en datatoetsen

`software_data_suite.py` meet acht rollen in twee afzonderlijke tabellen.
De private pack staat buiten de checkout; alleen
`reports/software_data_pack_commitment.json` staat in Git. De runner controleert
bestandstoegang, SHA-256 en rolverdeling vóór de eerste modelaanroep. Code
wordt in een netwerkloze, begrensde Docker-container beoordeeld; SQL is
alleen-lezen. Start de pilot pas nadat het huidige benchmarkslot vrij is en
gebruik dezelfde pack en decodeerinstellingen voor alle te vergelijken modellen.

```bash
python3 scripts/benchmarks/software_data_suite.py \
  --pack "$PRIVATE_PACK_PATH" \
  --base http://127.0.0.1:PORT --alias MODEL_ALIAS --model-id MODEL_ID
```

De 16-item pilot is alleen kalibratie. `software_agent_comparison.py` beantwoordt
de afzonderlijke vraag of **Toddler + Teacher + agent op ClaudeClaw** beter is
dan gewone ClaudeClaw-workers. Het script accepteert alleen gepaarde,
onafhankelijk geverifieerde agentresultaten op een verse, elders beheerde
toetsset; zonder die gegevens blijft de uitkomst “nee, nog niet”. Zie
`docs/SOFTWARE_DATA_SPECIALIST_PROTOCOL.md`.

De tweede, cloud-toegestane pack heeft een afzonderlijk commitment in
`reports/software_data_cloud_pack_commitment.json`. Draai eerst minstens twee
**verschillende modelaliassen** op precies die pack: Qwen3.8 Flash-Next op de
gezonde dagdienstendpoint en GLM-5.3-Flash REAP50 IQ4_XS in een apart
benchmarkslot. Die pilot is afgerond. De aansluitende 1Cat-vLLM 1.5.0
target-only TP2-run voltooide ook de volledige private suite ondanks een
3/9-canary; de score mag niet promoveren. Haiku 5.5 medium sloot de
cloud-toegestane pack met 16/16, waarna de runner nieuwe kandidaten op die
pack weigert. De oude 0,348- en 0,3357-composites blijven historisch; de
Flash-Next-NVFP4 TP2-placeholder is geen meting.

Een daaropvolgende **versievergelijking** op het lokale 27B QUASAR NVFP4-
checkpoint gebruikt de officieel gehashte 1Cat-vLLM 1.5.1-wheel in een
afzonderlijke omgeving. Met dezelfde 8K/FP16-configuratie slaagde de
openbare canary 9/9 en voltooide de private 8 taken, tien specialistlanes,
anti-contaminatie-audit, 256-token-doorvoer en GPU-board-energie. Een
tweede arm met E4M3 KV, 8192 prefillbudget en maximaal 16 resident verzoeken
voltooide de volledige private suite en de synthetische B1/B4/B16-proef:
48,35 / 176,11 / 577,59 totale wall-output-tok/s, inclusief prefill.
Houd versie- en configuratie-effecten
gescheiden in de tabel. Geen van deze armen heropent de gesloten
software/data-pack.

De TP2-canary op de geïsoleerde target-only endpoint (zonder private
prompts) wordt zo vastgelegd. Een niet-nul exit is een kwaliteitsignaal,
geen stop voor de volledige meting:

```bash
python3 scripts/benchmarks/probe_1cat_tp2_stability.py \
  --base http://127.0.0.1:18018 \
  --alias qwen38-27b-quasar-nvfp4-tp2-20261009 \
  --out reports/qwen38_1cat_tp2_stability_20261009.json
```

De probe herhaalt drie synthetische canaries elk driemaal op temperatuur 0.
Zijn score is uitsluitend een stabiliteitsbesluit en telt niet mee in een
composite. Gebruik een nieuw run-ID en nieuw outputbestand bij een herproef.
Publiceer na de volledige private meting uitsluitend geaggregeerde cijfers:

```bash
python3 scripts/benchmarks/redact_private_report.py \
  --source /pad/buiten/git/private_benchmark_20261009.json \
  --out reports/private_benchmark_aggregates_20261009.json
python3 scripts/benchmarks/build_dual_v100_html.py
```

De export gebruikt een vaste veldlijst en laat prompts, antwoorden, item-ID's
en per-item-auditsamples weg. Het dashboard leest alleen dit aggregaat;
ruwe logs en de volledige private rapporten blijven buiten de publieke Git.
Een eerdere revisie van PR #44 bevatte toch ruwe private items. Beschouw alle
betrokken packs van 9 oktober als blootgesteld: bewaar de aggregaten als
historisch bewijs, maar roteer de items én het commitment vóór een volgende
blinde vergelijking of promotie. Een branch-rewrite garandeert niet dat een
oude Git-commit of cache verdwenen is. De afzonderlijke software/data-pilot
is niet aantoonbaar door dit incident blootgesteld.
Bij 1Cat-vLLM geeft de OpenAI-respons geen llama.cpp-decodetiming terug.
De 256-tokenmeting gebruikt daarom `usage.completion_tokens` en de gemeten
verzoekduur **inclusief prefill**; het dashboard toont die methode per rij.
Vergelijk haar t/s niet als identieke decodeermaat met een llama.cpp-rij.
Voor Haiku is alleen synthetische Claude-CLI-doorvoer gemeten: B1 had drie
geldige aanvragen, B4 en B16 stuitten bij parallelle oproepen op HTTP 429.
Rapporteer voor die onvolledige batches geen t/s. Claude-CLI-outputtokens
omvatten denktokens; zichtbare outputtokens zijn hooguit een schatting door
de gerapporteerde denktokens af te trekken. CLI-wandklok omvat bovendien
opstart, prefill en netwerkvertraging.
De runner blokkeert Haiku als de twee complete lokale rijen ontbreken, als
beide rijen dezelfde modelalias hebben of als een lokale-only pack wordt
aangeboden. Na een complete Haiku-run is de pack gesloten voor nieuwe runs;
gebruik dan een nieuw commitment. Vergelijk de kwaliteit per item; de CLI
heeft een ander decodeprofiel en geen meetbare lokale GPU-board-energie.
Dit is een handmatig gecontroleerde vervolgqueue, geen tijdstipcron en geen
automatische claim van model- of agentverbetering.

## Specialist suite

The tracked starter packs are `config/benchmark_specialist_holdout_v1.csv` (Chemistry and Physics) and `config/specialist_benchmarks/{vision,video,iq,eq,fq,qq}.csv`. Before a real comparison, copy each selected pack to the ignored private path `data/eval_cache/specialist_packs/<specialist>.csv`, replace/rotate its questions, and use that private file thereafter. The runner automatically prefers private packs. Multiple-choice lanes expose only prompt and options to the model; `answer` stays in the scorer. Vision rows additionally name a local image asset. FQ rows contain robot geometry and target-pose fields; the model returns actuator JSON and the scorer runs the kinematic simulation.

Use a new filename when rotating a pack, keep the prior CSV read-only, and do not commit a future private pack to a public repository. Each result stores the selected filename and SHA-256, so changing a pack cannot silently alter a historical score. The checked-in v1 pack is a starter/audit template, not a permanent blind test after it has been published.

The public full-suite commands are retired. Run the current private specialist
and anti-contamination protocol from the active benchmark roadmap, using its
sealed pack and a shared decode profile. Historical public rows remain readable
but do not contribute to promotion. An independently verifiable model-release
date remains necessary when interpreting temporal contamination evidence.

The dashboard does not treat an absent date, a public/private score gap, or a canary result as proof of contamination. It records those as audit evidence that requires review.

## Tool-access profiles

Every benchmark result belongs to exactly one profile and is displayed in its own table:

| Profile | Permitted capability | What it measures |
|---|---|---|
| `sandbox` | No network and no filesystem tool | Pure model inference/reasoning. |
| `disk` | Search/read only within a purpose-built, read-only benchmark archive | Agentic retrieval, source selection and synthesis. The archive may deliberately contain answer-bearing artifacts when that is the task. |
| `internet_disk` | The same archive plus a logged, allow-listed web retrieval tool | Research-agent performance, not pure model intelligence. |

Never mount the real home directory, git checkout, chat export, wallet store, private answer packs, browser profile or credentials into either tool profile. If answer discovery is intentionally measured, place only synthetic/redacted answer-bearing documents in the benchmark archive and label the score `disk-assisted`. The current raw llama.cpp/vLLM chat endpoint is sandbox-only; a tool-agent wrapper is required before a `disk` or `internet_disk` score may be written.

The expensive three-profile matrix is gated by `config/benchmark_execution_policy.json`. The GLM-5.3 sandbox gate has passed for `glm53-flash-aj-iq2xxs`; the matrix is therefore available, but must still use a dedicated tool-agent wrapper for the two tool profiles. Verify the gate with `python3 scripts/benchmarks/benchmark_phase_gate.py`; a non-zero exit is an intentional block, not a benchmark failure.

### Tensor topology safety

`llama.cpp` tensor split and vLLM tensor parallel are different runtimes. Two historical *long llama.cpp GGUF tensor-split* decode suites left the V100 driver unable to provide a device handle. That is a driver-level failure signature, not proof of a particular kernel or of vLLM TP being faulty. Unattended long GGUF runs therefore use layer split. GLM-5.3 also uses an experimental runtime and automatic CPU offload, so its complete hardware matrix deliberately uses layer split.

The next requested candidate, `deepseek-ai/DeepSeek-V4.1-Flash` under vLLM TP=2 on physical V100s 1/2, is queued only behind `infra/vllm_v100/preflight.py`. At present the preflight rejects it: released FP8/FP4/DSpark weights exceed 64 GiB VRAM and cannot execute natively on V100 SM70. CPU offload increases capacity but does not create FP8/FP4 kernels. It must remain a documented blocked candidate rather than an invented benchmark or t/s result.

Vision is capability-gated: a text-only endpoint gets an explicit `unsupported` result rather than a fabricated score. Native diffusion-video is also separate; the initial video score evaluates a model-authored FFmpeg filtergraph rendered and checked locally within five minutes. It must not be compared to VBench results from a native text-to-video model.
