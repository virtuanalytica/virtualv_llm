# VirtualV LLM model test roadmap

Status date: 2026-10-09 (older dated sections remain historical). This document separates measured local evidence from
upstream reference numbers and from untested hypotheses.

**Integriteitsstatus PR #44:** een eerdere publieke PR-revisie bevatte ruwe
items uit de private acht-takentoets, specialistensuite en anti-contaminatie-audit.
De huidige revisie publiceert alleen aggregaten, maar de oude Git-commit kan
nog bereikbaar zijn. De op 9 oktober gebruikte packs zijn daarom uitsluitend
historische meetdata en mogen nooit opnieuw als blinde holdout of promotietoets
worden gebruikt. Roteer vragen, antwoorden en pack-hashes vóór de volgende
vergelijking. De afzonderlijke software/data-pilotpack is hierdoor niet als
gecompromitteerd vastgesteld.

## 2026-10-09 — softwaremakers, dataspecialisten en Haiku 5.5-baseline

De nieuwe private pilot `software-data-private-v1-20261009` heeft aparte
software- en datatabellen. Coder, reviewer, architect, debugger, data engineer,
data analyst, data architect en data steward krijgen elk twee verzegelde
startitems. De lokale pack blijft buiten Git en buiten de cloud; haar
SHA-256 en rolverdeling zijn gecommit. De kleine pack kalibreert alleen de
scorer en de meetketen. Er is nog geen modelslagingspercentage en geen
verbeterbewijs voor Toddler + Teacher + agent op ClaudeClaw.

**Uitvoeringsrij op 9 oktober, na de GLM Q4→Q3-keten en de gevalideerde
dagdiensthandoff:**

| Stap | Status | Uitvoer en poort |
|---|---|---|
| 0. GLM Q4→Q3 en handoff | **Voltooid 04:37 CEST** | Beide private rapporten zijn complete; de handoff valideerde ze en herstelde de gezonde dagdienst. Q4 mat 2,70 tok/s en 0,782 **GPU-board**-Wh/antwoord. Q3 mat 0,589 GPU-board-Wh/antwoord. Beide energiecijfers zijn uit ruwe samples geïnterpoleerd. |
| 1. Qwen3.8 Flash-Next AP-IQ2_S | **Voltooid** | De 16-item pilot gaf 8/8 software en 7/8 data op de cloud-toegestane pack. Deze aantallen zijn alleen kalibratie; per rol zijn er twee items. |
| 2. GLM-5.3-Flash REAP50 IQ4_XS | **Voltooid** | Dezelfde 16-item pack gaf 4/8 software en 8/8 data. De geïsoleerde runtime gebruikte V100 ×2 + Ada ×2, 47/47 lagen op GPU, `reasoning_effort=low`; dit zijn pilotresultaten, geen promotiegrond. |
| 3. Qwen3.8 1Cat-vLLM TP2 1.5.0 | **Voltooid; promotie uitgesloten** | Target-only 27B QUASAR NVFP4 op V100 ×2 en 8K: canary 3/9 (rekenen 3/3, JSON 0/3, code 0/3 met herhaallussen), private acht taken 0,7375, tien specialistlanes 0,2333, 256-tokenproef 47,84 wall-output-tok/s inclusief prefill en 0,517 GPU-board-Wh/antwoord. De aparte software/data-pilot gaf 2/8 en 4/8. Alle 140 private verzoeken zijn beantwoord; de dagdienst is hersteld. |
| 4. Claude Haiku 5.5, medium | **Voltooid; cloudpack gesloten** | Haiku gaf 8/8 software en 8/8 data op dezelfde cloud-toegestane 16-item pilotpack. Dit is een kleine kalibratie, geen agentbewijs. De CLI gebruikte adaptive thinking, effort medium, standaardtemperature en geen lokaal afdwingbare max-tokenlimiet. |
| 5. 1Cat-vLLM 1.5.1, identieke 1.5.0-configuratie | **Voltooid; dagdienst hersteld** | De officiële wheel met gecontroleerde SHA-256 draaide apart op hetzelfde checkpoint en V100-paar. De canary steeg van 3/9 naar 9/9. De private acht taken stegen van 0,7375 naar 0,8542 en de tien specialistlanes van 0,2333 naar 0,4417; de 256-tokenproef steeg van 47,84 naar 52,71 wall-output-tok/s, inclusief prefill. GPU-board-Wh per verzoek daalde van 0,517 naar 0,045, mede doordat herhaallussen wegvielen. Alle tien lanes en audits zijn compleet; twee dynamische specialistverzoeken minder dan bij 1.5.0. De gesloten software/data-pack wordt niet heropend. Dit is nog geen promotiebesluit. |
| 6. 1Cat-vLLM 1.5.1, verbeterde TP2-capaciteit | **Voltooid; dagdienst hersteld** | Dezelfde 27B-gewichten met E4M3 KV, prefillbudget 8192, block-size 2048, Mamba-block 8192 en maximaal 16 gelijktijdige verzoeken gaven 9/9 canary, 0,8542 op acht taken en 0,4417 op tien specialistlanes. De 256-tokenproef gaf 48,65 wall-output-tok/s en 0,043 GPU-board-Wh/verzoek. De aparte openbare capaciteitstest gaf B1 **48,35**, B4 **176,11** en B16 **577,59** totale output-tok/s, elk met twee volledige meetgolven na warmup. De upstream releaseprofile is op TP4 geijkt; `profile_hardware` blijft op dit TP2-systeem uit, terwijl E4M3- en Q8000-paden wel geladen zijn. Dit is een aparte configuratie, geen extra versie-effect. |

Een afzonderlijke **openbare synthetische Haiku-doorvoerproef** mat bij B1
drie geldige aanvragen: 157,0 outputtokens/s over de volledige CLI-wandklok,
inclusief denktokens, en naar schatting 53,6 zichtbare tokens/s. Dit is geen
server-decode-meting en geen score op de gesloten pilotpack. B4 gaf 6/8 geldige
aanvragen en B16 6/32; een aparte parallelle controle bevestigde HTTP 429.
Daarom zijn de B4/B16-doorvoercellen in het dashboard leeg en is er geen
betrouwbare capaciteitsschatting op die batchgroottes.

De uitvoercommando's en het commitment staan in
`docs/SOFTWARE_DATA_SPECIALIST_PROTOCOL.md`. De pilot heeft slechts twee items
per rol; geen van deze scores mag een promotie of bewijs van superioriteit van
Toddler + Teacher + agent op ClaudeClaw opleveren. De Claude CLI kan
temperature en het lokale max-tokenbudget niet identiek afdwingen. Label
Haiku daarom als een afzonderlijk runtimeprofiel; cloud-GPU-energie is
onbekend en wordt nooit als lokale GPU-board-Wh voorgesteld.

De canary is diagnostiek en mag de volledige TP2-benchmark niet afkappen.
De 1Cat-proef hergebruikt de historische `qwen38-1cat-vllm-target` score
0,348 **niet**: de analyse van 18 september wees op reproduceerbare
temperatuur-0-herhaallussen in NVFP4/TP2, ook met een andere KV-cache-dtype.
De rij `qwen38-27b-quasar-nvfp4-1cat-tp2` van 8 oktober heeft eveneens een
onvoldoende 0,3357-composite en is een aparte historische meting. De rijen
`qwen38-27b-quasar-nvfp4-1cat-tp2` en
`qwen38-flash-next-nvfp4-1cat-tp2` mogen niet als nieuwe geslaagde TP2-run
worden gekopieerd. De Flash-Next NVFP4-snapshot staat momenteel niet lokaal;
de eerdere TP2-load liep vast op geheugen en de upstream Flash-Next-prestatie
is op **vier** V100's met TP4 gemeten, niet op dit V100-paar. Een nieuwe
Flash-Next TP2-proef krijgt daarom pas een slot na verifieerbare
geheugen-/runtimepreflight. De 27B-herproef is de uitvoerbare TP2-kandidaat.

De latere agentproef gebruikt een andere, verse lokale pack. De
promptverbeteraar mag alleen oefenitems zien. Voor een positieve claim over
de volledige Toddler + Teacher + agent op ClaudeClaw-keten is een
onafhankelijke, gepaarde agentproef van minstens 73 taken per softwaremaker
vereist, met de poorten in `docs/SOFTWARE_DATA_SPECIALIST_PROTOCOL.md`.

De historische 115 reviewbevindingen zijn oorzaaklabels zonder gepaarde
controlegroep. Ze beantwoorden de vraag “is beter op softwaretaken bewezen?”
niet; de actieve uitkomst blijft **nee, nog niet**.

## 2026-10-09 besluit — private vergelijkingen en Toddler-router

De publieke GSM8K/BBH/MMLU/HumanEval-full-suite wordt niet meer gestart.
Bestaande scores zijn uitsluitend historisch en tellen niet mee voor promotie.
Het actieve lokale protocol `v3-private-specialist-anti-eight-gamedev-gamer-20261009`
meet acht vaste taken, tien private specialistlanes, drie anti-contaminatiemethoden,
geforceerde decode over 256 tokens en GPU-board-Wh per antwoord. Vergelijkbare
quants gebruiken dezelfde prompts, decodeerinstellingen, meeteenheden en
fysieke GPU-mapping. Ontbrekende lanes of audits maken een resultaat partieel.
De specialistpacks, gamedev-cases en post-cutoff-holdout waren vóór modelrespons
op SHA-256 vastgelegd. De inhoud van de gebruikte private packs kwam later in
een publieke PR-revisie terecht; zie de integriteitsstatus hierboven.
De gamer-v1-lane bevat vier interactieve tekstschermepisodes, geen bewijs van
visueel spelbegrip of PlayStation-besturing. Vision heeft een afzonderlijke,
verzegelde 12-beeldenset.

**Actieve volgorde:** voltooi GLM-5.3-Flash UD-Q4_K_XL en daarna UD-Q3_K_XL
op hetzelfde V100-paar en protocol. De Q3-shards op `claude-data` zijn tegen
het vastgepinde upstream manifest geverifieerd. De Ada-visionruns en de
Haiku-5.5 acht-takenreferentie zijn voltooid. Valideer alle GLM-, vision- en
Haiku-resultaten voordat de benchmarkhold wordt vrijgegeven; de handoff
`scripts/benchmarks/glm53_private_handoff.py` herstelt pas daarna de dagdienst.
Daarna berekenen we per vergelijkbare set het orakelrouter-plafond, toetsen we
snelle lokale kandidaten op verse private items en bouwen we pas bij de
vastgelegde promotiegrens een Toddler-mixture. De numerieke poorten staan in
`docs/MOM_PROMOTION_PROTOCOL_20261009.md` in de meetwerkboom.

**K3 afgesloten als actieve kandidaat:** de zes-GPU UD-IQ2_XXS-run is bewust
gestopt. Acht taken en een 256-tokenmeting zijn bewaard in
`reports/k3_sixgpu_partial_status_20261009.json`; de audit en specialistensuite
zijn onvolledig. Er is geen promotiebesluit. De eerdere herstelketen,
Q2_K_XL-download, officiële MXFP4-snapshot, Colibrì-run en K3-shardverdeling
zijn geannuleerd. Geen K3-download of -benchmark wordt automatisch hervat.
Ruwe logs, hashes en historische vergelijkingen blijven leesbaar. Een nieuw
K3-experiment vereist een nieuw besluit met expliciet protocol en budget.

**Opslagbeleid voor toekomstige modellen:** EDS1 en `claude-data` zitten op
een PCIe-root; EDS2 gebruikt een andere root en telt mee bij I/O-planning.
Verdeel hele shards uitsluitend op basis van actuele vrije ruimte, geverifieerde
hashes en gemeten doorvoer; reserveer minimaal 200 GB vrij per SSD. VRAM,
DRAM, PMem100 en NVMe zijn aparte geheugenniveaus; een cacheclaim vraagt
metingen van werkelijke hitrate en latency. Dit is beleid, geen reservering
voor K3.

**BridgeBench:** blijft buiten de actieve suite; er is geen controleerbare
open-source release met bewezen anti-contaminatiemaatregelen en een lokale
looptijd onder twee uur. De eerdere onderzoeksafweging is historisch.

## 2026-10-07 update — EDSQ-Volta takeover cycle (fieldintelligence)


**Completed 5–7 Oct.** Tier-balanced MoE judges live and measured: K2.5 1.24,
DeepSeek-V4-Flash-0731 2.30, GLM-5.3-Flash 2.41–6.20, K2.7-Code 0.68 tok/s
(E-config flags, 6-GPU attention; full evidence in
`fieldintelligence/EDSQ-Volta` → `evidence/`). MoM v2 (K2.5 judge)
19m37s/question → MoM v3 (DSv4 judge) 5m21s at equal verdict quality
(portfolio 8.2 % vs wrong drafts on all three proposers). PMem 100 Optane
4×512 G tier live (App Direct/fsdax/DAX): 10.1–10.5 GB/s per mount local
socket, remote-socket 0.4–1.0 → NUMA-local placement mandatory. 6.4 TB NVMe
model home (`eds1`, mount by-UUID). Day/night profile live via
`~/bin/vllm-profile` (cron 07:30/19:00): day = 1Cat TP2 NVFP4 target
46.6–49.0 t/s (meets the 50 t/s goal), night = judges + queue worker.

**Baseline discrepancy to resolve:** the DSv4-0731 re-test measured 2.30 t/s
(UD-IQ4_XS, E-config) against the Sept row of 13.89 t/s (UD-IQ3_XXS), and
GLM-5.3-Flash 2.41–6.20 t/s (UD-Q4) against 21.49 t/s (AJ-IQ2_XXS). Different
quant, flags and tier state — re-test under the Sept configuration before
treating either number as canonical.

**Backlog (ordered):**
1. P1 — RESOLVED 2026-10-07. The HTTP 404 was not the gsm8k step (GSM8K
   and TruthfulQA had finished). The batteries were started with
   `--external-url http://host:port/v1`, and the suite appends `/v1/...`
   itself, so its first direct request went to `/v1/v1/chat/completions`;
   llama-server answers 200 on `/v1/chat/completions` and 404 on the doubled
   path (checked on the running :8011 server). `well_known_suite.py` now
   refuses such a URL at start. Still to do: re-run the GLM-5.3-Flash and
   DSv4-Flash-0731 batteries with the server root; their rows stay failed
   until then.
2. K2.7-Code UD-Q3_K_XL (432 G) downloaded — serve + compare vs Q4_K_XL
   (0.43–0.68 t/s baseline; Q3 fits the page cache without a PMem tail).
   Test chain pattern: `evidence/microbench/kimi_k27_code_tier_shard_20261006.md`.
3. K3 UD-IQ2_XXS: historical, partial six-GPU probe. The old auto-chain
   and all follow-up downloads were cancelled on 2026-10-09; see the current
   decision above and the retained partial-status report.
4. MoM v3.1 day lane: judges are night-profile by design; evaluate the
   1Cat NVFP4 lane for scored daytime MoM after item 7's quality gate.
5. GLM-5.3-Flash REAP50 GGUF conversion bug (missing
   `glm5-next.attention.indexer.kpool` tensor) — report upstream; the
   official unsloth conversion works.
6. Colibri source checkout (`coli build` needs a clone) + qwen38 A/B
   against llama.cpp — decides whether the CPU expert tier switches engines.
7. NVFP4-TP2 repetition-loop instability: quality gate (8-task battery) on
   the day lane before scored daytime outputs are trusted.
8. JEV SystemOne adapter for well_known_suite: stateful protocol
   (`POST /v1/systemone`, state + questions), key `virtualv-mom-local`
   (hash-only in `~/.config/toddler-jev/keys.json`) — needs a small
   adapter, not a suite flag.

## 2026-09-24 decisions

- MiMo-V2.6-Pro q4 (MXFP4) and q3 (BPW3.0/BPW3.5) are not planned. They do
  not fit this disk or these V100s, and they are the wrong next step.
- A usable rate under 3 tokens/second is not acceptable on this machine.
  Mixture throughput is the sequential rate, `1 / sum(1/member_t/s)`, not
  the slowest member's own rate.
- The live MiMo-V2.6-Pro BPW2.5 suite stays. Its measured decode is above
  that floor. Mixture rows already use the sequential rate, and the search
  keeps combinations at or above 3 tok/s. The next work is the ordered queue
  below, after the disk gate, not another MiMo quant.

## Current local baselines

| Goal | Local baseline | Result |
|---|---|---|
| Fast, balanced model | Qwen3.8 Flash-Next AP-IQ2_S, all four GPUs | 40.79 tok/s; public composite 0.9232 |
| Fastest measured Qwen row | Qwen3.8 Flash-Next AP-IQ2_S, 2x V100 | 42.79 tok/s; public composite 0.9113 |
| Quality candidate | DeepSeek-V4-Flash-0731 UD-IQ3_XXS | 13.89 tok/s; public composite 0.9103 |
| Agentic candidate | GLM-5.3-Flash AJ-IQ2_XXS | 21.49 tok/s; reasoning-effort fix still requires a clean re-test |

These values are read from `reports/well_known_suite_20260917.json`. Different
engines, contexts and topologies remain separate rows.

## Disk gate before the queue

Checked 2026-09-24 11:24 CEST. `/media/knight2/EDS2/models/llm` contains only
`mimo-v26-pro-bpw2.5` (319,743,531,488 bytes, suite still running) and
`qwen38-27b` (live chat on GPU 3, port 8011). `df` showed about 74 GB free.
A download starts only when free space exceeds the artifact size plus the
24 GB reserve, so nothing of about 50 GB or larger fits while the MiMo shards
stay. Do not delete those shards until `mimo-v26-pro-bpw25.service` has exited
and `/tmp/v100_exclusive.lock` is free. The Qwen chat weights stay.

## Ordered experiment queue

1. **GLM AJ-IQ2_XXS re-test, after a re-download.** The weights are not on
   disk. Recorded size is 87,346,006,560 bytes (two shards in
   `scripts/benchmarks/run_glm53_hardware_matrix.py`, repo
   `aj9o9/GLM-5.3-Flash-GGUF`, revision
   `07c62fcdeaf1c05d22bd123c3da8058a1b1e63e2`). Re-run only after the disk gate
   and the V100 lock both pass. Preserve the old result as superseded evidence
   and use a new run identifier. `glm53-reap50-retest.service` is inactive, not
   waiting on the lock. Its IQ3_M weights (72,132,392,352 bytes) are also gone,
   so starting that unit would download before it can score. Leave it stopped
   until the same gate passes.
2. **DeepSeek-V4-Flash-0731 placement matrix, after a re-download.** The
   UD-IQ3_XXS directory is gone. The benchmark registry describes four shards,
   about 104 GiB. Same disk and lock gates. Once the weights are back, use the
   measured 13.89 tok/s row as baseline. Test V100 layer split and the supported
   all-four layer split with identical context and prompts. Treat “V100 experts,
   RTX attention/KV, CPU remainder” as a hypothesis until a runtime exposes and
   verifies tensor-class placement; never infer it from aggregate VRAM use.
3. **Qwen quant improvement.** AP-Q4_K_XL (101,142,769,536 bytes) was removed
   on 2026-09-24; the restore command is in
   `/media/knight2/EDS2/models/ARCHIVED_MODELS_MANIFEST.md` section 3. Download
   only when free space exceeds that size plus the 24 GB reserve. The measured
   rows stay valid. Promote either a speed champion (at least 42.79 tok/s and
   composite at least 0.91) or a balanced champion (composite above 0.9232 and
   at least 35 tok/s).
4. **1Cat-vLLM baseline before branch work.** Pin release `v1.5.0`, run its
   SM70 preflight in an isolated environment and establish target-only quality
   before MTP. Upstream reports 80.732 tok/s target-only and 138.26 tok/s MTP4
   for Flash-Next under its own recorded contract; those are not local claims.
5. **1Cat branch trials.** Evaluate only branches tied to Qwen Flash-Next,
   DeepSeek V4 or GLM 5.3. Record branch commit, wheel hash and build log. Promote
   a branch only if it passes API, determinism, quality and throughput regression
   gates against `v1.5.0`; never merge an experimental fork directly into the
   benchmark controller.
6. **DS4-specific DeepSeek quant.** A DS4 Q2/mixed checkpoint requires its own
   pinned runtime and a new download. Start only after the disk gate passes and
   keep its score separate from generic llama.cpp GGUF results.

## 2026-10-07: re-baseline on the current hardware

The machine now holds four RTX 4000 Ada 20 GB (CUDA 0, 1, 2, 5) and two Tesla
V100-SXM2 32 GB on NVLink (CUDA 3, 4); the RTX A4000 is gone. The Qwen3.8
Flash-Next AP-IQ2_S baselines above were measured on the old four-card set
(A4000 + 2x V100 + one Ada), so they are historical. They are re-measured
first, under the unchanged protocol `v4-mmlu-fewshot-20260918`:

| Row to measure | Topology | Old reference | Result |
|---|---|---|---|
| Qwen3.8 Flash-Next AP-IQ2_S, six GPUs | 4x Ada 20 GB + 2x V100 32 GB, layer split | 40.79 tok/s, composite 0.9232 (old all-four) | **40.80 tok/s, composite 0.9148** (`qwen38-flash-next-ap-iq2s-six`) |
| Qwen3.8 Flash-Next AP-IQ2_S, four Ada | 4x RTX 4000 Ada 20 GB, layer split | none | **40.61 tok/s, composite 0.9180** (`qwen38-flash-next-ap-iq2s-ada4`, 4K context, about 53 GB in VRAM); inside the margin of the old 0.9232 |
| Qwen3.8 Flash-Next AP-IQ2_S, V100 pair | 2x V100 32 GB NVLink, layer split | 42.79 tok/s, composite 0.9113 | **40.77 tok/s, composite 0.9113** (`qwen38-flash-next-ap-iq2s-v100pair`) |

This table is updated in place when a row is measured: the value, the result
row's model identifier and the pull request that published it.

## 2026-10-07: five variants expected to beat the baseline on both axes

Target to beat: 40.79 tok/s and composite 0.9232 together. Each line is a
hypothesis with the measurement it rests on; none is a result yet.

| # | Variant | Why it should be faster | Why composite should hold or rise | Result |
|---|---|---|---|---|
| 1 | Qwen3.8 Flash-Next NVFP4, 1Cat-vLLM TP2 target-only, V100 pair | Upstream reports 80.7 tok/s; locally the same engine gives 111 tok/s on Qwen3.6-35B-A3B NVFP4 | 4-bit weights versus IQ2_S. The existing `qwen38-1cat-vllm-target` row scores 0.34, which points at a template or scoring fault to fix first | **Not servable here** (`qwen38-flash-next-nvfp4-1cat-tp2`): 1Cat 1.5.0 needs `--language-model-only` and an FP16 KV cache, then runs out of memory on two 32 GB V100s even with 62 GB CPU offload per GPU; its SM70 route is gated on four V100s |
| 2 | Variant 1 with DFlash2/MTP speculative decoding | 81.59 tok/s measured at B1 on this pair | Speculation is verified against the target, so quality equals variant 1; reported as its own row | Not measurable: depends on variant 1 |
| 3 | Qwen3.8 Flash-Next AP-Q4_K_XL, six GPUs, fully in VRAM | 34.59 tok/s on the V100 pair with the remainder outside VRAM; 144 GB now holds all 101 GB | Highest measured local composite, 0.9272 | **Not faster: 38.31 tok/s, composite 0.9193** (`qwen38-flash-next-ap-q4kxl-six`); about 73 GB was in VRAM, so the model was not fully offloaded |
| 4 | Qwen3.8 Flash-Next AP-IQ2_S, V100 pair, tensor split | Tensor split is the fastest llama.cpp topology on this NVLink pair (44.87 tok/s for the 27B model versus 33 with layer split) | Same weights as the baseline | **Not supported**: llama.cpp reports `LLAMA_SPLIT_MODE_TENSOR not implemented for architecture 'qwen4exp'` (`qwen38-flash-next-ap-iq2s-v100pair-tensor`); only the `build-v100` binary was tried |
| 5 | Qwen3.8 Flash-Next AP-Q4_K_M, six GPUs | 40.00 tok/s on the old set, where the A4000 was the slowest card | 0.9175 measured, inside the baseline's margin, at 4-bit | **Not faster: 38.52 tok/s, composite 0.9342** (`qwen38-flash-next-ap-q4km-six`); highest local composite so far, inside the margin of the baseline |

Reading the result: the composite has a 95% margin of about ±3.8 points on
these rows (`reports/score_confidence.json`), and all measured Flash-Next rows
are tied. A variant counts as faster on a measured tok/s difference. It counts
as better on composite only if a confirmation run on a larger sample separates
it; until then "not lower" is the honest claim. Variants that are faster but
clearly lower (Qwen3.6-35B-A3B NVFP4 at 111 tok/s and 0.841, Kat-Coder v2.5 at
91.7 tok/s and 0.907) stay in the speed lane and are not listed here.

Measured 2026-10-07: AP-IQ2_S runs at about 40.7 tok/s on four Ada cards, on
the V100 pair and on all six cards alike, so adding cards does not speed this
model up under layer split. Variants 3 and 5 are slower than the baseline and
statistically level on composite; neither beats it on both axes. Variants 1,
2 cannot run on this hardware either (variant 1 does not load on two V100s),
and variant 4 is not supported by the runtime. The AP-IQ2_S baseline stands
and is the day lane since 2026-10-08.

Day-lane quality gate (backlog item 7), measured 2026-10-08 against the live
server without stopping it: Qwen3.8-27B QUASAR NVFP4 on 1Cat-vLLM TP2
target-only scores composite 0.3357 (`qwen38-27b-quasar-nvfp4-1cat-tp2`;
GSM8K 0.32, BBH 0.35, MMLU 0.44, HumanEval 0.225). This reproduces the old
0.34 row, so that row was not a scoring fault: the answers contain wrong
arithmetic and loops on the model's own reasoning. The same model as GGUF
Q4_K_M scores 0.8511 at 33 tok/s. The lane meets its speed goal and fails
its quality gate; Flash-Next AP-IQ2_S at 40.8 tok/s and 0.915 is the measured
alternative for the daytime lane. Variant 1 proper (Flash-Next NVFP4) is not
on disk and stays unmeasured.

Day-lane quality gate (backlog item 7), measured 2026-10-08 against the live
server without stopping it: Qwen3.8-27B QUASAR NVFP4 on 1Cat-vLLM TP2
target-only scores composite 0.3357 (`qwen38-27b-quasar-nvfp4-1cat-tp2`;
GSM8K 0.32, BBH 0.35, MMLU 0.44, HumanEval 0.225). This reproduces the old
0.34 row, so that row was not a scoring fault: the answers contain wrong
arithmetic and loops on the model's own reasoning. The same model as GGUF
Q4_K_M scores 0.8511 at 33 tok/s. The lane meets its speed goal and fails
its quality gate; Flash-Next AP-IQ2_S at 40.8 tok/s and 0.915 is the measured
alternative for the daytime lane. Variant 1 proper (Flash-Next NVFP4) is not
on disk and stays unmeasured.

Claude through the `claude -p` CLI: Opus 5.5 leads the table at 0.9818
(tied with GPT-6 Astra at 0.9802) at 86 tok/s, canary recall 0.8; Sonnet 5.5
scores 0.9542 at 117.7 tok/s, canary recall 0.9. Haiku 4.5
reads 0.8046, but its MMLU cell is a prompt-format artifact and the row is
flagged. Scope is every Claude model from 4.5 to the current one, Fable
excluded: Opus 5 scores 0.9820 at 58 tok/s and Sonnet 5 0.9643 at 74 tok/s;
Opus 4.8 scores 0.9700 at 61 tok/s. Opus 4.7, 4.6, 4.5 and Sonnet 4.6 and
4.5 are still open: the Claude CLI returned 502 on three attempts.

## 2026-10-09: batteries and Kimi on the V100 pair

- **DeepSeek-V4-Flash-0731 UD-IQ4_XS**, experts in RAM, reasoning off:
  composite 0.9362 at 3.16 tok/s (`dsv4-flash-0731`), above the 0.9103 of the
  UD-IQ3_XXS row. The baseline discrepancy for DSv4 is narrowed, not settled:
  this run (V100 pair, reasoning off) and the 2.30 tok/s judge figure (six
  GPUs, reasoning on) both keep the experts in RAM, while the 13.89 tok/s row
  was a smaller quant; the September configuration itself was not re-run.
- **GLM-5.3-Flash UD-Q4_K_XL**: no complete row yet. It decodes at about 1 to
  1.7 tok/s in this configuration and two runs were cut short.
- **Kimi K2.7-Code** (backlog item 2): UD-Q3_K_XL decodes 1.04 tok/s on a
  1200-token code task against 0.68 for UD-Q4_K_XL, and 2 to 3 tok/s against
  0.75 to 1.34 on short probes; both answer the two checked tasks correctly.
  Q3 fits the page cache without a PMem tail.
- **Kimi K3 UD-IQ2_XXS** (backlog item 3): 0.14 to 0.22 tok/s on the V100
  pair with 10 of 16 shards on PMem; the word problem is answered correctly.
  A six-GPU run is in progress separately.

Kimi evidence (raw timings, commands, shard placement):
fieldintelligence/EDSQ-Volta, `evidence/microbench/kimi_k27_q3_vs_q4_k3_20261008.md`.
These are probes, not suite rows, and are not ranked.

## 2026-10-07: cloud providers through the omp CLI

`omp` (18.6.1) fronts several providers with one print mode and one
accounting format, so each goes through the same suite as a local model
(`run_external_provider_cascade.py --provider omp-…`, no tools, no local
context). Throughput comes from `omp bench` as a distribution
(`omp_throughput.py`, median and 95th percentile), not from one completion.

| Provider key | Models | Smoke test | Full suite |
|---|---|---|---|
| `omp-zai` | glm-5.3-flash, glm-5.3 | pass | both done: 0.9466 and 0.9436 |
| `omp-openai-codex` | gpt-6-luna, gpt-6-astra, gpt-6-sol | pass | gpt-6-astra 0.9802, gpt-6-sol 0.9505, gpt-6-luna 0.9257 |
| `omp-google` | gemini-3.8-flash | pass | done (`cloud-omp-google-gemini-3.8-flash`) |
| github-copilot | all tried | 400 "model not supported" | not registered |
| google-antigravity | gemini-3.8-flash | omp: unhandled API mapping | not registered |
| grok-build | grok-4.5 | no answer (not signed in) | not registered |

GPT-6 Astra leads the table at 0.9802 (0.952 to 1.000), at 21 tok/s and
about 4 USD list price for the suite. With it on top, 8 of 53 complete rows
are tied with the leader; the best local row (Qwen3.8 Flash-Next AP-Q4_K_XL,
0.9272) is now measurably below it. The other five omp rows (0.9257 to
0.9658) remain tied with each other.
On the finance lane GLM-5.3 and GPT-6 Luna answer 24 of 24, which with 24
items bounds the true rate above 86%, not at 100%.

GLM-5.3-Flash as served by Z.ai scores composite 0.9466, against 0.8861 for
the local REAP50 IQ4_XS quant and 0.6024 for AJ-IQ2_XXS: the first measured
size of what pruning and quantisation cost this model on our suite. The gap
to the IQ4_XS row is about 2.3 standard errors.

Decode rate over 10 chat requests each (`reports/cloud_cli_throughput.json`),
median with the observed range: Gemini 3.8 Flash 94 tok/s (83 to 702; the top
value is a burst on a very short decode window, which is why the median is
published), GPT-6 Luna 83 (53 to 104), GLM-5.3 60 (47 to 72), GLM-5.3-Flash
46 (43 to 53), GPT-6 Sol 44 (33 to 74), GPT-6 Astra 21 (18 to 28). Time to
first token is 2.0 to 4.5 s at the median.

These rows run without the per-request output cap local rows get (the CLI
cannot truncate), which the row records in `output_budget`.

## 2026-10-07: tool experts for the FQ and video lanes

`specialist_experts.py` scores the bare model and the same model with one
tool on identical items (FQ: forward-kinematics simulator; video: FFmpeg
render and repair, at most three rounds). Results go to
`reports/specialist_experts.json`. First row, GLM-5.3-Flash through omp: FQ
0 of 2 bare to 2 of 2 with the simulator, video 0 of 3 to 1 of 3 with the
repair loop. The bare FQ answers failed as unparseable JSON, so part of the
gain is format repair.

On the 73-item parametric FQ pack (`fq_v2_parametric.csv`), same model:

| | Correct | 95% interval |
|---|---:|---|
| Bare model | 35 of 73 (48%) | 37% to 59% |
| With simulator expert | 65 of 73 (89%) | 80% to 94% |

Paired on the same items the expert fixes 34 and breaks 4 (exact McNemar
p < 0.0001), at 2 or 3 model calls per item instead of 1. Of the bare
model's 38 misses, 31 were unparseable output and 7 a wrong pose. Of the
expert's 8 misses, 6 reached the pose within tolerance but took longer than
the 10 s limit, which the task text does not state; 2 were a wrong pose. The
next pack version states the limit in the prompt.

Pack v3 (`fq_v3_parametric.csv`, same 73 targets, limit stated): bare 31 of
73 (42%, 31% to 54%), with the expert 70 of 73 (96%, 89% to 99%); paired, 41
fixed and 2 broken. The three remaining misses are wrong poses. The lane is
not at 100%, and one model is not yet a claim about the mixture.

## 2026-10-07: disk freed for the Flash-Next runs

Removed from the model disk after the retention check, with restore commands
in `ARCHIVED_MODELS_MANIFEST.md`: Qwen3.5-397B-A17B UD-Q4_K_M (244 GB, never
measured, sizes matched the Hub) and MiMo-V2.6-Pro BPW2.5 (298 GB, complete
row kept, SHA-256 per shard recorded). The model disk went from 111 GB to
570 GB free. Kimi K2.7 Q3 and Q4 are kept on purpose. Flash-Next AP-IQ2_S,
AP-Q4_K_M and AP-Q4_K_XL (81.6, 94.5 and 101.1 GB, pinned revision) are on
disk for the re-baseline and for variants 3 and 5; 378 GB remains free.

## 2026-10-07: related plans

- Specialist mixture of experts with the self-healing and self-learning
  loops: [SPECIALIST_MIXTURE_OF_EXPERTS_PLAN.md](SPECIALIST_MIXTURE_OF_EXPERTS_PLAN.md).
- Agent CLIs as benchmark rows: see the omp section below. The direct ZCode
  backend in `external_cli_agent_proxy.py` still waits on a model provider in
  the headless CLI's own config; Z.ai's GLM is reached through omp instead.
- Lessons from professional suites and what was adopted:
  [PROFESSIONAL_SUITE_LESSONS.md](PROFESSIONAL_SUITE_LESSONS.md).

## Sources and non-portable reference results

- [1Cat-vLLM 1.5.0](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.0)
  documents SM70 paths for Qwen3.8 Flash-Next, DeepSeek-V4-Flash and GLM-5.3,
  with explicit warnings that results are workload/topology specific.
- [Qwen3.8 Flash-Next AP-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-AP-GGUF)
  is the source of the locally tested AP quant family.
- [Qwen3.8 Flash-Next NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
  is a mixed-format checkpoint; support must be demonstrated by the selected
  runtime rather than inferred from the model name.
- [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
  is the official quality target. [DS4 model guidance](https://github.com/antirez/ds4/blob/main/docs/MODELS.md)
  applies only to DS4-compatible files and runtime revisions.
- [NVIDIA GLM-5.3-Flash NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)
  remains disk-gated; the existing GGUF re-test comes first.

## Promotion and evidence rules

- Smoke gate: pinned source hash, engine health, correct model identity, no OOM,
  and a deterministic short output check.
- Full gate: current protocol GSM8K, BBH, MMLU and HumanEval plus specialist
  holdout; public-suite leader and private/holdout leader are separate titles.
- Throughput gate: report prompt and decode separately, including context,
  batch/concurrency, warm/cold state, physical GPU IDs and telemetry.
- MTP/speculative rows never replace target-only rows. Failed or unsupported
  profiles remain visible without a numeric score.
- Do not delete a model until its result, source revision, SHA-256 and raw logs
  are present in durable evidence and a retained winner is identified.
