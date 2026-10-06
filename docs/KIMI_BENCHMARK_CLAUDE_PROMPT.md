# Prompt voor de aparte Claude-benchmarkchat

Voer de Kimi K2.5-vergelijking in `virtualv_llm` uit zodra de lopende run klaar
is. Gebruik de geïsoleerde featurebranch `feat/kimi-knowledge-mom` of de latere
merge daarvan. Behandel Moonshot AI als maker van het model en Unsloth als
bron van de GGUF-quant; vermeld onze engine-/MoM-configuratie apart.

**Readiness vóór elke score:** verifieer het volledige Unsloth UD-IQ3_XXS
GGUF (9 shards) onder `/media/knight2/eds1/models/kimi-k2.5-gguf/UD-IQ3_XXS`
tegen `.sha256`; vind het exacte Q4-GGUF-artifact, alle shards en hash, en
noteer zijn quant-type. Q4 stond bij het opstellen van deze prompt nog niet
lokaal en eds1 had ongeveer 470 GB vrij. Verifieer
`infra/kimi_k25/preflight.py` en beide beoogde engines met een echte laad-,
generatie- en nauwkeurigheidssmoke op deze host. Colibri v1.11.0 wees
`kimi_k25` af. 1Cat-vLLM 1.5.0 heeft de modelklasse, SM70 TurboMind INT4,
Triton WNA16 MoE en een FusedMoE weight loader, maar geen geverifieerde
tier-loader voor 595 GB. Onderzoek Colibri als primaire tierroute en de
1Cat FusedMoE-inhaak als research-pad. Een descriptor, registry-entry of
kernel telt niet als werkende engine. Laat PMem/NUMA-configuratie,
GPU-indices en thermische status zien. Start niets dat de lopende benchmark
verstoort. Als een artifact of engine ontbreekt, noteer de blokkade; vul geen
cijfers in en verzin geen resultaat.

**Vergelijking:** gebruik voor elk van Q3 en Q4 dezelfde hardware, context,
sampling, instant-/thinking-instelling, warmups, promptlijst en tijdslimiet.
Meet de kale geoptimaliseerde engine afzonderlijk (B1 en B4 decode, prefill,
geheugen, energie). Vergelijk dan vier antwoordsystemen in deze volgorde:

1. Kale Kimi-engine, geen MoM/kennis/JEV.
2. Kimi als dezelfde selectieve specialist in dezelfde MoM, met lokale JEV;
   laat overige leden en aggregator vaststaan.
3. Conditie 2 plus de bevroren private LightRAG-, GitNexus- en
   codegraph-bundel uit `docs/KIMI_KNOWLEDGE_MOM.md`.
4. Conditie 3 plus de lokale Lens-connector op exact die bundel.

Gebruik het 2×V100-SXM2 NVLink-bord als één 64 GB TP2-domein voor de
Kimi-hot-path wanneer de engine dat echt ondersteunt. Laat de Ada-modellen
op hun eigen kaarten en endpoints parallel voorstellen genereren; log
V100- en Ada-bezetting afzonderlijk. Houd de andere MoM-leden vast tussen
Q3 en Q4. Als een GGUF-engine op deze host geen stabiele TP2-split biedt,
registreer dat als aparte topologie en vergelijk hem niet alsof hij TP2 is.
`infra/model_serve_configs/mom-live.sh` heeft hiervoor `MOM_KIMI_TP2_URL`:
Kimi extern op V100 TP2, Qwen3.8 aggregator op Ada 5 en drie proposers op
Ada 0/1/2. Meet eerst of die Ada-aggregator met 8192 context/parallel 1
daadwerkelijk past. Houd deze layout voor beide quants gelijk. Zet
`MOM_SPECIALIST_MODEL` als het endpoint een model-ID vereist.

Gebruik na de readinesscheck deze configuratievariabelen per MoM-rij. Vervang
`KIMI_URL`, `KIMI_MODEL`, `BUNDLE`, `LENS_ROWS` en het rapportpad pas nadat
de endpoints en bestanden zijn geverifieerd:

```bash
# JEV is lokaal via ~/.config/virtualv_llm/mom-jev.env (mode 600).
# Start de bestaande jevserver op 127.0.0.1:8092 als hij niet draait.
export MOM_KIMI_TP2_URL="$KIMI_URL"   # volledig http://127.0.0.1:PORT/v1
export MOM_SPECIALIST_MODEL="$KIMI_MODEL"
export MOM_EVENTS_OUT="$EVIDENCE/mom-events.jsonl"
# B: unset MOM_KNOWLEDGE_BUNDLE MOM_LENS_ROWS
# C: export MOM_KNOWLEDGE_BUNDLE="$BUNDLE"; unset MOM_LENS_ROWS
# D: export MOM_KNOWLEDGE_BUNDLE="$BUNDLE"; export MOM_LENS_ROWS="$LENS_ROWS"
infra/model_serve_configs/mom-live.sh start
python3 scripts/benchmarks/eight_task_external.py "$ROW_NAME" \
  --external-url http://127.0.0.1:8030 --external-model mom-live \
  --mom-task-labels --topology 'V100-SXM2 TP2 Kimi + Ada asynchronous MoM' \
  --out "$EVIDENCE/eight_task.json"
python3 scripts/benchmarks/summarize_mom_events.py "$MOM_EVENTS_OUT" \
  --out "$EVIDENCE/mom-system-metrics.json"
infra/model_serve_configs/mom-live.sh stop
```

Gebruik voor A rechtstreeks `eight_task_external.py` op het kale Kimi-endpoint
zonder `--mom-task-labels`; houd de instantmodus gelijk aan B/C/D. Maak per
quant en conditie een eigen evidence-map en start geen tweede model op dezelfde
V100's. Leg bij B vast dat JEV een fact/rule-classificatie zonder externe
kennis doet, en bij C/D de ondersteuning van opgehaalde bronnen beoordeelt.
Dat zijn verschillende adviesvragen; tel dit mee bij de interpretatie.

Gebruik expliciete `X-MoM-Task`-labels (`coding`, `humaneval`,
`complex_reasoning`, `long_context`, `project_facts`) zodat Kimi alleen op
de aangewezen taken wordt ingezet. Meet de bestaande 8-takenbatterij met
`scripts/benchmarks/eight_task_external.py --mom-task-labels` voor de MoM-rijen
op dezelfde prompts en een apart,
vooraf bevroren private holdout voor feit-/regelvragen en codingvragen. Houd
deze holdout buiten LightRAG, GitNexus, Lens en de publicatie-export. Draai
`well_known_suite.py --mom-task-labels` alleen wanneer de Q3-snelheid de duur
toelaat; een afgebroken suite krijgt status `timeout`, geen deelscore als
volledige score. Bouw de codegraph opnieuw op na elke codecommit en bewaar
commit, bundeldigest en taaklabels bij elk resultaat.

**Eerlijke rapportage:** rapporteer per rij kwaliteit, fouttypes, p50/p95
antwoordlatency, antwoorden/min, GPU/CPU/PMem-verbruik, Wh/antwoord,
directe kennisantwoorden, JEV-calls, Lens-hits en specialist-invocaties.
Rapporteer raw engine tokens/s alleen uit de kale engine-run, met quant,
engineversie, context en fysieke hardware. Een opgeslagen antwoord gebruikt
nul modeltokens; dat mag de geclaimde model-tokens/s niet verhogen. De
JEV-backend gebruikt zelf Qwen en zijn score is buiten PIQA ongecalibreerd;
tel zijn latency en modelverbruik mee bij het hele systeem. Scheid engine,
MoM, kennis en Lens effecten in de tabel. Herhaal meetbare verschillen om
run-to-run-variantie vast te stellen. Bewaar ruwe verzoeken/responses waar
toegestaan, timings, configs, hashes en failures onder `evidence/` of een
private evidence-map met verwijzing in de repo.

De private JEV-regels, keys, LightRAG-data en GitNexus-DB blijven lokaal.
`prepare_public_knowledge.py` levert alleen een lokale unsigned stage voor
Knitweb/Pulse, Lens en ClosedIntelligence; publiceer daaruit uitsluitend de
records die expliciet `publish_allowed:true` zijn en waarvan bron/licentie en
de exacte export zijn beoordeeld. Geen upload als bijwerking van benchmarken.

Geef mij tot slot een Q3-versus-Q4-tabel per engine en per vier condities,
een korte analyse van kwaliteitswinst versus extra kosten, de reproduceerbare
commando's en bewijsbestanden, plus de open blokkades. Kies alleen een
standaardvariant als hij op dezelfde holdout beter of aantoonbaar zuiniger is.
