# VirtualV LLM-testsuite — standaard en proces

Documenteigenaar: VirtualV AI Assurance  
Versie: 1.0  
Peildatum: 22 september 2026  
Status: operationele standaard van de huidige implementatie  
Classificatie: publiek (onderdeel van de virtualv_llm-repository)

## 1. Doel en reikwijdte

Deze standaard beschrijft hoe VirtualV lokale large-language-models selecteert,
uitvoert, meet, vergelijkt en archiveert. Het document geldt voor de publieke
algemene suite, de afgeschermde specialistensuite, hardwareprofielen, individuele
modellen, mixtures en hervatbare cascades. Het is zowel een governancekader als
een reproduceerbaar werkproces.

De standaard voorkomt drie veelvoorkomende fouten: scores vergelijken die onder
een ander protocol zijn verkregen, throughput los van runtime en hardware
presenteren, en mislukte of onuitvoerbare proeven als gemeten resultaat
voorstellen. Alleen bewijs dat volgens dit document is vastgelegd, mag als
VirtualV-benchmark worden gepubliceerd.

## 2. Kernprincipes

1. **Bewijs vóór conclusie.** Iedere score verwijst naar protocol, runtime,
   modelbron, hardwareprofiel en ruwe uitvoer.
2. **Kwaliteit en snelheid blijven gescheiden.** Tok/s is contextinformatie en
   geen vervanging voor taakaccuratesse.
3. **Vergelijk alleen gelijkwaardige runs.** Protocolversie, promptbeleid,
   context, engine en hardwaretopologie moeten zichtbaar zijn.
4. **Geen verzonnen uitkomsten.** OOM, ontbrekende kernel, timeout en
   onondersteunde modaliteit krijgen een expliciete status zonder fictieve score.
5. **Duurzame uitvoering.** Lange cascades draaien onder systemd, schrijven
   atomair en kunnen na een storing hervatten zonder bewezen resultaten te
   verliezen.
6. **Minimale toegang.** Evaluaties krijgen alleen de hulpmiddelen en gegevens
   die het gekozen profiel toestaat; echte home-, secret- en productiegegevens
   blijven buiten bereik.
7. **Behoud vóór opruimen.** Gewichten mogen pas worden verwijderd nadat een
   volledige, gevalideerde rij, herkomst, hash en herinstallatiepad duurzaam zijn
   opgeslagen.

## 3. Begrippen en statussen

| Begrip | Betekenis |
|---|---|
| Model/checkpoint | De logisch benoemde gewichten met een vastgezette bron en revisie. |
| Quant | De concrete numerieke representatie, bijvoorbeeld GGUF IQ4_XS of NVFP4. |
| Engine | De uitvoerende runtime, bijvoorbeeld llama.cpp, qwen4exp of 1Cat-vLLM. |
| Hardwareprofiel | Fysieke GPU's, zichtbare devicevolgorde, topologie en split/parallelisatie. |
| Resultaatrij | Eén model- of mixture-uitvoering onder één aantoonbaar protocol. |
| Huidig-volledig | Geen fout, huidige protocolversie en alle vier composietcomponenten aanwezig. |
| Partieel | Geldige deelmetingen, maar niet voldoende voor de algemene ranglijst. |
| Niet ondersteund | Aantoonbaar onuitvoerbaar; geen snelheid of kwaliteit verzinnen. |
| Mislukt | De uitvoering stopte door een fout; logpad en foutreden blijven bewaard. |
| Geblokkeerd | Een bekende externe of technische voorwaarde verhindert uitvoering. |

Een status is zelf een resultaat. Alleen `huidig-volledig` mag in de algemene
composietranglijst worden opgenomen.

## 4. Hardware- en runtimeprofielen

De fysieke GPU-indexering is vast en wordt niet afgeleid uit de volgorde die een
runtime toevallig toont.

| Fysieke index | GPU | Geheugen | Primaire rol |
|---:|---|---:|---|
| 0 | NVIDIA RTX A4000 | 16 GB-klasse | gemengd/all-four-profiel |
| 1 | Tesla V100-SXM2 | 32 GB | V100-paar, NVLink |
| 2 | Tesla V100-SXM2 | 32 GB | V100-paar, NVLink |
| 3 | NVIDIA RTX 4000 Ada | 20 GB-klasse | gemengd/all-four-profiel |

Voor iedere run geldt `CUDA_DEVICE_ORDER=PCI_BUS_ID`. De resultaatrij bevat ten
minste `cuda_visible_devices`, fysieke GPU-identiteiten, uitgesloten GPU's,
topologie, engine en decode-telemetrie.

De uitvoeringspaden worden strikt onderscheiden:

- **llama.cpp/GGUF:** lokale GGUF-gewichten; langdurige V100-runs gebruiken
  layer split. Tensor split is niet de standaard voor onbeheerde V100-runs door
  eerder waargenomen driver-handlefouten.
- **qwen4exp/llama.cpp:** experimentele Qwen-runtime voor Qwen3.8 Flash-Next
  GGUF en eventueel een afzonderlijk MTP-resultaat met passende draft-sidecars.
- **1Cat-vLLM:** zelfstandig vLLM-pad voor ondersteunde safetensors- en
  quantisatieruntimes. Een 1Cat-resultaat mag niet als llama.cpp-resultaat worden
  gepresenteerd of omgekeerd.
- **Mixture:** orkestratie over meerdere reeds gemeten modellen. De
  kwaliteitswinst is niet rechtstreeks gelijk aan de kosten of latency van één
  checkpoint en staat daarom in een aparte tabel.

De mislukte Qwen3.8 Flash-Next W4A16-proef betrof specifiek de gebruikte
compressed-tensors/Merlin-route. Dit is geen algemene uitspraak dat alle W4A16-
of 1Cat-vLLM-routes op SM70 onmogelijk zijn. Nieuwe 1Cat-branches of releases
worden eerst op commit vastgezet, in een geïsoleerde omgeving gepreflight en pas
na een volledige huidige-protocolrun bevorderd.

## 5. Algemene kwaliteitssuite

De geldende protocol-ID is `v4-mmlu-fewshot-20260918`. De openbare batterij is:

| Onderdeel | Omvang | Meetwaarde |
|---|---:|---|
| GSM8K | 50 opgaven | hoogste geldige waarde van strict-match en flexible-extract |
| BBH | 6 subtaken × 8 voorbeelden = 48 | gemiddelde accuratesse |
| MMLU | 8 vakgebieden × 20 vragen = 160 | gemiddelde 5-shot accuratesse |
| TruthfulQA generation | 30 prompts | contextdiagnostiek, niet in composiet |
| HumanEval | 40 programmeertaken | pass@1 in geïsoleerde subprocessen |

BBH omvat boolean expressions, causal judgement, date understanding, logical
deduction met vijf objecten, navigation en temporal sequences. MMLU omvat
abstract algebra, anatomy, astronomy, college computer science, high-school
psychology, formal logic, professional law en marketing.

Generatie is deterministisch met `temperature=0` en de modelgebonden chat-
template. Voor llama.cpp staat reasoning uit wanneer reasoning-content anders
een leeg antwoordveld veroorzaakt. Stoptekens worden taakgericht gekozen;
MMLU gebruikt `</s>`, `Q:` en `<|im_end|>`, terwijl BBH `</s>` en `Q` gebruikt.
Een generieke newline-stop is verboden omdat die geldige antwoorden kan
afkappen. De rescoring normaliseert alleen expliciete, losstaande keuzen.

De algemene composietscore is het ongewogen gemiddelde:

`composiet = (GSM8K + BBH + MMLU + HumanEval) / 4`

TruthfulQA en throughput worden getoond als context maar wijzigen deze score
niet. Een model komt alleen in de ranglijst als alle vier componenten onder de
huidige protocol-ID aanwezig zijn.

## 6. Throughputmeting

Throughput wordt gemeten met één deterministische completion van maximaal 256
outputtokens en `ignore_eos=True`, met GPU-telemetrie tijdens decode. De eenheid
is completion tokens per seconde. De waarde is alleen betekenisvol naast engine,
quant, context, parallelisatie en fysieke hardware.

MTP-resultaten krijgen een aparte rij en vervangen nooit de normale decode-
meting. Target-only, MTP, prompt-prefill en pure decode mogen niet onder één
label worden samengevoegd.

## 7. Specialistensuite en toegangsprofielen

De specialistensuite gebruikt protocol
`v2-private-specialist-packs-vision-video-iq-eq-fq-qq-20260922`. De lanes zijn
chemie, fysica, vision, video, IQ, EQ, FQ en QQ. Resultaten komen in
`reports/specialist_suite_20260922.json` en tellen niet mee in de algemene
composiet.

De openbare repository bevat startpakketten; geroteerde privésets staan onder
de genegeerde map `data/eval_cache/specialist_packs`. Hashes van gebruikte packs
worden in het resultaat opgeslagen en antwoordsleutels worden nooit aan het
model aangeboden. FQ gebruikt een deterministische differentiële-drive-
simulatie. Video verwacht een door het model opgesteld FFmpeg-filtergraph dat
via vaste, veilige argumenten en een wandtijdlimiet van vijf minuten draait.

| Profiel | Toegang |
|---|---|
| sandbox | alleen prompt en modelantwoord; geen externe hulpmiddelen |
| disk | synthetisch, alleen-lezen evaluatiearchief |
| internet_disk | hetzelfde archief plus gelogde, allow-listed webtool |

Raw endpoints ondersteunen alleen sandbox. Geen profiel krijgt toegang tot de
echte home-directory, gitcredentials, chatgeschiedenis, secrets of
productiedata. Een tekstmodel krijgt voor vision/video de status `unsupported`,
niet een nulscore die modelkwaliteit suggereert.

## 8. Bewijs- en herkomstcontract

Een publiceerbare rij bevat, waar van toepassing:

- model-ID, quantisatie, engine en engineversie;
- bronrepository, vastgezette revisie, publicatiedatum, gewichtsgrootte en
  SHA-256;
- protocol-ID, taakconfiguratie en scorer;
- fysieke GPU's, zichtbare devicevolgorde, topologie, context en splitbeleid;
- start/eindtijd, runtime, tok/s en GPU-decodetelemetrie;
- taakscores, composiet, status en fout- of logverwijzing.

Het primaire openbare resultaatbestand is
`reports/well_known_suite_20260917.json`. Schrijfacties gaan via een tijdelijk
bestand en atomaire vervanging. Nieuwe modelrijen mogen bestaande rijen niet
stilzwijgend overschrijven. Ruwe logs staan per model onder
`reports/lm_eval_runs/`; het dashboard wordt afgeleid naar
`reports/dual_v100_nvlink_benchmark.html`.

## 9. Operationeel proces

1. **Intake.** Leg zakelijke vraag, modelbron, licentie, revisie, quant en
   verwachte geheugenbehoefte vast.
2. **Preflight.** Controleer bestandsgrootte/hash, architectuur, compute
   capability, kernels, vrije schijfruimte en incompatibele actieve services.
3. **Classificeer.** Kies engine, hardwareprofiel, context en suite. Nieuwe
   branches zijn kandidaat, geen productiestandaard.
4. **Reserveer.** Verkrijg de exclusieve GPU-lock, stop alleen bekende
   conflicterende diensten en verifieer dat geen compute-context resteert.
5. **Start.** Gebruik een expliciete commandoregel of een beheerde systemd-unit;
   schrijf state en gebeurtenis vóór elke lange fase.
6. **Evalueer.** Voer kwaliteit, throughput en relevante specialistlanes uit
   zonder protocolparameters tijdens de run te wijzigen.
7. **Persisteer.** Schrijf iedere duurzame overgang en resultaatrij atomair,
   inclusief foutstatus.
8. **Valideer.** Controleer volledigheid, protocol-ID, herkomst, GPU-mapping,
   scoreberekening en ruwe logs.
9. **Publiceer.** Herbouw het HTML-dashboard en deze gedateerde rapportage.
10. **Behoud of ruim op.** Verwijder uitsluitend een aantoonbaar lagere,
    volledige kandidaat nadat bewijs en reproduceerpad zijn veiliggesteld.
11. **Herstel.** Start eerder gestopte bekende diensten opnieuw, laat geen
    verweesde server achter en geef lock en poorten vrij.

De automatische lokale sweep draait iedere vijftien minuten met
`/tmp/llm_bench_sweep.lock` en kiest maximaal één reeds lokaal aanwezig model
zonder huidige resultaatrij. Er worden geen gewichten gedownload. De HTML-
renderer draait afzonderlijk op minuut 17, 32, 47 en 57 met
`/tmp/llm_bench_html.lock`.

## 10. Wijzigings- en acceptatiebeleid

Een wijziging in prompts, samples, few-shotvoorbeelden, stoptekens, normalisatie,
scorer of aggregatie vereist een nieuwe protocol-ID. Oude rijen blijven
historisch zichtbaar maar zijn niet automatisch rangschikbaar naast de nieuwe
standaard. Een engine- of kernelwijziging zonder protocolwijziging vereist
minimaal een nieuwe runtime-identificatie en een regressieproef.

Een kandidaat wordt geaccepteerd wanneer de run reproduceerbaar is, de
bewijsvelden compleet zijn, de ruwe logs aansluiten, het resultaat door een
tweede persoon of geautomatiseerde validatie is gecontroleerd en er geen
onverklaarde score- of snelheidsafwijking bestaat.

## 11. Rollen en verantwoordelijkheden

| Rol | Verantwoordelijkheid |
|---|---|
| Suite-eigenaar | protocol, acceptatiecriteria en wijzigingsbesluit |
| Operator | veilige uitvoering, monitoring, state en herstel |
| Reviewer | herkomst, logs, berekening en vergelijkbaarheid controleren |
| Rapportage-eigenaar | dashboard/PDF publiceren zonder brondata te wijzigen |

Dezelfde persoon mag meerdere rollen uitvoeren, maar protocolwijzigingen en
definitieve verwijdering van modelgewichten moeten aantoonbaar worden gereviewd.

# Bijlage A — Instructiehandleiding voor operators

## A.1 Voorbereiding

Werk vanuit de repository (het pad naar je eigen kloon):

```bash
cd virtualv_llm
```

Controleer vóór iedere handmatige actie de actieve services, GPU-processen,
vrije ruimte en repositorywijzigingen:

```bash
systemctl --user status qwen38-flash-next-gguf-cascade.service --no-pager
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
df -h .   # of het specifieke volume waar je modelgewichten staan
git status --short
```

Een actieve benchmark wordt niet onderbroken om documentatie te genereren. Start
geen tweede run op dezelfde GPU's en omzeil de exclusieve lock niet.

## A.2 Lokale algemene suite starten

Gebruik de één-model-scheduler voor reeds lokale kandidaten:

```bash
flock -n /tmp/llm_bench_sweep.lock \
  python3 scripts/benchmarks/run_next_benchmark.py
```

Gebruik voor een expliciet model de runner met de lokaal vastgelegde configuratie:

```bash
python3 scripts/benchmarks/well_known_suite.py MODEL_ID \
  --out reports/well_known_suite_20260917.json
```

Controleer eerst `--help`; voeg geen niet-bestaande optie toe en verander de
protocolparameters niet ad hoc.

## A.3 Externe of reeds draaiende endpoint meten

Bij een extern endpoint zijn fysieke GPU's, topologie en engine verplicht:

```bash
python3 scripts/benchmarks/well_known_suite.py MODEL_ID \
  --external-url http://127.0.0.1:PORT \
  --external-model SERVED_MODEL \
  --physical-gpus 1,2 \
  --topology '2x Tesla V100-SXM2-32GB + NVLink' \
  --engine 'ENGINE_NAAM' \
  --out reports/well_known_suite_20260917.json
```

Gebruik alleen localhost of een expliciet goedgekeurde endpoint. Verifieer
`/health` en modelalias vóór de suite start.

## A.4 Qwen3.8 Flash-Next-cascade bedienen

De actieve cascade is hervatbaar en chat-onafhankelijk:

```bash
systemctl --user status qwen38-flash-next-gguf-cascade.service --no-pager
journalctl --user -u qwen38-flash-next-gguf-cascade.service -f
tail -f reports/benchmark_logs/qwen38_flash_next_gguf_cascade.service.log
```

Veilig stoppen en later hervatten:

```bash
systemctl --user stop qwen38-flash-next-gguf-cascade.service
systemctl --user start qwen38-flash-next-gguf-cascade.service
```

De duurzame state staat in
`reports/qwen38_flash_next_gguf_cascade_20260922.json`. Bewerk dit bestand niet
handmatig terwijl de service draait. De geplande volgorde is AP-IQ4_XS op het
V100-paar en alle vier GPU's, daarna AP-Q4_K_M op beide profielen, dan selectie
en pas daarna een afzonderlijke MTP-proef voor de winnaar.

## A.5 Specialistensuite uitvoeren

Gebruik alleen aanwezige, gehashte packs en het minimaal benodigde
toegangsprofiel:

```bash
python3 scripts/benchmarks/specialist_suite.py --help
```

Controleer vóór publicatie dat de packhash, lane, toolprofiel en status zijn
opgeslagen. Deel privévragen en antwoordsleutels niet in logs, dashboards of de
PDF.

## A.6 Monitoring

Volg proces, fase, loggroei en GPU-telemetrie. Een lage GPU-utilisatie kan
normaal zijn tijdens datasetvoorbereiding, scorerwerk of een CPU-gebonden taak;
beoordeel daarom ook proces-CPU en recente logregels.

```bash
ps -eo pid,etime,%cpu,%mem,cmd | rg 'well_known_suite|specialist_suite|llama-server|vllm'
tail -n 80 reports/benchmark_logs/qwen38_flash_next_gguf_cascade.service.log
nvidia-smi dmon -s pucvmet
```

Fan Manager mag core-, hotspot- en geheugentemperaturen tonen voor zover de
driver/sensor die levert. Een ontbrekende geheugentemperatuur wordt als niet
beschikbaar gemarkeerd en nooit afgeleid uit de coretemperatuur.

## A.7 Valideren en publiceren

Voer syntaxiscontrole uit op gewijzigde runners, controleer de JSON en bouw het
dashboard opnieuw:

```bash
python3 -m py_compile scripts/benchmarks/*.py
python3 -m json.tool reports/well_known_suite_20260917.json >/dev/null
python3 scripts/benchmarks/build_dual_v100_html.py
git diff --check
```

Controleer vervolgens dat alleen huidige-volledige rijen gerangschikt zijn, de
composiet uit exact vier componenten bestaat, mixture-resultaten apart staan en
fouten zichtbaar blijven.

## A.8 Incidentprocedure

1. Noteer tijd, service, model, engine, GPU's en laatste logregel.
2. Stop alleen de betrokken beheerde service; gebruik geen brede kill-opdracht.
3. Bewaar state, resultaat-JSON en logs vóór herstelwerk.
4. Controleer lock, poort, compute-context, vrije VRAM/schijf en bestandshash.
5. Herstel de oorzaak in code of configuratie en voer minimaal compile/dry-run
   uit.
6. Hervat via systemd en verifieer dat geen dubbele runner of download bestaat.
7. Registreer fout én herstel als duurzame gebeurtenis.

Nooit doen: een score handmatig invullen, een incomplete rij als voltooid
markeren, een actieve productie- of benchmarkdienst willekeurig beëindigen,
GPU-indexen aannemen zonder fysieke mapping, privé-evaldata archiveren of een
modelmap verwijderen vóór de bewaartoets.

## A.9 Afsluitchecklist

- Resultaat en foutstatus atomair opgeslagen.
- Protocol, engine, bronrevisie en hardwaretopologie aanwezig.
- Ruwe logs en state leesbaar.
- Geen verweesde servers of GPU-compute-contexten.
- Bekende lokale diensten hersteld.
- Dashboard opnieuw opgebouwd.
- Verwijdering, indien van toepassing, beperkt tot de gevalideerde verliezer.
- Codewijziging compileert en `git diff --check` is schoon.

# Bijlage B — Actuele benchmarkresultaten

Deze bijlage wordt bij het bouwen van de PDF rechtstreeks samengesteld uit
`reports/well_known_suite_20260917.json`. De generator voegt de peiltijd,
bestandshash, volledige tabellen voor individuele modellen en mixtures,
foutstatussen en een zakelijke toelichting toe. De JSON blijft de primaire
bewijsbron; de PDF is een gedateerde, leesbare momentopname.
