# Lessen uit de live mixture-of-models-benchmark (2026-10-05) en roadmap GLM-5.3 Flash → Kimi K2.5

Dit document legt vast wat de live mixture-of-models-runs van 2026-10-05 hebben geleerd, zodat de volgende
modellen (GLM-5.3 Flash, daarna Kimi K2.5) meteen goed worden gemeten. Alle cijfers komen uit
`reports/well_known_suite_20260917.json`, `reports/live_mixture_of_models_*.json`,
`reports/specialist_suite_20260922.json` en `reports/contamination_audit_20260923.json`.

## 1. Kwaliteit

| Opstelling | Composite | Opmerking |
|---|---|---|
| `mom-live-4`: Qwen3.8 Q4_K_M (llama.cpp) aggregator ×2 + Devstral, Qwen3.5 ×2, Gemma4 | **0,885** | beter dan elk eigen lid (beste lid Qwen3.8: 0,851); bbh 0,646 → 0,771 |
| `mom-live-4-tp2`: Qwen3.8 NVFP4 (1Cat-vLLM, TP2) aggregator, zelfde proposers | 0,578 | NVFP4 via 1Cat kost kwaliteit; strookt met de losse 1Cat-rij (gsm8k 0,28) |

- **De aggregator bepaalt de kwaliteit.** Dezelfde concepten leverden met een zwakkere aggregator een veel lagere
  score op. Kies de aggregator op losse kwaliteit en meet die eerst alleen.
- **Niet elke NVFP4-checkpoint is gelijk.** Qwen3.8 NVFP4 op 1Cat scoort laag, Qwen3.6-35B-A3B NVFP4 op 1Cat scoort
  los 0,841 bij 115 t/s. Beoordeel per checkpoint, niet per formaat.
- **Contaminatie-audit hoort bij elke run** (`well_known_suite.py` doet die standaard): canary-recall, parafrase-
  invariantie en post-cutoff holdout. De live mix: canary 0,0, parafrase-gat −0,13, holdout 0,75.

## 2. Energie en doorvoer

- **Vergelijk energie alleen op hetzelfde werk.** Eerst werd een replica-venster (einde mmlu + crashpauze +
  humaneval/specialist/audit) naast een volledige TP2-suite gezet; de conclusie "gelijk" was fout. Op hetzelfde
  deel (humaneval + specialistsuite + audit, 142 antwoorden in beide): replica's 2,16 Wh/antwoord bij 543 W en
  4,19 antwoorden/min, TP2 2,90 Wh bij 611 W en 3,51/min. Gebruik `live_mixture_report.py --phase-since/--phase-until`.
- **Rustverbruik domineert.** Twee V100's trekken samen ~110 W in rust; de aggregator-replica's draaiden maar ~10%
  van de tijd. Energiewinst zit in minder of zuiniger actieve kaarten, niet in een andere parallellisatie.
- **Dense 27B op een RTX 4000 Ada is traag** (17–22 t/s, 360 GB/s geheugenbandbreedte); MoE (Gemma4-26B-A4B,
  73 t/s) past beter op die kaarten. Dense modellen horen op de V100's (900 GB/s).
- **Row split over het V100-paar werkt niet** in de llama.cpp-V100-build ("does not support split buffers");
  layer split versnelt een 27B niet. Twee replica's (data-parallel) of 1Cat TP2 zijn de opties.

## 3. Operationele lessen

| Valkuil | Oplossing |
|---|---|
| GPU-indexen veranderd (V100 nu CUDA 3,4; A4000 weg) | `infra/lib/v100_pair.sh` zoekt het paar op naam |
| 1Cat-vLLM weigert GPU-UUID's in `CUDA_VISIBLE_DEVICES` | indexen doorgeven, geen UUID's |
| Run stierf met de Claude-sessie | lange runs als `systemd-run --user --unit=…` |
| `systemd --user`: 1024 open bestanden; torch deelt tensors via fd's | gewichten als numpy naar workers, of `-p LimitNOFILE=1048576` |
| Ongetrackte `data/eval_cache` ontbrak in een worktree | symlink naar de hoofdrepo; suite met `--resume` hervatten |
| `--external-url …/v1` gaf `/v1/v1/…` | basis-URL zonder `/v1` meegeven |
| Qwen/Gemma-templates weigeren een tweede systeembericht | MoA-concepten in het eerste systeembericht (`mixture_proxy.py`) |
| 20 commits stonden maanden alleen lokaal | `git log origin/main..HEAD` vóór elke dashboard-publicatie |

## 4. Roadmap: GLM-5.3 Flash, daarna Kimi K2.5

Volgorde: eerst GLM-5.3 Flash, daarna onze eigen Kimi K2.5-versie (NVFP4 en GGUF Q3/Q4). Elke stap: los model
benchmarken (zelfde suite + specialist + audit + energie), daarna als aggregator of lid in de live mix.

### GLM-5.3 Flash (313B / ~17B actief, glm5next)

- Gemeten: REAP50 IQ3_M op het V100-paar 0,843 bij 16,4 t/s; op vier GPU's 0,807 bij 22,4 t/s; IQ2_XXS 0,602.
- NVFP4 was op 2026-09-21 geblokkeerd: de FP4-MoE-kernels en de DeepGEMM-indexer vereisen Blackwell/Hopper.
- Haalbaarheid (`~/backlog/GLM53_FLASH_FEASIBILITY_20261005.md`): NVFP4 TP2 kan niet (≥195 GB gewichten > 144 GB VRAM).
  De beste werkbare variant is REAP50 IQ4_XS (88 GB, sha256 begint met 39562e8a), volledig in VRAM.
- IQ4_XS-opstelling: llama.cpp layer split over V100 CUDA 3,4 + Ada CUDA 2,5. De split 29,29,12,12 gaf een OOM
  (32,7 GB op één V100); 26,26,14,14 laadt (V100's 27,0/30,1 GB, Ada's 15–19 GB).
- Live gemeten tijdens de suite: prompt ~250 t/s, generatie ~23,6 t/s.
- Resultaat los (2026-10-05, 46 min): GSM8K 0,92, HumanEval 1,00, MMLU 0,69, BBH 0,917 → composite 0,881
  (mom-live-4: 0,885). Hoogste lokale HumanEval en BBH tot nu toe; MMLU is het zwakke punt. Generatie 23,6 t/s.

### Kimi K2.5 (MoE, DeepSeek-V3-architectuur; ~1T totaal)

Beschikbaar op Hugging Face (2026-10-05): `moonshotai/Kimi-K2.5`, `nvidia/Kimi-K2.5-NVFP4`,
`kwanhee/Kimi-K2.5-REAP50-NVFP4-W4A4-GS16`, `syntheticlab/Kimi-K2.5-NVFP4A16`, `amd/Kimi-K2.5-MXFP4`,
`unsloth/Kimi-K2.5-GGUF`, `bartowski/moonshotai_Kimi-K2.5-GGUF`. Al op schijf: Kimi-K2.7-Code UD-Q4_K_XL (544 GB)
(compleet, 14/14 shards) onder `/media/knight2/claude-data/knight1/eds1/model/`. Kimi-K3 UD-Q4_K_XL is
onvolledig (17/32 shards, 730 GB) en heeft een dubbele deelkopie van 9 shards (365 GB) in `eds1/model/UD-Q4_K_XL`.

**Opslagblokkade (2026-10-05):** claude-data is 99 % vol (40 GB vrij), EDS2 heeft 232 GB vrij. K2.5 Q3 (~450 GB)
past nergens zonder opruimen. K2.7 Code Q4 (544 GB) is groter dan RAM (503 GB) + VRAM (144 GB) samen en zou
van schijf pagen; niet werkbaar. Verwijderen gebeurt alleen met akkoord van de eigenaar.

Wat de lessen hierboven betekenen voor K2.5 (te verifiëren, nog niet getest):
- **NVFP4 W4A4 (FP4-activaties) op V100/Ada:** zelfde risico als GLM: FP4-MoE-kernels zijn Blackwell-gericht.
  NVFP4A16 (alleen gewichten in FP4) heeft een betere kans op de Ada-kaarten (sm89) dan op de V100's (sm70).
- **Geheugen:** een volledig ~1T-model past niet in 144 GB VRAM. Realistisch is GGUF Q3/Q4 met llama.cpp: attention
  en gedeelde lagen op de GPU's, experts in het RAM (503 GB totaal, ~370 GB beschikbaar) (`-ot` expert-offload). Een REAP50-snoei halveert dit.
- **Meetplan:** (1) Q3-GGUF met expert-offload, los: suite + specialist + audit + energie; (2) Q4 als het past;
  (3) NVFP4A16 op de vier Ada-kaarten als vLLM die route ondersteunt; (4) de beste variant als aggregator in de mix
  en vergelijken met `mom-live-4` op hetzelfde suitedeel (`--phase-since/--phase-until`).
