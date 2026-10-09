# Dashboardcontrole: Haiku 5.5 en 1Cat-vLLM TP2

Peildatum: 9 oktober 2026. Inventaris van alle 30 tabellen in
`reports/dual_v100_nvlink_benchmark.html`, in weergavevolgorde. Een cijfer
komt alleen in een bestaande tabel als taak, meeteenheid en toegangsniveau
passen. De acht vaste taken zijn openbaar en dienen voor regressiediagnose;
promotie gebruikt de afgeschermde specialist- en anti-contaminatieproeven.

| # | Tabel | Toevoeging of besluit |
|---:|---|---|
| 1 | Alle lokale resultaten | 1Cat 1.5.0 en 1.5.1 B1 256-token wall-t/s en 1.5.1 E4M3 B1/B4/B16 als afzonderlijke profielen. RIV komt alleen van de gelijknamige openbare taak. VRAM en GPU-utilisatie blijven leeg omdat deze proef ze niet per batch registreerde. Haiku is cloud en hoort hier niet. |
| 2 | Acht vaste taken | Drie versie/configuratierijen uit uitsluitend numerieke scores van dezelfde acht openbare taken; de exporter verwijdert antwoorden en rubricdetails. v1.5.1 repareert de codetaak (0 naar 1), maar RIV zakt van 6/6 naar 5/6; rekenen blijft 0. |
| 3 | Publieke sandbox-suite | Historisch archief. Geen nieuwe volledige publieke run. |
| 4 | Statistische betrouwbaarheid publieke ranglijst | Historische ranglijst; nieuwe private rijen hebben andere onzekerheidsberekening. |
| 5 | Contaminatiebestendige composites | Oudere broncombinatie; de nieuwe tien-lane scores blijven in tabel 7 tot de protocollen zijn geharmoniseerd. |
| 6 | Oudere inhoudsspecialisten | Per-lane 1Cat-scores zijn niet in het huidige geaggregeerde TP2-rapport bewaard. De 10-lane totaalwaarde staat in tabel 7; geen lege cellen met een composite vullen. |
| 7 | Private benchmarkaggregaten | Vijf afgeronde modellen/configuraties, acht taken, tien-lane composite, post-cutoff, stabiliteitscanary, 256-token-t/s en GPU-board-Wh per antwoord. V1.5.0: 3/9 canary; beide 1.5.1-armen: 9/9. `promotion_eligible` betekent beoordelingsbaar, niet gepromoveerd. |
| 8 | Softwareontwikkeling | De gesloten 16-item pilot bevat Qwen, GLM, 1Cat 1.5.0 en Haiku. 1.5.1 krijgt geen toegang na afsluiting door Haiku; een nieuwe versie vereist een verse pack. |
| 9 | Data | Zelfde pack- en afsluitregel als tabel 8; Haiku 8/8 data, maar twee items per rol zijn geen bewijs voor een agentketen. |
| 10 | Haiku CLI B1/B4/B16 | Openbare synthetische proef: B1 157,0 output-t/s inclusief denktokens en circa 53,6 zichtbare t/s over de hele CLI-wandklok. B4 6/8 en B16 6/32; HTTP 429 bevestigd. Geen geldig B4/B16-t/s. |
| 11 | 1Cat 1.5.1 TP2 B1/B4/B16 | Openbare synthetische proef op twee V100's met E4M3: B1 48,4, B4 176,1 en B16 577,6 totale output-t/s, telkens inclusief prefill. Alle 42 gemeten verzoeken slaagden; warmup is uitgesloten. |
| 12 | Live MoM met twee llama.cpp-replica's | Bestaande MoM-run, niet dezelfde 1Cat-versie of cloud-CLI-proef. Alleen een nieuwe echte mixture-run kan deze tabel uitbreiden. |
| 13 | Live MoM met 1Cat TP2 | Historische mixture-run; vervang haar model/energiecijfers pas na een nieuwe volledig gemeten mixture. |
| 14 | Live MoM met Qwen3.6 TP2 | Ander model; geen 1Cat Qwen3.8- of Haiku-cijfers invoegen. |
| 15 | Live MoM met GLM | Ander aggregatorprofiel; geen nieuwe meting. |
| 16 | Live MoM met twee proposers | Bestaande configuratie; geen nieuwe meting. |
| 17 | Live MoM met één proposer | Bestaande configuratie; geen nieuwe meting. |
| 18 | Live MoM met Gemma-aggregator | Ander model en werkverdeling; geen nieuwe meting. |
| 19 | Live MoM met Qwen3.5-aggregator | Ander model en werkverdeling; geen nieuwe meting. |
| 20 | Vergelijking aggregatoropstellingen | Alleen gemeten volledige mixtures. Losse TP2- of Haiku-probes tellen niet als MoM-antwoord. |
| 21 | Contaminatie-audit per model | Oudere publieke en specialistbronnen. Nieuwe private auditaggregaten staan in tabel 7 zonder private vragen te publiceren. |
| 22 | Contaminatiemaatregelen | Methodetabel, geen modelscores. De afgesloten cloudpack en score-exportregel zijn in de protocolstukken beschreven. |
| 23 | Qwen3.6 op andere hardware | Ander model; een nieuwe GPU-topologie vereist een aparte gelijkwaardige hardwareproef. |
| 24 | GLM-hardwarematrix | Ander model; ongewijzigd. |
| 25 | GLM-quants | Ander model; ongewijzigd. |
| 26 | Grote-modelbronnen en bewaring | Opslag/provenance, geen TP2-snelheidsvergelijking. De 1Cat-wheelversie staat in de modelroadmap. |
| 27 | Flash-Next-vLLM-proefplanning | 27B QUASAR TP2 is een ander checkpoint; Flash-Next NVFP4 blijft een afzonderlijke, nog niet laadbare proef op dit V100-paar. |
| 28 | BetterBench-zelfbeoordeling | Meet de benchmarkopzet, niet een model; ongewijzigd. |
| 29 | EDSQ-Volta MoE-judges | Ander proces en andere metrieken; ongewijzigd. |
| 30 | Meetintegriteit | Hardware- en protocolnoot; fysieke V100-mapping is 3 en 4. |

De aanvullende korte proeven waren: negen openbare stabiliteitscanaries per
1Cat-configuratie, drie B1-Haiku-CLI-aanvragen plus B4/B16-pogingen, en
1Cat B1/B4/B16 met een afzonderlijke warmup en twee meetgolven. De volledige
private TP2-versievergelijking gebruikte hetzelfde 8K- en decodeerprofiel;
de E4M3-arm is als **andere configuratie** gelabeld. De Haiku-CLI-output omvat
denktokens en netwerk/CLI-tijd; de 1Cat-batchcijfers omvatten prefill en zijn
geen zuivere decode-t/s. Geen van deze doorvoercijfers bewijst antwoordkwaliteit.
