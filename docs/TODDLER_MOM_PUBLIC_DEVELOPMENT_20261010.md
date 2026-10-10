# Toddler modelmix: openbare ontwikkelproef, 10 oktober 2026

Deze proef is uitsluitend route-onderzoek. Zij vervangt geen afgeschermde specialist-, anti-contaminatie- of Haiku-vergelijking en mag geen generatie promoveren. Het 90-item-pack werd na de lokale gewichtbestanden met een verse 128-bit seed gegenereerd: 30 rekenvragen, 30 Python-code-traces en 30 gestructureerde extracties. Elk item heeft een nieuwe 96-bit nonce. Daardoor zijn de exacte prompts nieuw ten opzichte van deze bevroren gewichten; bekende taakvormen kunnen nog steeds een leereffect uit pretraining hebben.

Alle drie de GGUF's draaiden op één V100 per run met llama.cpp, temperatuur 0, één parallelle slot, maximaal 80 outputtokens en een afgedwongen JSON-antwoordveld. Een afzonderlijke 256-token-proef gaf de decode-tokens/s. `nvidia-smi` werd tijdens elke 30-vragenblok herhaald bevraagd; de geïntegreerde GPU-board-Wh per antwoord omvat board-idle tijdens de bloktijd, maar geen CPU, RAM of rest van het systeem. De ruwe antwoorden en de antwoordsleutel staan in de openbare auditbestanden. De exporter herberekent iedere score uit die antwoorden voordat hij de Toddler-export schrijft.

De eerste 48-token-pilot en de latere pack-v2/v3 pilots zijn ongeldig voor vergelijking: open uitleg werd afgekapt of de moeilijkheid was slecht gespreid. Alleen pack-v4 is de huidige openbare ontwikkelproef. Ook deze pack is tijdens canary en ontwikkelwerk bekeken; een volgende claim vereist een nieuwe, elders beheerde blinde toets.

| Model | Rekenen | Code-trace | Extractie | 256-token decode | Resident VRAM |
|---|---:|---:|---:|---:|---:|
| Qwen3-Coder-30B-A3B Q4_K_M | 19/30 | 6/30 | 30/30 | circa 132 t/s | circa 18 GB |
| Qwen3.8-35B-A3B distill Q4_K_M | 20/30 | 8/30 | 30/30 | circa 107 t/s | circa 20 GB |
| Qwen3-30B-A3B-Instruct-2507 Q4_K_M | 20/30 | 8/30 | 30/30 | circa 128 t/s | circa 18 GB |

De beste enkele modelscore is 0,6444. Een vaste taakroute wint hier nog geen kwaliteit; de perfecte per-item-router komt maximaal op 0,6889 bij maximaal drie resident modellen en 80 GB VRAM. Het gat van 0,0445 is een **optimistisch ontwikkelplafond**: de router kent dan het juiste antwoord op elk item vooraf. Drie resident modellen gebruiken samen circa 56 GB VRAM; de extra idle-kosten zijn niet in een end-to-end routermeting meegenomen. Dit is dus geen argument om nu een drie-modelroute in de dagdienst te zetten. Een voorafgaande run gaf Qwen3-30B op één code-item 9/30 in plaats van 8/30 ondanks temperatuur 0; die itemvariatie is zichtbaar als onzekerheid, niet als een extra onafhankelijke toets.

De operationele export staat op `reports/toddler_mom_public_dev.json` met schema `toddler-mom-public-dev/v1`. Teacher kan daarop uitsluitend een onderzoeksroute zoeken. Alle toekomstige kwaliteits- en energieclaims vergen een nieuw, verzegeld pack, een vaste router zonder antwoordtoegang en gelijktijdig gemeten systeemcontext.
