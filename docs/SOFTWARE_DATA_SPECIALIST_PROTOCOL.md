# Software- en dataspecialisten: private pilot en bewijsgrens

Status: 2026-10-09. Protocol `software-data-private-v1-20261009`.

## Twee aparte modeltabellen

De softwaretabel meet **coder, reviewer, architect en debugger**. De datatabel
meet **data engineer, data analyst, data architect en data steward**. Per model
worden de juiste items, het aantal items, de slaagkans en een Wilson-interval
per rol bewaard. Het rolgemiddelde is alleen een beschrijvende composite;
deze twee tabellen tellen niet mee in de bestaande tien-lane inhoudscomposiet.
De eerste pack bevat twee items per rol. Daarmee kan geen rangorde of 100%-claim
betrouwbaar worden vastgesteld.

Coder en debugger leveren werkende Python-functies die verborgen inputgevallen
moeten doorstaan. De kandidaatcode draait met een niet-root-gebruiker in een
kortlevende Docker-container zonder netwerk, met alleen een tijdelijke,
alleen-lezen werkmap, CPU-/geheugen-/procesgrenzen en een tijdslimiet. De
data-engineer levert één read-only SQLite-query, beoordeeld op meerdere
verborgen fixtures. Reviewer, architect, analyst en steward leveren strikt
gestructureerde antwoorden die een onafhankelijke rubric controleert. Een
antwoordfout telt als fout; een ontbrekende modelrespons maakt de run partieel.

## Afgeschermde toetsstof

De private pack staat buiten Git op een door de evaluator beheerde locatie,
met directorymodus 0700 en bestandmodus 0600. Alleen het SHA-256-commitment,
de rolverdeling, scorer en protocoldefinitie staan in de repository.
`software_data_suite.py` weigert een pack in de repository, een te ruim
leesbaar bestand, gewijzigde hash, dubbele item-ID of gewijzigde rolverdeling.
De evaluator stuurt alleen de prompt naar het lokale model; antwoorden,
verwachte uitkomsten en verborgen cases komen nooit in de modelprompt.
De prompt-verbeteraar krijgt uitsluitend een afzonderlijke oefenset en geen
pad of leestoegang tot deze pack via de benchmarkinterface. Roteer de private
pack vóór herhaald optimaliseren; de oude set wordt dan ontwikkelstof.

Er is daarnaast een **aparte cloud-toegestane pilotpack** met eigen SHA-256-
commitment (`reports/software_data_cloud_pack_commitment.json`). Die pack
wordt eerst door minstens twee lokale modellen beantwoord en daarna als laatste
door Haiku 5.5. De oorspronkelijke lokale pack wordt nooit aan de cloud
verstuurd. Scores van verschillende packhashes worden niet gerangschikt.

Dit is **afscherming tegen Git-publicatie en tegen het modelendpoint**, niet
een harde scheiding van alle processen onder dezelfde Unix-gebruiker. Een
agent met dezelfde hostrechten of Docker-toegang kan de private pack mogelijk
lezen. Voor een formele bewijsrun moet de pack bij een onafhankelijke evaluator
onder een andere identiteit of op een andere machine staan. De huidige pilot
kan daarom niet als definitief anti-contaminatiebewijs dienen.

De bestaande ClaudeClaw-promptverbeteraar heeft nog zijn eigen, hergebruikte
vragen. Zijn oude score mag niet worden gepresenteerd als blind resultaat.
Koppel hem pas aan deze methode als zijn training en evaluatie aantoonbaar
gescheiden zijn; de huidige private pack wordt niet aan die lus gekoppeld.

## De softwareclaim: Toddler + Teacher + agent op ClaudeClaw

De vraag *“is beter op softwaretaken bewezen?”* vergelijkt de volledige
**Toddler + Teacher + agent op ClaudeClaw**-keten met gewone ClaudeClaw-workers.
Losse modelscores in de twee tabellen zijn diagnose, geen bewijs voor de keten.
De oude 115 reviewbevindingen zijn één oorzaaktaxonomie zonder gepaarde
makergegevens en tellen evenmin als controlegroep.

Voor een positieve interne conclusie vereist `software_agent_comparison.py`:

1. Precies dezelfde afgeschermde pack, protocolversie en geregistreerde taken
   in beide armen, met onafhankelijk geverifieerde taakresultaten en hashes.
2. Minstens 73 onafhankelijke softwaretaken per rol, dus ten minste 292
   gepaarde taken in totaal; de pilot van twee per rol faalt deze poort.
3. Minstens drie procentpunt winst in het ongewogen gemiddelde van de vier
   softwaremakers; de onderste grens van een gestratificeerde gepaarde 95%-
   bootstrap ligt boven nul.
4. Geen rol verliest meer dan vijf procentpunt en er zijn niet meer fouten
   die na merge ontsnappen dan bij de controle.
5. Een onafhankelijk beheerde toetsset. De huidige pack met gedeelde
   hostidentiteit is alleen een pilot; de bewijsrun krijgt een apart
   evaluatoraccount of een aparte machine en een nieuwe, verse pack.

Zonder die voorwaarden luidt de dashboarduitkomst **“nee, nog niet”**.
Rapporteer ook het aantal reviewbevindingen, verworpen of herschreven patches,
doorlooptijd en rekentijd per arm als secundaire maten. Leg model, prompt,
Teacher-generatie, Toddler-afstamming, agentversie en ClaudeClaw-run-ID per
taak vast, zodat een verschil aan de maker kan worden gekoppeld.

## Uitvoering

Wacht met echte modelruns tot de huidige GLM Q4→Q3-benchmark en handoff klaar
zijn. Gebruik daarna voor **ieder model dezelfde pack en decodeerinstellingen**:

```bash
python3 scripts/benchmarks/software_data_suite.py \
  --pack "$PRIVATE_PACK_PATH" \
  --base http://127.0.0.1:PORT --alias MODEL_ALIAS --model-id MODEL_ID
```

Voor de vergelijkbare Haiku-reeks gebruiken beide lokale kandidaten dezelfde
cloud-toegestane pack en het eigen commitment. De Haiku-run weigert te starten
totdat twee complete lokale modelrijen met die hash zijn opgeslagen:

```bash
python3 scripts/benchmarks/software_data_suite.py \
  --pack "$PRIVATE_CLOUD_PACK_PATH" \
  --commitment reports/software_data_cloud_pack_commitment.json \
  --base http://127.0.0.1:PORT --alias MODEL_ALIAS --model-id MODEL_ID

python3 scripts/benchmarks/software_data_suite.py \
  --pack "$PRIVATE_CLOUD_PACK_PATH" \
  --commitment reports/software_data_cloud_pack_commitment.json \
  --provider haiku55 --effort medium --model-id claude-haiku-5-5-medium
```

Haiku 5.5 gebruikt de geïsoleerde Claude CLI zonder tools, met adaptive
thinking en expliciet effort. De CLI kan het lokale max-tokenbudget en
temperature niet op dezelfde manier afdwingen; behandel die resultaten als
een ander runtimeprofiel. Cloud-GPU-energie is onbekend.
De [officiële Haiku-handleiding](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-haiku-5-5)
bevestigt dat `medium` de standaard effort is en thinking standaard aan staat.

De resultaten komen in `reports/software_data_specialists.json`; het
dashboard toont aparte software- en datatabellen. Een gepaard agentexperiment
wordt pas na onafhankelijke verificatie ingevoerd met
`scripts/benchmarks/software_agent_comparison.py --candidate ... --baseline ...
--commitment ...`. Dat commitment moet vóór de eerste respons zijn vastgelegd
door de onafhankelijke evaluator met status `sealed-independent`; het huidige
`sealed-pilot`-commitment kan geen positieve uitkomst opleveren.
Het script publiceert expliciet een negatieve/onbesliste uitkomst zolang de
poort niet is gehaald. Een publieke full suite wordt hiervoor niet gebruikt.
