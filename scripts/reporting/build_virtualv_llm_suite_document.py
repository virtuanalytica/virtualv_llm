#!/usr/bin/env python3
"""Build the controlled VirtualV LLM suite standard with a live result appendix."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import markdown
from weasyprint import HTML


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/VIRTUALV_LLM_TESTSUITE_STANDARD.md"
RESULTS = ROOT / "reports/well_known_suite_20260917.json"
DEFAULT_OUTPUT = ROOT / "output/pdf/VirtualV_LLM_Testsuite_Standaard_en_Handleiding_2026-09-22.pdf"
PROTOCOL = "v4-mmlu-fewshot-20260918"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def metric(row: dict, group: str, key: str) -> float | None:
    value = (row.get(group) or {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def gsm8k_score(row: dict) -> float | None:
    """Mirror rank_models.py: retain the best valid exact-match extractor."""
    values = [
        float(value)
        for key, value in (row.get("gsm8k") or {}).items()
        if key.startswith("exact_match")
        and "stderr" not in key
        and isinstance(value, (int, float))
    ]
    return max(values) if values else None


def components(row: dict) -> tuple[float | None, ...]:
    return (
        gsm8k_score(row),
        metric(row, "bbh", "mean_accuracy"),
        metric(row, "mmlu_sample", "mean_accuracy"),
        metric(row, "humaneval", "pass_at_1"),
    )


def composite(row: dict) -> float | None:
    values = components(row)
    if row.get("error") or row.get("eval_protocol") != PROTOCOL:
        return None
    return sum(values) / 4 if all(value is not None for value in values) else None


def percentage(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def throughput(value: object) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "—"


def is_mixture(row: dict) -> bool:
    return str(row.get("engine", "")).startswith("mixture(") or str(row.get("model", "")).startswith("mixture")


def topology(row: dict) -> str:
    raw = str(row.get("topology") or row.get("hardware_profile") or "niet vastgelegd")
    raw = raw.replace("physical GPU [1, 2], split=layer", "2×V100 · layer")
    raw = raw.replace("physical GPU [0], split=none", "RTX A4000")
    raw = raw.replace("physical GPU [0, 1, 2, 3], split=layer", "4 GPU's · layer")
    return raw


def engine(row: dict) -> str:
    return str(row.get("engine") or "llama.cpp (historische rij)")


def table_for_rows(rows: list[dict]) -> str:
    body: list[str] = []
    for index, row in enumerate(rows, 1):
        gsm, bbh, mmlu, humaneval = components(row)
        body.append(
            "<tr>"
            f"<td class='num rank'>{index}</td><td class='model'>{esc(row.get('model', '—'))}</td>"
            f"<td>{esc(engine(row))}</td><td>{esc(topology(row))}</td>"
            f"<td class='num'>{percentage(gsm)}</td><td class='num'>{percentage(bbh)}</td>"
            f"<td class='num'>{percentage(mmlu)}</td><td class='num'>{percentage(humaneval)}</td>"
            f"<td class='num strong'>{percentage(composite(row))}</td>"
            f"<td class='num'>{throughput(row.get('completion_tokens_per_second'))}</td>"
            "</tr>"
        )
    return (
        "<table class='results'><thead><tr><th>#</th><th>Model / configuratie</th>"
        "<th>Engine</th><th>Hardwareprofiel</th><th>GSM8K</th><th>BBH</th>"
        "<th>MMLU</th><th>HumanEval</th><th>Composiet</th><th>tok/s</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )


def failure_explanation(row: dict) -> str:
    name = str(row.get("model", ""))
    error = str(row.get("error") or "Onvolledige rij")
    if "merlin-w4a16" in name:
        return (
            "De specifieke 1Cat-vLLM compressed-tensors/Merlin-route stopte vóór de suite. "
            "Dit bewijst niet dat iedere W4A16-route of 1Cat-branch op SM70 onmogelijk is."
        )
    if name == "glm45-air-106b-q4":
        return (
            "Past niet als gewone 50/50 V100-layer-split. Met zware CPU-offload werd 7,57 tok/s "
            "gemeten, onder de ingestelde snelheidsdrempel; daarom bestaat geen composietscore."
        )
    if name == "glm53-reap50-iq3m-v100":
        return "llama-server stopte tijdens initialisatie; de fout is bewaard en er is geen score afgeleid."
    return re.sub(r"/media/[^ ]+", "[lokaal logpad]", error)[:360]


def active_snapshot() -> tuple[str, str]:
    status = subprocess.run(
        ["systemctl", "--user", "is-active", "qwen38-flash-next-gguf-cascade.service"],
        capture_output=True, text=True, check=False,
    ).stdout.strip() or "onbekend"
    processes = subprocess.run(
        ["ps", "-eo", "cmd"], capture_output=True, text=True, check=False
    ).stdout
    matches = re.findall(r"well_known_suite\.py\s+([^\s]+)", processes)
    active = next((name for name in matches if name != "([^\\s]+)"), None)
    return status, active or "geen actieve well_known_suite-run gedetecteerd"


def result_appendix(payload: dict, snapshot: datetime) -> str:
    rows = payload.get("results", [])
    complete = [row for row in rows if composite(row) is not None]
    direct = sorted((row for row in complete if not is_mixture(row)), key=composite, reverse=True)
    mixtures = sorted((row for row in complete if is_mixture(row)), key=composite, reverse=True)
    incomplete = [row for row in rows if composite(row) is None]
    service, active = active_snapshot()
    digest = hashlib.sha256(RESULTS.read_bytes()).hexdigest()

    top_direct = direct[0] if direct else None
    top_mix = mixtures[0] if mixtures else None
    speed_rows = [row for row in direct if isinstance(row.get("completion_tokens_per_second"), (int, float))]
    fastest = max(speed_rows, key=lambda row: row["completion_tokens_per_second"]) if speed_rows else None
    cards: list[tuple[str, str, str]] = []
    if top_direct:
        cards.append(("Beste individuele composiet", top_direct["model"], percentage(composite(top_direct))))
    if fastest:
        cards.append(("Hoogste gemeten decode", fastest["model"], f"{fastest['completion_tokens_per_second']:.2f} tok/s"))
    if top_mix:
        cards.append(("Beste mixture-composiet", top_mix["model"], percentage(composite(top_mix))))

    failures = "".join(
        "<tr>"
        f"<td class='model'>{esc(row.get('model', '—'))}</td><td>{esc(engine(row))}</td>"
        f"<td>{esc(topology(row))}</td><td>{esc(failure_explanation(row))}</td>"
        "</tr>" for row in incomplete
    ) or "<tr><td colspan='4'>Geen incomplete of mislukte rijen.</td></tr>"
    card_html = "".join(
        f'<div class="metric-card"><span>{esc(label)}</span><b>{esc(name)}</b><em>{esc(value)}</em></div>'
        for label, name, value in cards
    )

    return f"""
<section class="result-annex">
  <h1 id="bijlage-b">Bijlage B — Actuele benchmarkresultaten</h1>
  <div class="snapshot">
    <strong>Momentopname:</strong> {snapshot:%d-%m-%Y %H:%M:%S %Z} &nbsp;·&nbsp;
    <strong>Bron:</strong> reports/well_known_suite_20260917.json &nbsp;·&nbsp;
    <strong>SHA-256:</strong> <code>{digest}</code><br>
    <strong>Omvang:</strong> {len(rows)} rijen; {len(direct)} individuele huidige-volledige resultaten;
    {len(mixtures)} huidige-volledige mixtures; {len(incomplete)} niet-rangschikbare rijen.<br>
    <strong>Cascade:</strong> service {esc(service)}; actieve fase bij opmaak: {esc(active)}.
  </div>
  <div class="cards">{card_html}</div>
  <div class="note"><strong>Leeswijzer.</strong> Composiet is het ongewogen gemiddelde van GSM8K, BBH,
  MMLU en HumanEval. Tok/s is contextgebonden en niet in de composiet opgenomen. Een mixture kan
  meerdere modelaanroepen en dus andere kosten/latency hebben; daarom staat die apart.</div>
  <h2>B.1 Individuele modellen en runtimes</h2>
  {table_for_rows(direct)}
  <p class="caption">Rangschikking binnen de actuele individuele huidige-volledige rijen. Een score zegt
  niets over licentie, totale gebruikskosten of geschiktheid voor een specifieke productieflow.</p>
  <h2>B.2 Mixtures en orkestratie</h2>
  {table_for_rows(mixtures)}
  <p class="caption">Mixture-resultaten gebruiken dezelfde vier kwaliteitscomponenten, maar vormen geen
  één-op-één snelheidsvergelijking met één checkpoint. Ontbrekende tok/s wordt als — getoond.</p>
  <h2>B.3 Niet-rangschikbare, mislukte of partiële rijen</h2>
  <table class="failures"><thead><tr><th>Model</th><th>Engine</th><th>Hardware</th><th>Toelichting</th></tr></thead>
  <tbody>{failures}</tbody></table>
  <h2>B.4 Zakelijke duiding</h2>
  <ul>
    <li>Gebruik de composiet als brede kwaliteitsindicator, niet als zelfstandig productiebesluit.</li>
    <li>Bekijk voor capaciteit alleen runs met dezelfde engine, quant, context en hardwaretopologie.</li>
    <li>De 1Cat-vLLM-rijen en llama.cpp/GGUF-rijen zijn verschillende uitvoeringspaden.</li>
    <li>Een fout- of unsupported-rij blijft zichtbaar als auditbewijs en krijgt geen fictieve nul- of snelheidsscore.</li>
    <li>Een actieve run komt pas in de tabel na atomaire voltooiing en een nieuwe PDF-build.</li>
  </ul>
</section>
"""


CSS = r"""
@page {
  size: A4; margin: 19mm 17mm 18mm 17mm;
  @top-left { content: "VIRTUALV  |  AI ASSURANCE"; color: #12344d; font-size: 8pt; font-weight: 700; letter-spacing: .08em; }
  @top-right { content: string(section); color: #607587; font-size: 7.5pt; }
  @bottom-left { content: "INTERN GEBRUIK  ·  V1.0  ·  22 SEPTEMBER 2026"; color: #718392; font-size: 7pt; }
  @bottom-right { content: "PAGINA " counter(page) " / " counter(pages); color: #718392; font-size: 7pt; }
}
@page cover { size: A4; margin: 0; @top-left { content: none; } @top-right { content: none; } @bottom-left { content: none; } @bottom-right { content: none; } }
@page results { size: A4 landscape; margin: 15mm 13mm; }
* { box-sizing: border-box; }
body { font-family: "DejaVu Sans", sans-serif; color: #172b3a; font-size: 9.4pt; line-height: 1.48; margin: 0; }
.cover { page: cover; height: 297mm; padding: 29mm 25mm 22mm; color: white; background: linear-gradient(145deg, #0d2a3f 0%, #164d63 62%, #16817d 100%); position: relative; page-break-after: always; }
.brand { font-size: 15pt; font-weight: 700; letter-spacing: .16em; border-left: 5px solid #efbd57; padding-left: 12px; }
.cover h1 { font-size: 32pt; line-height: 1.08; margin: 49mm 0 8mm; max-width: 150mm; color: white; border: 0; page-break-before: auto; }
.cover .subtitle { font-size: 15pt; color: #d6eef0; max-width: 145mm; }
.cover .rule { width: 45mm; height: 3px; background: #efbd57; margin: 12mm 0; }
.cover-meta { position: absolute; left: 25mm; right: 25mm; bottom: 27mm; display: grid; grid-template-columns: 1fr 1fr; gap: 5mm 18mm; padding-top: 8mm; border-top: 1px solid rgba(255,255,255,.35); font-size: 9pt; }
.cover-meta span { display: block; color: #a7d6db; font-size: 7.5pt; text-transform: uppercase; letter-spacing: .08em; }
.toc { page-break-after: always; }
h1 { color: #12344d; font-size: 20pt; line-height: 1.15; margin: 0 0 8mm; padding-bottom: 3mm; border-bottom: 2px solid #1a8583; string-set: section content(); page-break-before: always; }
.toc h1, article > h2:first-child { page-break-before: auto; }
h2 { color: #126b70; font-size: 13.5pt; line-height: 1.2; margin: 8mm 0 3mm; page-break-after: avoid; string-set: section content(); }
h3 { color: #12344d; font-size: 10.5pt; margin: 5mm 0 2mm; page-break-after: avoid; }
p { margin: 0 0 3.2mm; }
ul, ol { margin: 1.5mm 0 4mm 5.5mm; padding-left: 4mm; }
li { margin: 0 0 1.3mm; }
strong { color: #102f45; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: .88em; background: #edf3f5; padding: .2em .35em; border-radius: 2px; overflow-wrap: anywhere; }
pre { background: #102f45; color: #edf7f7; border-left: 4px solid #1a8583; padding: 3.3mm 4mm; font-family: "DejaVu Sans Mono", monospace; font-size: 7.2pt; line-height: 1.4; white-space: pre-wrap; overflow-wrap: anywhere; page-break-inside: avoid; }
pre code { background: transparent; color: inherit; padding: 0; }
table { width: 100%; border-collapse: collapse; margin: 3mm 0 5mm; font-size: 8pt; }
thead { display: table-header-group; }
th { background: #123e56; color: white; text-align: left; padding: 2.2mm 2mm; font-weight: 700; }
td { border-bottom: 1px solid #d8e2e7; padding: 2mm; vertical-align: top; }
tbody tr:nth-child(even) { background: #f3f7f8; }
tr { page-break-inside: avoid; }
.toc { padding-top: 4mm; }
.toc-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 5mm 10mm; }
.toc a { display: block; text-decoration: none; color: #12344d; border-bottom: 1px dotted #91a5b1; padding: 2.3mm 0; }
.toc a span { color: #16817d; font-weight: 700; margin-right: 2mm; }
.doc-control { margin-top: 10mm; background: #f3f7f8; border: 1px solid #d5e1e6; padding: 5mm; }
.result-annex { page: results; }
.snapshot { background: #e9f4f3; border-left: 5px solid #16817d; padding: 4mm 5mm; margin-bottom: 5mm; font-size: 8.3pt; }
.snapshot code { font-size: 6.7pt; }
.cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 4mm; margin: 4mm 0 5mm; }
.metric-card { border: 1px solid #d5e1e6; border-top: 4px solid #efbd57; padding: 3.3mm; background: white; }
.metric-card span, .metric-card b, .metric-card em { display: block; }
.metric-card span { color: #607587; font-size: 7pt; text-transform: uppercase; letter-spacing: .05em; }
.metric-card b { margin: 1.2mm 0; color: #12344d; font-size: 9pt; }
.metric-card em { color: #16817d; font-style: normal; font-weight: 700; font-size: 12pt; }
.note { background: #fff8e8; border-left: 4px solid #efbd57; padding: 3mm 4mm; margin: 4mm 0 6mm; font-size: 8.2pt; }
table.results { font-size: 6.2pt; line-height: 1.2; table-layout: fixed; }
table.results th, table.results td { padding: 1.45mm 1.2mm; overflow-wrap: anywhere; }
table.results th:nth-child(1) { width: 5mm; }
table.results th:nth-child(2) { width: 45mm; }
table.results th:nth-child(3) { width: 34mm; }
table.results th:nth-child(4) { width: 49mm; }
table.results .num { text-align: right; white-space: nowrap; }
table.results .model { font-weight: 700; color: #12344d; }
.strong { color: #126b70; font-weight: 700; }
.rank { color: #607587; }
.caption { color: #607587; font-size: 7.2pt; margin-top: -3mm; }
table.failures { font-size: 7.1pt; table-layout: fixed; }
table.failures th:nth-child(1) { width: 52mm; }
table.failures th:nth-child(2) { width: 36mm; }
table.failures th:nth-child(3) { width: 55mm; }
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = json.loads(RESULTS.read_text())
    snapshot = datetime.now(ZoneInfo("Europe/Amsterdam"))
    source = SOURCE.read_text()
    source = source[source.index("## 1. Doel en reikwijdte"):]
    source = source.split("# Bijlage B — Actuele benchmarkresultaten", 1)[0]
    article = markdown.markdown(source, extensions=["extra", "sane_lists", "toc"])
    cover = """
<section class="cover">
  <div class="brand">VIRTUALV</div>
  <h1>LLM-testsuite<br>standaard &amp; proces</h1>
  <div class="subtitle">Governance, meetprotocol, operationele instructie en actuele benchmarkresultaten</div>
  <div class="rule"></div>
  <div class="cover-meta">
    <div><span>Documenteigenaar</span>VirtualV AI Assurance</div><div><span>Versie</span>1.0</div>
    <div><span>Peildatum</span>22 september 2026</div><div><span>Classificatie</span>Intern gebruik</div>
    <div><span>Status</span>Operationele standaard van de huidige implementatie</div>
    <div><span>Bijlagen</span>A · Instructiehandleiding &nbsp; B · Resultaten</div>
  </div>
</section>"""
    toc = """
<section class="toc"><h1>Documentoverzicht</h1>
<p>Dit gecontroleerde document combineert de norm, de werkinstructie en een gedateerde bewijsbijlage.</p>
<div class="toc-grid">
  <div><a href="#1-doel-en-reikwijdte"><span>01</span>Doel en reikwijdte</a><a href="#2-kernprincipes"><span>02</span>Kernprincipes</a><a href="#3-begrippen-en-statussen"><span>03</span>Begrippen en statussen</a><a href="#4-hardware-en-runtimeprofielen"><span>04</span>Hardware en runtimes</a><a href="#5-algemene-kwaliteitssuite"><span>05</span>Algemene kwaliteitssuite</a><a href="#6-throughputmeting"><span>06</span>Throughput</a></div>
  <div><a href="#7-specialistensuite-en-toegangsprofielen"><span>07</span>Specialisten en toegang</a><a href="#8-bewijs-en-herkomstcontract"><span>08</span>Bewijscontract</a><a href="#9-operationeel-proces"><span>09</span>Operationeel proces</a><a href="#10-wijzigings-en-acceptatiebeleid"><span>10</span>Wijzigingsbeleid</a><a href="#bijlage-a-instructiehandleiding-voor-operators"><span>A</span>Instructiehandleiding</a><a href="#bijlage-b"><span>B</span>Actuele resultaten</a></div>
</div>
<div class="doc-control"><strong>Bronhiërarchie.</strong> De resultaat-JSON en ruwe logs zijn primair bewijs. Deze PDF is de bestuurlijk leesbare momentopname. Bij verschil prevaleert het herleidbare bronbewijs.</div>
</section>"""
    document = f"""<!doctype html><html lang="nl"><head><meta charset="utf-8">
<title>VirtualV LLM-testsuite — standaard en proces</title>
<meta name="author" content="VirtualV AI Assurance"><meta name="description" content="Operationele standaard, operatorhandleiding en actuele resultaten"><style>{CSS}</style></head>
<body>{cover}{toc}<article>{article}</article>{result_appendix(payload, snapshot)}</body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=document, base_url=str(ROOT)).write_pdf(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
