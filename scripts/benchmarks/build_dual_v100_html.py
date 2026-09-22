#!/usr/bin/env python3
"""Build the self-contained dual-V100 benchmark page from raw JSON evidence."""

from __future__ import annotations

import html
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import well_known_suite as wks  # noqa: E402
import rank_models as rm  # noqa: E402
import benchmark_phase_gate as bpg  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
OUT = Path(os.environ.get("VIRTUALV_DASHBOARD_OUT", REPORTS / "dual_v100_nvlink_benchmark.html"))


def load(name: str) -> dict:
    path = REPORTS / name
    return json.loads(path.read_text()) if path.exists() else {}


def fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def gguf_rows() -> list[dict]:
    source = []
    for name in (
        "qwen38_nvlink_profiles_20260917.json",
        "local_gguf_dual_v100_tensor_20260917.json",
        "qwen25_72b_nvlink_profiles_20260917.json",
    ):
        source.extend(load(name).get("results", []))
    labels = {
        "qwen38-27b-q4": ("Qwen3.8-27B", "27B", "Q4_K_M"),
        "deepseek-r1-qwen32b-q4": ("DeepSeek-R1-Qwen", "32B", "Q4_K_M"),
        "qwen36-27b-iq3": ("Qwen3.6-27B", "27B", "IQ3_XXS"),
        "qwen35-27b-q4": ("Qwen3.5-27B", "27B", "Q4_K_M"),
        "gemma4-26b-a4b-q4": ("Gemma4-26B-A4B", "26B / ~4B active", "Q4_K_M"),
        "devstral-small2-24b-q4": ("Devstral Small 2", "24B", "Q4_K_M"),
        "qwen25-72b-q4": ("Qwen2.5-72B", "72.7B dense", "Q4_K_M"),
    }
    rows = []
    qwen_single = next((r.get("completion_tokens_per_second") for r in source
                        if r.get("model") == "qwen38-27b-q4" and r.get("profile") == "single-v100"), None)
    for row in source:
        if row.get("error"):
            continue
        label, params, quant = labels.get(row["model"], (row["model"], "—", "—"))
        gpu = row.get("gpu_decode_telemetry", {})
        selected = ["1"] if row.get("profile") == "single-v100" else ["1", "2"]
        memory = [gpu.get(i, {}).get("max_memory_mib") for i in selected]
        util = [gpu.get(i, {}).get("mean_util_pct") for i in selected]
        tps = row.get("completion_tokens_per_second")
        speedup = (tps / qwen_single) if qwen_single and row.get("model") == "qwen38-27b-q4" else None
        rows.append({
            "model": label, "params": params, "engine": "llama.cpp", "quant": quant,
            "profile": row.get("profile", "—"), "tps": tps,
            "prompt_tps": row.get("prompt_tokens_per_second"), "speedup": speedup,
            "quality": f"{row.get('riv_au_score', '—')}/{row.get('riv_au_max', 6)}",
            "memory": " / ".join(fmt(v / 1024) for v in memory if v is not None),
            "util": " / ".join(fmt(v, 1) + "%" for v in util if v is not None),
            "scope": "single" if len(selected) == 1 else "dual",
            "gpu_caption": "GPU 1 · Tesla V100-SXM2-32GB" if len(selected) == 1
            else "GPU 1 + 2 · 2× Tesla V100-SXM2-32GB · NVLink",
        })
    return rows


TASK_LABELS = {
    "riv_au_lifecycle": "RIV",
    "arithmetic_payout": "Arith",
    "code_exec_drawdown": "Code",
    "json_schema_facts": "JSON",
    "gics_format_following": "GICS",
    "needle_in_haystack": "Needle",
    "summary_faithfulness": "Summary",
    "refusal_calibration": "Refusal",
}
TASK_ORDER = list(TASK_LABELS)

BENCHMARK_LABELS = {
    "qwen38-27b-q4": ("Qwen3.8-27B", "llama.cpp"),
    "deepseek-r1-qwen32b-q4": ("DeepSeek-R1-Qwen", "llama.cpp"),
    "qwen36-27b-iq3": ("Qwen3.6-27B", "llama.cpp"),
    "qwen35-27b-q4": ("Qwen3.5-27B", "llama.cpp"),
    "gemma4-26b-a4b-q4": ("Gemma4-26B-A4B", "llama.cpp"),
    "devstral-small2-24b-q4": ("Devstral Small 2", "llama.cpp"),
    "qwen25-72b-q4": ("Qwen2.5-72B", "llama.cpp"),
    "qwen38-target": ("Qwen3.8 target", "1Cat-vLLM"),
    "qwen38-dflash2": ("Qwen3.8 + DFlash2", "1Cat-vLLM"),
}


def benchmark_score_rows() -> list[dict]:
    rows = []
    for r in load("local_gguf_8bench_dual_v100_20260917.json").get("results", []):
        if r.get("error") or "benchmarks" not in r:
            continue
        label, engine = BENCHMARK_LABELS.get(r["model"], (r["model"], "llama.cpp"))
        rows.append({"model": label, "engine": engine, "benchmarks": r["benchmarks"]})
    for filename, key in (
        ("1cat_target_tp2_8bench_20260917.json", "qwen38-target"),
        ("1cat_dflash2_tp2_8bench_20260917.json", "qwen38-dflash2"),
    ):
        r = load(filename)
        if not r or "benchmarks" not in r:
            continue
        label, engine = BENCHMARK_LABELS[key]
        rows.append({"model": label, "engine": engine, "benchmarks": r["benchmarks"]})
    return rows


def score_cell(score: float | None) -> str:
    if score is None:
        return "<td class='num'>—</td>"
    cls = "score-hi" if score >= 0.75 else ("score-mid" if score >= 0.4 else "score-lo")
    return f"<td class='num {cls}'>{round(100 * score)}%</td>"


def benchmark_table(rows: list[dict]) -> str:
    body = []
    for row in rows:
        cells = [f"<td><strong>{html.escape(row['model'])}</strong><small>{html.escape(row['engine'])}</small></td>"]
        for task_id in TASK_ORDER:
            entry = row["benchmarks"].get(task_id)
            cells.append(score_cell(entry["score"] if entry else None))
        mean = row["benchmarks"].get("_mean_score")
        cells.append(score_cell(mean))
        body.append("<tr>" + "".join(cells) + "</tr>")
    return "".join(body)


def onecat_rows() -> list[dict]:
    specs = [
        ("1cat_target_tp2_8k_verified_20260917.json", "Qwen3.8 target", "B1"),
        ("1cat_dflash2_tp2_8k_verified_20260917.json", "Qwen3.8 + DFlash2", "B1"),
        ("1cat_target_tp2_8k_b4_20260917.json", "Qwen3.8 target", "B4"),
        ("1cat_dflash2_tp2_8k_b4_20260917.json", "Qwen3.8 + DFlash2", "B4"),
    ]
    rows = []
    for filename, label, batch in specs:
        row = load(filename)
        if not row:
            continue
        gpu = row.get("gpu_telemetry", {})
        rows.append({
            "model": label, "params": "27B", "engine": "1Cat-vLLM 1.5", "quant": "NVFP4",
            "profile": f"TP2 / 8K / {batch}", "tps": row.get("wall_output_tokens_per_second"),
            "prompt_tps": None, "speedup": None,
            "quality": f"{row.get('riv_au_score', '—')}/{row.get('riv_au_max', 6)}",
            "memory": " / ".join(fmt(gpu.get(i, {}).get("max_memory_mib", 0) / 1024) for i in ("1", "2")),
            "util": " / ".join(fmt(gpu.get(i, {}).get("max_util_pct"), 0) + "%" for i in ("1", "2")),
            "scope": "dual",
            "gpu_caption": "GPU 1 + 2 · 2× Tesla V100-SXM2-32GB · NVLink",
        })
    return rows


def table(rows: list[dict]) -> str:
    body = []
    for row in rows:
        speedup = "—" if row["speedup"] is None else f"{row['speedup']:.2f}×"
        # 2026-09-22 (user feedback): "als we op het model klikken willen we de
        # uitleg" -- only the 1Cat-vLLM rows (target vs. +DFlash2, B1 vs. B4)
        # have a matching explanation section; GGUF rows keep a plain label.
        model_cell = (f'<td><a class="benchmark-link" href="#onecat-uitleg" '
                      f'title="Ga naar uitleg van {html.escape(row["model"])}">'
                      f'<strong>{html.escape(row["model"])}</strong></a>'
                      f'<small>{html.escape(row["params"])}</small></td>'
                      if row["engine"] == "1Cat-vLLM 1.5" else
                      f'<td><strong>{html.escape(row["model"])}</strong><small>{html.escape(row["params"])}</small></td>')
        body.append("<tr>" + "".join([
            model_cell,
            f"<td>{html.escape(row['engine'])}<small>{html.escape(row['quant'])}</small></td>",
            f"<td><span class='pill'>{html.escape(row['profile'])}</span></td>",
            f"<td class='num hot'>{fmt(row['tps'])}</td>",
            f"<td class='num'>{fmt(row['prompt_tps'])}</td>",
            f"<td class='num'>{speedup}</td>",
            f"<td class='num'>{html.escape(row['quality'])}</td>",
            f"<td class='num'>{html.escape(row['memory'])}</td>",
            f"<td class='num'>{html.escape(row['util'])}</td>",
        ]) + "</tr>")
    return "".join(body)


def bars(rows: list[dict]) -> str:
    selected = [row for row in rows if row["tps"] is not None]
    maximum = max((row["tps"] for row in selected), default=1)
    return "".join(
        f"<div class='barrow'><div class='barlabel'>{html.escape(row['model'])}"
        f"<small>{html.escape(row['profile'])} · {html.escape(row.get('gpu_caption', 'GPU-provenance ontbreekt'))}</small></div>"
        f"<div class='track'><div class='bar' style='width:{100 * row['tps'] / maximum:.1f}%'></div></div>"
        f"<div class='barvalue'>{row['tps']:.2f}</div></div>" for row in selected
    )


def gpu_caption(topology: str | None, visible: str | None = None) -> str:
    """Make physical GPU provenance legible in the compact throughput chart."""
    text = " ".join(part for part in (topology or "", visible or "") if part)
    if "[0, 1, 2, 3]" in text:
        return "GPU 0 A4000 + GPU 1/2 V100 NVLink + GPU 3 RTX 4000 Ada"
    if "[1, 2]" in text or "GPU 1+2" in text:
        return "GPU 1 + 2 · 2× Tesla V100-SXM2-32GB · NVLink"
    if "[0, 3]" in text or "A4000+RTX4000Ada" in text:
        return "GPU 0 RTX A4000 + GPU 3 RTX 4000 Ada · PCIe"
    if "[0]" in text or "GPU 0" in text:
        return "GPU 0 · NVIDIA RTX A4000"
    if "[3]" in text or "GPU 3" in text:
        return "GPU 3 · NVIDIA RTX 4000 Ada"
    if "[2]" in text or "GPU 2" in text:
        return "GPU 2 · Tesla V100-SXM2-32GB"
    if "[1]" in text or "GPU 1" in text:
        return "GPU 1 · Tesla V100-SXM2-32GB"
    return "GPU-provenance ontbreekt"


def throughput_rows(primary_rows: list[dict]) -> list[dict]:
    """One chart of every measured model/configuration, never inferred speeds.

    The wide tables retain their detailed telemetry. This deliberately compact
    view collects their measured decode (or explicitly-labelled aggregate wall)
    rate and repeats the physical GPU contract under every bar.
    """
    rows = list(primary_rows)

    def append(model: str, profile: str, tps: float | None, topology: str | None,
               prompt_tps: float | None = None, engine: str = "llama.cpp") -> None:
        if not isinstance(tps, (int, float)):
            return
        rows.append({
            "model": model, "profile": profile, "tps": tps,
            "prompt_tps": prompt_tps, "engine": engine,
            "gpu_caption": gpu_caption(topology), "scope": "mixed",
        })

    # Complete well-known suite rows include every large model that has already
    # received a throughput probe. Avoid duplicate model/profile entries already
    # shown in the primary local table; distinct suite measurements stay visible.
    primary_models = {row["model"] for row in primary_rows}
    for result in load("well_known_suite_20260917.json").get("results", []):
        name = result.get("model")
        label = WELL_KNOWN_LABELS.get(name, name or "onbekend model")
        if label in primary_models:
            continue
        append(label, "well-known suite", result.get("completion_tokens_per_second"),
               result.get("topology"), result.get("prompt_tokens_per_second"),
               result.get("engine", "llama.cpp"))

    # Same Qwen3.6 weights across all cards/splits: these are genuine additional
    # configurations, not estimates, and therefore belong beside model variants.
    for result in load("hardware_scaling_qwen36_20260921.json").get("results", []):
        append("Qwen3.6-27B", f"hardware · {result.get('profile', '—')}",
               result.get("completion_tokens_per_second"), result.get("topology"),
               result.get("prompt_tokens_per_second"))

    for result in load("hardware_scaling_glm53_20260921.json").get("results", []):
        append("GLM-5.3-Flash 313B-A17B", f"hardware · {result.get('profile', '—')}",
               result.get("completion_tokens_per_second"), result.get("topology"),
               result.get("prompt_tokens_per_second"))

    # This final row is aggregate capacity of two *independent* requests, not a
    # model ensemble score; label it as such to avoid a misleading comparison.
    serving = load("additional_gpu_serving_matrix_20260921.json")
    serial = serving.get("serial_mean", {})
    append("Qwen3.6-27B", "parallel serving · single endpoint", serial.get("a4000_qwen36", {}).get("wall_tokens_per_second"),
           "physical GPU [0]", serial.get("a4000_qwen36", {}).get("prompt_tokens_per_second"), "llama.cpp · wall")
    append("Qwen3.8-27B", "parallel serving · single endpoint", serial.get("ada_qwen38", {}).get("wall_tokens_per_second"),
           "physical GPU [3]", serial.get("ada_qwen38", {}).get("prompt_tokens_per_second"), "llama.cpp · wall")
    append("Qwen3.6 + Qwen3.8", "parallel serving · aggregate 2 requests",
           serving.get("concurrent_aggregate_mean_tokens_per_second"), "physical GPU [0, 3]", None,
           "llama.cpp · aggregate wall")
    return rows


WELL_KNOWN_LABELS = {
    "qwen38-27b-q4": "Qwen3.8-27B", "deepseek-r1-qwen32b-q4": "DeepSeek-R1-Qwen",
    "qwen36-27b-iq3": "Qwen3.6-27B", "qwen35-27b-q4": "Qwen3.5-27B",
    "gemma4-26b-a4b-q4": "Gemma4-26B-A4B", "devstral-small2-24b-q4": "Devstral Small 2",
    "qwen25-72b-q4": "Qwen2.5-72B", "llama31-70b-instruct-q4": "Llama-3.1-70B",
    "glm45-air-106b-iq3": "GLM-4.5-Air-106B-A12B", "gpt-oss-120b-q4": "gpt-oss-120B",
    "kat-coder-v2.5-dev": "KAT-Coder-V2.5-Dev", "granite-4.2-30b": "Granite-4.2-30B",
    "nemotron35-lightning-30b-a3b": "Nemotron-3.5-Lightning-30B-A3B",
    "muse-glimmer-30b": "Muse-Glimmer-30B",
    "qwen36-35b-a3b-nvfp4": "Qwen3.6-35B-A3B (1Cat-vLLM NVFP4)",
    "qwen38-1cat-vllm-target": "Qwen3.8-27B (1Cat-vLLM NVFP4)",
    "mixture-of-models": "Mixture-of-models",
    "mixture-of-models-4-quality": "Mixture-of-models 4 (quality)",
    "mixture-of-models-4-gemma-routed": "Mixture-of-models 4 (Gemma route)",
    "mixture-of-models-4-gemma-direct": "Mixture-of-models 4 (Gemma direct)",
    "mixture-ultimate-6-explicit": "Mixture ultieme 6 (nemotron+deepseek+glm+kat+qwen35+qwen38-iq2s)",
    "gpt-oss-20b-q4": "gpt-oss-20B",
    "mistral-small4-119b-q4-adaa4000": "Mistral Small 4 (Ada+A4000)",
    "deepseek-v4-flash-reap150b-q2k-adaa4000": "DeepSeek-V4-Flash-REAP-150B (Ada+A4000)",
    "deepseek-v4-flash-0731-iq3xxs": "DeepSeek-V4-Flash-0731 UD-IQ3_XXS",
    "llama3-70b-instruct-q4": "Llama-3-70B-Instruct Q4_K_M",
    "qwen35-122b-a10b-iq3s": "Qwen3.5-122B-A10B IQ3_S",
    "command-r-plus-104b-0824-iq3m": "Command R+ 104B 08-2024 IQ3_M",
    "mixtral-8x22b-instruct-q3ks": "Mixtral 8x22B Instruct Q3_K_S",
    "wizardlm2-8x22b-iq3s": "WizardLM-2 8x22B IQ3_S",
    "glm53-flash-aj-iq2xxs": "GLM-5.3-Flash 313B-A17B AJ-IQ2_XXS",
    "glm53-reap50-iq3m": "GLM-5.3-Flash REAP50 IQ3_M",
    "glm53-reap50-q3km": "GLM-5.3-Flash REAP50 Q3_K_M",
    "glm53-reap50-iq4xs": "GLM-5.3-Flash REAP50 IQ4_XS",
    "glm53-reap50-q4km": "GLM-5.3-Flash REAP50 Q4_K_M",
    "glm53-reap50-iq3m-v100": "GLM-5.3-Flash REAP50 IQ3_M · 2×V100",
    "glm53-reap50-iq3m-allfour": "GLM-5.3-Flash REAP50 IQ3_M · 4 GPU's",
    "glm53-reap50-q3km-v100": "GLM-5.3-Flash REAP50 Q3_K_M · 2×V100",
    "glm53-reap50-q3km-allfour": "GLM-5.3-Flash REAP50 Q3_K_M · 4 GPU's",
    "glm53-reap50-iq4xs-v100": "GLM-5.3-Flash REAP50 IQ4_XS · 2×V100",
    "glm53-reap50-iq4xs-allfour": "GLM-5.3-Flash REAP50 IQ4_XS · 4 GPU's",
    "glm53-reap50-q4km-v100": "GLM-5.3-Flash REAP50 Q4_K_M · 2×V100",
    "glm53-reap50-q4km-allfour": "GLM-5.3-Flash REAP50 Q4_K_M · 4 GPU's",
    "qwen38-flash-next-merlin-w4a16-v100": "Qwen3.8 Flash-Next Merlin W4A16 · 2×V100",
    "qwen38-flash-next-merlin-w4a16-allfour": "Qwen3.8 Flash-Next Merlin W4A16 · 4 GPU's",
    "qwen38-flash-next-awq-w4a16-v100": "Qwen3.8 Flash-Next AWQ W4A16 · 2×V100",
    "qwen38-flash-next-awq-w4a16-allfour": "Qwen3.8 Flash-Next AWQ W4A16 · 4 GPU's",
    "qwen38-flash-next-ap-iq4xs-v100": "Qwen3.8 Flash-Next AP-IQ4_XS · 2×V100",
    "qwen38-flash-next-ap-iq4xs-allfour": "Qwen3.8 Flash-Next AP-IQ4_XS · 4 GPU's",
    "qwen38-flash-next-ap-q4km-v100": "Qwen3.8 Flash-Next AP-Q4_K_M · 2×V100",
    "qwen38-flash-next-ap-q4km-allfour": "Qwen3.8 Flash-Next AP-Q4_K_M · 4 GPU's",
    "qwen38-flash-next-ap-iq2s-v100": "Qwen3.8 Flash-Next AP-IQ2_S · 2×V100",
    "qwen38-flash-next-ap-iq2s-allfour": "Qwen3.8 Flash-Next AP-IQ2_S · 4 GPU's",
}
WELL_KNOWN_ORDER = list(WELL_KNOWN_LABELS)
ACCESS_PROFILES = (
    ("sandbox", "Sandbox · geen internet of schijftools"),
    ("disk", "Disk · gecontroleerde read-only werkmap, geen internet"),
    ("internet_disk", "Internet + disk · gelogde webtool en gecontroleerde werkmap"),
)


def sortable_header(label: str, explanation: str | None = None,
                    direction: str | None = None) -> str:
    """Keep navigation and sorting as separate, keyboard-accessible controls."""
    escaped = html.escape(label)
    heading = (f'<a class="benchmark-link" href="#{html.escape(explanation, quote=True)}" '
               f'title="Ga naar uitleg van {escaped}">{escaped}</a>') \
        if explanation else f"<span>{escaped}</span>"
    sort_state = f' data-sort-dir="{direction}"' if direction else ""
    return (f'<th><span class="th-content">{heading}'
            f'<button class="sort-control" type="button"{sort_state} '
            f'aria-label="Sorteer op {escaped}" title="Sorteer op {escaped}"></button>'
            f'</span></th>')


def well_known_rows() -> list[dict]:
    results = {r.get("model"): r for r in load("well_known_suite_20260917.json").get("results", [])}
    throughput = {
        r.get("model"): r.get("completion_tokens_per_second")
        for r in load("local_gguf_8bench_dual_v100_20260917.json").get("results", [])
        if not r.get("error")
    }
    rows = []
    dynamic_mixtures = sorted(
        name for name in results
        if isinstance(name, str) and name.startswith("mixture-optimized-")
    )
    for name in WELL_KNOWN_ORDER + dynamic_mixtures:
        r = results.get(name, {})
        gsm8k = rm.gsm8k_score(r) if r else None
        bbh = (r.get("bbh") or {}).get("mean_accuracy")
        mmlu = (r.get("mmlu_sample") or {}).get("mean_accuracy")
        truthfulqa = r.get("truthfulqa_gen", {}).get("rougeL_acc,none")
        humaneval = (r.get("humaneval") or {}).get("pass_at_1")
        composite_parts = [v for v in (gsm8k, humaneval, mmlu, bbh) if isinstance(v, (int, float))]
        # Same unweighted-mean-of-4 formula as rank_models.py's composite; only
        # populated once all 4 accuracy metrics exist so it's directly comparable.
        composite = sum(composite_parts) / len(composite_parts) if len(composite_parts) == 4 else None
        current_protocol = r.get("eval_protocol") == wks.EVAL_PROTOCOL
        # 2026-09-19: status used to require truthfulqa too (all 5 metrics), but
        # truthfulqa was never part of the composite (see rank_models.py) and
        # mixture-of-models deliberately omits it (free-text task, no discrete
        # target -- see mixture_of_models.py). That made a row with a complete,
        # correct composite show as "onvolledig" (incomplete), which is wrong --
        # completeness is judged on the composite's 4 metrics only.
        is_complete_current = current_protocol and composite is not None
        error = str(r.get("error", ""))
        if (name.startswith("qwen38-flash-next-merlin-w4a16-") and name.endswith("-v100")) or (
                "Min capability: 75" in error and "Current capability: 70" in error):
            status = "geteste backend onverenigbaar · SM75 vereist; 1Cat-herbeoordeling volgt"
        elif (name.startswith("qwen38-flash-next-merlin-w4a16-") and name.endswith("-allfour")) or (
                "OutOfMemoryError" in error or "CUDA out of memory" in error):
            status = "onverenigbaar · onvoldoende VRAM voor dit profiel"
        elif name.startswith("qwen38-flash-next-ap-") and r.get("error"):
            status = "mislukt · GGUF diagnose/herpoging gepland"
        elif r.get("error"):
            status = "mislukt · diagnose/herpoging gepland"
        elif is_complete_current:
            status = f"compleet · {r.get('engine', 'llama.cpp-gguf')}"
        elif current_protocol:
            status = "huidig protocol · onvolledig"
        elif r:
            status = "oud protocol · hermeting loopt"
        elif name.startswith(("mixture-of-models", "mixture-optimized-")):
            status = "na individuele modellen"
        elif name in {"glm45-air-106b-iq3", "gpt-oss-120b-q4"}:
            status = "finale · download/test gepland"
        else:
            status = "in benchmarkwachtrij"
        label = WELL_KNOWN_LABELS.get(name, name.replace("mixture-optimized-", "Optimized mixture "))
        if name.startswith(("mixture-of-models", "mixture-optimized-")) and r.get("ensemble_members"):
            weights = r.get("ensemble_weights")
            if weights:
                members = ", ".join(
                    f"{WELL_KNOWN_LABELS.get(m, m)} ({w:.0%})"
                    for m, w in zip(r["ensemble_members"], weights)
                )
            else:
                members = ", ".join(WELL_KNOWN_LABELS.get(m, m) for m in r["ensemble_members"])
            method = r.get("ensemble_method", "unknown")
            method_label = {
                "majority_vote": "meerderheidsstemming", "mean_logits": "gemiddelde logits",
                "weighted_logits": "gewogen logits", "confidence_weighted": "confidence-gewogen",
                "benchmark_best": "beste-op-benchmark", "router": "router", "stacked": "stacked",
                "task_router": "taakrouter",
            }.get(method, method)
            label = f"Mixture-of-models — {method_label} ({members})"
        rows.append({
            "model": label,
            "gsm8k": gsm8k, "bbh": bbh, "mmlu": mmlu, "truthfulqa": truthfulqa,
            "humaneval": humaneval, "composite": composite,
            "tps": r.get("completion_tokens_per_second", throughput.get(name)), "status": status,
            "hardware": r.get("topology") or rm.hardware_label(r) if r else "—",
            # Old results predate access provenance and were raw chat-server
            # measurements, hence they are sandbox-only by construction.
            "access_profile": r.get("access_profile", "sandbox"),
        })
    # Ranked rows (composite available) sort highest-first; rows still pending
    # a full suite keep their WELL_KNOWN_ORDER position at the bottom instead
    # of jumping around as partial data arrives.
    ranked = sorted((r for r in rows if r["composite"] is not None), key=lambda r: r["composite"], reverse=True)
    pending = [r for r in rows if r["composite"] is None]
    return ranked + pending


def pct_cell(value: float | None) -> str:
    if value is None:
        return "<td class='num'>—</td>"
    cls = "score-hi" if value >= 0.5 else ("score-mid" if value >= 0.25 else "score-lo")
    return f"<td class='num {cls}'>{round(100 * value)}%</td>"


def well_known_section(rows: list[dict]) -> str:
    if not rows:
        return ""
    def table(profile_rows: list[dict]) -> str:
        if not profile_rows:
            return "<div class='callout'>Nog geen meting onder dit toegangsprofiel.</div>"
        body = "".join(
        "<tr>" + f"<td><strong>{html.escape(r['model'])}</strong></td>"
        + score_cell(r["composite"])
        + pct_cell(r["gsm8k"]) + pct_cell(r["bbh"]) + pct_cell(r["mmlu"])
        + pct_cell(r["truthfulqa"]) + pct_cell(r["humaneval"])
        + f"<td class='num' data-sort='{r['tps'] if r['tps'] is not None else ''}'>{fmt(r['tps'])}</td>"
        + f"<td>{html.escape(r['hardware'])}</td>"
        + f"<td>{html.escape(r['status'])}</td>" + "</tr>"
            for r in profile_rows
        )
        return f"""<div class="tablewrap"><table class="sortable"><thead><tr>
{sortable_header("Model")}{sortable_header("Composite", "benchmark-composite", "desc")}
{sortable_header("GSM8K", "benchmark-gsm8k")}{sortable_header("BBH", "benchmark-bbh")}
{sortable_header("MMLU", "benchmark-mmlu")}{sortable_header("TruthfulQA", "benchmark-truthfulqa")}
{sortable_header("HumanEval pass@1", "benchmark-humaneval")}{sortable_header("t/s", "benchmark-throughput")}
{sortable_header("Hardware")}{sortable_header("Status")}</tr></thead><tbody>{body}</tbody></table></div>"""
    profile_tables = "".join(f"<h3>{title}</h3>{table([r for r in rows if r['access_profile'] == key])}"
                             for key, title in ACCESS_PROFILES)
    return f"""
<h2>Bekende benchmarks (lm-eval-harness + eigen HumanEval)</h2>
<p>GSM8K (50 samples, flexible-extract), BBH (6 representatieve subtaken x 8 samples,
CoT few-shot), MMLU-steekproef (8 vakken x 20 samples, generatief met eigen
letter-extractie i.p.v. lm-eval's te strikte exact-match -- zie legend), TruthfulQA-gen
(30 samples, ROUGE-L-acc), HumanEval (40 van 164 problemen, echte pass@1 via
uitvoering). Geen LLM-jury, alles objectief/uitvoerbaar gescoord. Composite = ongewogen
gemiddelde van GSM8K/HumanEval/MMLU/BBH (zelfde formule als <code>rank_models.py</code>),
alleen getoond zodra alle vier compleet zijn -- standaard op composite gesorteerd,
klik een kolomkop om te hersorteren. Niet elk model draait op dezelfde hardware: naast
het 2×V100-paar staan ook A4000- en A4000+Ada-resultaten in deze tabel. De Hardware-kolom
is daarom onderdeel van ieder resultaat; vergelijk t/s alleen bij een gelijk profiel. Elke tabel hieronder heeft een
andere tooltoegang; scores worden nooit tussen de profielen gekopieerd.</p>
{profile_tables}
<div class="callout"><strong>Live voortgang.</strong> Een rij wordt pas na de volledige suite
atomair vervangen. “Oud protocol” is uitsluitend historische context en telt niet mee voor de
eindrangschikking. Protocol v2 gebruikt de model-chattemplate, veilige MMLU-stopcondities en
gelogde samples. Lege scorecellen betekenen “nog niet voltooid”, niet 0%.</div>
<div class="legend"><span>MMLU-fix (los van de stop-bug hierboven): lm-eval's
<code>mmlu_generative</code> exact-match keurde "B. 4" af als fout antwoord op target "B"
(het model had wél gelijk) -- eigen scorer pakt nu de eerste vrijstaande A/B/C/D-letter uit
de ruwe respons.</span><span>BBH-fix: lm-eval liet model-eindtokens/punctuatie in het
gefilterde antwoord staan (<code>False.&lt;|im_end|&gt;</code> versus target <code>False</code>).
<code>normalized_logged_answer_v1</code> vergelijkt de eerste getypeerde keuze uit de bewaarde
samplelogs; alle beschikbare historische rijen zijn daarmee offline herwaardeerd.</span>
<span>t/s bij mixture-rijen (Mixture-of-models — ...): geen gemeten mixture-doorvoer -- leden
worden sequentieel bevraagd, nooit gelijktijdig bediend. De getoonde waarde is de t/s van het
traagste lid (bottleneck), oftewel het plafond als elk lid parallel op aparte hardware zou
draaien. Praktische latency op gedeelde hardware, met laden/wisselen tussen leden, ligt
hoger.</span></div>
<section id="benchmark-uitleg" class="benchmark-explanations" aria-labelledby="benchmark-uitleg-title">
<h3 id="benchmark-uitleg-title">Uitleg van de benchmarks</h3>
<div class="explanation-grid">
<article id="benchmark-composite"><h4>Composite</h4><p>Ongewogen gemiddelde van GSM8K,
BBH, MMLU en HumanEval. TruthfulQA telt niet mee omdat de generatie-/ROUGE-maat een
andere semantiek heeft.</p><a href="#benchmark-uitleg-title">Terug naar boven</a></article>
<article id="benchmark-gsm8k"><h4>GSM8K</h4><p>50 basisschool-wiskundeproblemen;
exactheid van het uiteindelijke numerieke antwoord met flexibele extractie.</p><a href="#benchmark-uitleg-title">Terug naar boven</a></article>
<article id="benchmark-bbh"><h4>BBH</h4><p>48 voorbeelden uit zes BIG-Bench Hard-taken.
De opgeslagen eindantwoorden worden genormaliseerd vóór exact-match scoring.</p><a href="#benchmark-uitleg-title">Terug naar boven</a></article>
<article id="benchmark-mmlu"><h4>MMLU</h4><p>160 vragen uit acht vakgebieden. De scorer
extraheert de eerste zelfstandige antwoordletter A–D uit de modelrespons.</p><a href="#benchmark-uitleg-title">Terug naar boven</a></article>
<article id="benchmark-truthfulqa"><h4>TruthfulQA</h4><p>30 vrije-tekstvragen, weergegeven
als ROUGE-L-accuracy. Deze score is diagnostisch en zit niet in Composite.</p><a href="#benchmark-uitleg-title">Terug naar boven</a></article>
<article id="benchmark-humaneval"><h4>HumanEval pass@1</h4><p>40 programmeertaken;
de eerste gegenereerde oplossing wordt werkelijk uitgevoerd tegen de tests.</p><a href="#benchmark-uitleg-title">Terug naar boven</a></article>
<article id="benchmark-throughput"><h4>t/s</h4><p>Gemeten decode-throughput in tokens per
seconde. Vergelijk dit alleen wanneer hardware, runtime en profiel gelijk zijn.</p><a href="#benchmark-uitleg-title">Terug naar boven</a></article>
</div></section>
"""


SPECIALIST_LABELS = {
    "chemistry": "Chemistry · eigen holdout",
    "physics": "Physics · eigen holdout",
    "vision": "Vision · object/spatial",
    "video": "Video · MP4 artifact",
    "iq": "IQ · rule reasoning",
    "eq": "EQ · social reasoning",
    "fq": "FQ · robot simulatie",
    "qq": "QQ · quantum",
}


def specialist_section() -> str:
    """One row per known model; pre-protocol cells deliberately remain blank."""
    records = {row.get("model"): row for row in load("specialist_suite_20260922.json").get("results", [])}
    names = list(WELL_KNOWN_ORDER) + [name for name in records if name not in WELL_KNOWN_ORDER]
    headers = "".join(sortable_header(label, f"specialist-{key}") for key, label in SPECIALIST_LABELS.items())
    def table(profile: str) -> str:
        body = ""
        for name in names:
            record = records.get(name, {})
            if record and record.get("access_profile", "sandbox") != profile:
                continue
            if not record and profile != "sandbox":
                continue
            results = record.get("results", {})
            cells, states = [], []
            for key in SPECIALIST_LABELS:
                result = results.get(key, {})
                cells.append(pct_cell(result.get("accuracy") if result.get("status") == "complete" else None))
                if result:
                    states.append(f"{key}: {result.get('status', 'unknown')}")
            body += ("<tr><td><strong>" + html.escape(WELL_KNOWN_LABELS.get(name, name)) + "</strong></td>" +
                     "".join(cells) + "<td>" + html.escape("; ".join(states) if states else "niet uitgevoerd (historisch)") + "</td></tr>")
        if not body:
            return "<div class='callout'>Nog geen specialistische meting onder dit toegangsprofiel.</div>"
        return f"<div class=\"tablewrap\"><table class=\"sortable\"><thead><tr>{sortable_header('Model')}{headers}{sortable_header('Status')}</tr></thead><tbody>{body}</tbody></table></div>"
    profile_tables = "".join(f"<h3>{title}</h3>{table(key)}" for key, title in ACCESS_PROFILES)
    return f"""
<h2>Eigen specialistische suite</h2>
<p>Optionele expert-suite. Bestaande modelrijen blijven bewust leeg: een em-dash betekent <em>niet gedraaid</em>, nooit 0%. Elke specialist heeft een eigen lokale, bewerkbare CSV waarvan antwoord- en rubricvelden nooit naar het model gaan. Vision bevat synthetische object- en ruimtelijke assets; Video meet een lokaal gerenderd MP4-artifact; FQ valideert actuatorcommando’s in een deterministische differential-drive-simulator. Vision en Video hebben elk een harde grens van vijf minuten. IQ/EQ zijn taaklabels voor abstract respectievelijk sociaal-emotioneel redeneren, geen klinische persoonsmetingen. Geen score telt mee in de algemene Composite.</p>
{profile_tables}
<section id="specialist-uitleg" class="benchmark-explanations"><h3>Specialistische meetdefinities</h3><div class="explanation-grid">
<article id="specialist-chemistry"><h4>Chemistry · eigen holdout</h4><p>Lokale CSV met eigen vragen en antwoordletters; de GPQA-publicatie (2023-11-20) is alleen de moeilijkheidsreferentie.</p></article>
<article id="specialist-physics"><h4>Physics · eigen holdout</h4><p>Afzonderlijke, wijzigbare physics-vragen. Roteer de CSV-versie vóór een nieuwe vergelijkingsreeks.</p></article>
<article id="specialist-vision"><h4>Vision · MMMU-Pro</h4><p>16 vision-only expertvragen met beelden, diagrammen en tabellen; publicatie 2024-09-04. Text-only endpoints worden als unsupported gelogd.</p></article>
<article id="specialist-video"><h4>Video · MP4 artifact</h4><p>Drie FFmpeg-filtergraphs; render, duur en bestandsgrootte worden objectief gevalideerd. Maximaal vijf minuten, geen LLM-jury.</p></article>
<article id="specialist-iq"><h4>IQ · rule reasoning</h4><p>Eigen abstracte regels, recurrences en formele deductie. Dit is geen psychometrische IQ-test.</p></article>
<article id="specialist-eq"><h4>EQ · social reasoning</h4><p>Kalibratie, empathische triage en conflicthantering met expliciete rubrics; geen klinische EQ-diagnose.</p></article>
<article id="specialist-fq"><h4>FQ · robot simulatie</h4><p>Het model produceert wielactuator-JSON; de scorer berekent positie en heading met differential-drive-kinematica.</p></article>
<article id="specialist-qq"><h4>QQ · quantum</h4><p>Quantumtoestanden, meting, communicatie, metrologie en quantumchemie/VQE.</p></article>
</div></section>
"""


def contamination_section() -> str:
    """Separate mitigation/audit table; never infer training exposure from a score alone."""
    standard = {r.get("model"): r for r in load("well_known_suite_20260917.json").get("results", [])}
    specialist = {r.get("model"): r for r in load("specialist_suite_20260922.json").get("results", [])}
    body = ""
    for name in WELL_KNOWN_ORDER:
        base = standard.get(name, {})
        parts = [rm.gsm8k_score(base), (base.get("bbh") or {}).get("mean_accuracy"),
                 (base.get("mmlu_sample") or {}).get("mean_accuracy"), (base.get("humaneval") or {}).get("pass_at_1")]
        static_score = sum(v for v in parts if isinstance(v, (int, float))) / 4 if all(isinstance(v, (int, float)) for v in parts) else None
        custom = (specialist.get(name, {}).get("results") or {})
        private_parts = [item.get("accuracy") for key, item in custom.items()
                         if key != "video" and item.get("status") == "complete"]
        private_score = sum(private_parts) / len(private_parts) if private_parts else None
        gap = static_score - private_score if static_score is not None and private_score is not None else None
        release = base.get("model_release_date") or "nog vast te leggen"
        release_source = base.get("model_release_source")
        release_html = html.escape(release) + (" <small>bron opgeslagen</small>" if release_source else "")
        status = "vergelijkbaar" if gap is not None else "wacht op private/live score + releasebron"
        body += ("<tr><td><strong>" + html.escape(WELL_KNOWN_LABELS.get(name, name)) + "</strong></td>" +
                 f"<td>{release_html}</td>{pct_cell(static_score)}{pct_cell(private_score)}" +
                 f"<td class='num'>{'—' if gap is None else f'{gap:+.1%}'}</td><td>{status}</td></tr>")
    return f"""
<h2>Data-contaminatie: mitigaties en audit</h2>
<p>Publieke benchmarks kunnen in pretraining, post-training of benchmark-optimalisatie terechtkomen. Een hoge publieke score is daarom geen zelfstandig bewijs van algemene vaardigheid. De audit vergelijkt dezelfde modelconfiguratie pas nadat zowel standaard- als private/live-resultaat bestaan; de gap is een <em>onderzoekssignaal</em>, geen bewijs van memorisatie of fraude.</p>
<div class="tablewrap"><table class="sortable"><thead><tr>{sortable_header("Model")}{sortable_header("Modelpublicatie")}{sortable_header("Standaard composite", "contamination-static")}{sortable_header("Private/live score", "contamination-private")}{sortable_header("Verschil")}{sortable_header("Auditstatus")}</tr></thead><tbody>{body}</tbody></table></div>
<div class="tablewrap"><table><thead><tr><th>Maatregel</th><th>Waarom</th><th>Bron / publicatiedatum</th><th>Lokale uitvoering</th></tr></thead><tbody>
<tr><td>Private held-out set</td><td>Modelbouwers zien vragen/antwoorden niet vóór de eindmeting.</td><td>ARC-AGI-2 private eval · 2025</td><td>Private CSV-packs met SHA-256; publiceer een nieuwe pack niet vóór de run.</td></tr>
<tr><td>Dynamische benchmark</td><td>Vragen ontstaan na de bekende modelcutoff.</td><td>LiveBench · 2024-06-27</td><td>Optionele live-lane; score, bron- en vraagdatum opslaan.</td></tr>
<tr><td>Recente code-opgaven</td><td>Publicatiedatum van de opgave kan tegen modelrelease worden afgezet.</td><td>LiveCodeBench · 2024-03-12</td><td>Alleen opgaven ná release/cutoff; apart rapporteren.</td></tr>
<tr><td>Canary-string probe</td><td>Onwaarschijnlijke string kan trainingsblootstelling detecteren.</td><td>BIG-bench GUID canary · 2022</td><td>Alleen logprob-geschikte engines; vergelijk met willekeurige GUID-controls.</td></tr>
<tr><td>Modelprovenance</td><td>Release- en trainingscutoff maken tijdsvergelijking controleerbaar.</td><td>Modelcard/release note</td><td>Elke nieuwe run bewaart datum plus primaire bron-URL.</td></tr>
</tbody></table></div>
<section id="contamination-uitleg" class="benchmark-explanations"><h3>Interpretatie</h3><div class="explanation-grid">
<article id="contamination-static"><h4>Standaard composite</h4><p>De bestaande publieke GSM8K/BBH/MMLU/HumanEval-composite.</p></article>
<article id="contamination-private"><h4>Private/live score</h4><p>Gemiddelde van vergelijkbare, voltooide private of tijdgebonden lanes; video blijft apart wegens andere metriek.</p></article>
</div></section>
"""


def matrix_gate_section() -> str:
    gate = bpg.matrix_gate()
    state = "vrijgegeven" if gate["matrix_enabled"] else "geblokkeerd"
    return f"""
<h2>Toegangsprofiel-matrix: fasepoort</h2>
<div class="callout"><strong>{state}.</strong> {html.escape(gate['reason'])}<br>
Huidige fase: <code>{html.escape(gate['phase'])}</code>. De drie profielen per standaard- en specialistische suite starten pas na een complete GLM-5.3 sandboxrun met huidige protocol, core-scores en t/s.</div>
"""


def candidate_research_section() -> str:
    candidates = load("dual_v100_candidate_research_20260918.json").get("candidates", [])
    if not candidates:
        return ""
    body = "".join(
        "<tr>" + f"<td><strong>{html.escape(c.get('model', '—'))}</strong></td>"
        + f"<td>{html.escape(c.get('local_quant', '—'))}</td>"
        + f"<td>{html.escape(c.get('published_at', '—'))}</td>"
        + f"<td>{html.escape(c.get('status', '—'))}</td>"
        + f"<td style='white-space:normal;min-width:360px'>{html.escape(c.get('resolution', c.get('reason', '—')))}</td></tr>"
        for c in candidates
    )
    return f"""
<h2>Onderzochte grote modellen</h2>
<p>Volledige kandidatenlijst, inclusief modellen die na geheugen-, reproduceerbaarheids- en
kwaliteitsonderzoek niet in de finale kwamen. Publicatiedatum en bron staan in het bron-JSON.
Vrije tekst in deze historische onderzoeksnotities kan de oorspronkelijke lm-eval-BBH-score
noemen; de gesorteerde benchmarktabel hierboven bevat de latere, geldige
<code>normalized_logged_answer_v1</code>-herwaardering en is leidend.</p>
<div class="tablewrap"><table><thead><tr><th>Model</th><th>Formaat / quant</th>
<th>Publicatie</th><th>Status</th><th>Beslissing</th></tr></thead><tbody>{body}</tbody></table></div>
"""


def large_model_provenance_section() -> str:
    results = [row for row in load("well_known_suite_20260917.json").get("results", [])
               if row.get("source_repo") and row.get("source_revision")]
    if not results:
        return ""
    state = load("remaining_large_model_cascade_20260921.json")
    pruned = {event.get("model") for event in state.get("events", [])
              if event.get("event") == "pruned"}
    winner = state.get("winner") or state.get("current_winner")
    rows = []
    for row in results:
        name = row.get("model", "—")
        source = row.get("model_source") or f"https://huggingface.co/{row['source_repo']}"
        if name == winner:
            status = "winnaar · weights behouden"
        elif name in pruned:
            status = "volledig getest · veilig verwijderd"
        else:
            status = "volledig getest · cascade actief"
        rows.append("<tr>" + "".join([
            f"<td><strong>{html.escape(WELL_KNOWN_LABELS.get(name, name))}</strong></td>",
            f"<td><a href='{html.escape(source, quote=True)}'>{html.escape(row['source_repo'])}</a></td>",
            f"<td>{html.escape(str(row.get('source_published_at', '—')))}</td>",
            f"<td><code>{html.escape(str(row.get('source_revision', '—'))[:12])}</code></td>",
            f"<td>{html.escape(str(row.get('quantization', '—')))}</td>",
            f"<td class='num'>{fmt((row.get('weight_bytes') or 0) / 1_000_000_000)}</td>",
            f"<td>{html.escape(status)}</td>",
        ]) + "</tr>")
    return f"""
<h2>Bronnen en bewaarcascade grote modellen</h2>
<p>Iedere download is aan een repository-revisie vastgezet. Publicatiedatum en bron blijven
in het resultaat staan nadat een gevalideerde verliezer is verwijderd; de opgeslagen
reinstall-opdracht in het cascade-JSON is niet afhankelijk van een rate-limited inference-API.</p>
<div class="tablewrap"><table><thead><tr><th>Model</th><th>Bron</th><th>Publicatie</th>
<th>Revisie</th><th>Quant</th><th>GB weights</th><th>Status</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
"""


def qwen38_flash_next_vllm_section() -> str:
    """Separate measured GGUF results from reproducible 1Cat research targets."""
    return """
<h2>Qwen3.8-Flash-Next · volgende runtimeproeven</h2>
<p>De AP-GGUF-metingen in de hoofdtabel zijn lokale llama.cpp-resultaten. De hieronder
genoemde 1Cat-routes zijn afzonderlijke experimenten en krijgen pas een score na een
lokale run met vastgelegde checkpoint-, runtime- en hardware-revisie.</p>
<div class="tablewrap"><table class="sortable"><thead><tr><th>Profiel</th><th>Fysieke GPU’s</th>
<th>vLLM-configuratie</th><th>t/s</th><th>Benchmarks</th><th>Status</th></tr></thead><tbody>
<tr><td><strong>V100-only</strong></td><td>GPU 1 + 2 · 2× Tesla V100-SXM2-32GB · NVLink</td>
<td><code>CUDA_VISIBLE_DEVICES=1,2</code> · TP=2</td><td class="num">—</td><td>smoke · kwaliteit · pure decode</td>
<td>1Cat-vLLM 1.5.0 baseline eerst; alleen publiceren indien SM70-preflight slaagt</td></tr>
<tr><td><strong>Heterogeen onderzoek</strong></td><td>GPU 0 A4000 + GPU 1/2 V100 NVLink + GPU 3 RTX 4000 Ada</td>
<td>alleen een backend die ongelijke compute capabilities expliciet ondersteunt</td><td class="num">—</td>
<td>identieke suite; afzonderlijk gerapporteerd</td><td>geen TP=4-claim zolang de mixed-GPU preflight niet slaagt</td></tr>
</tbody></table></div>
<div class="callout"><strong>Bewijsregel.</strong> De upstream 1Cat-snelheden zijn gemeten onder
andere topologieën en workloads. Ze zijn onderzoeksdoelen, geen lokale resultaten. MTP krijgt
altijd een eigen rij en vervangt de normale decode-score nooit. Zie
<a href="../docs/MODEL_TEST_ROADMAP.md">de actuele model-roadmap</a>.</div>
"""


def hardware_scaling_section() -> str:
    """Render an apples-to-apples Qwen3.6 hardware matrix when it exists."""
    results = [row for row in load("hardware_scaling_qwen36_20260921.json").get("results", [])
               if not row.get("error")]
    if not results:
        return ""
    labels = {
        "single-a4000": "RTX A4000 15GB",
        "single-ada": "RTX 4000 Ada 20GB",
        "dual-a4000ada": "A4000 + RTX 4000 Ada (PCIe layer split)",
        "single-v100": "1× V100-SXM2 32GB",
        "dual-layer": "2× V100-SXM2 (NVLink layer split)",
        "dual-tensor": "2× V100-SXM2 (NVLink tensor split)",
    }
    baseline = next((row.get("completion_tokens_per_second") for row in results
                     if row.get("profile") == "single-a4000"), None)
    body = []
    for row in results:
        profile = row.get("profile", "—")
        tps = row.get("completion_tokens_per_second")
        gain = 100 * (tps / baseline - 1) if baseline and tps is not None else None
        telemetry = row.get("gpu_decode_telemetry") or {}
        memory = " / ".join(
            fmt(device.get("max_memory_mib") / 1024)
            for device in telemetry.values() if device.get("max_memory_mib") is not None
        )
        utilization = " / ".join(
            fmt(device.get("mean_util_pct"), 1) + "%"
            for device in telemetry.values() if device.get("mean_util_pct") is not None
        )
        body.append("<tr>" + "".join([
            f"<td><strong>{html.escape(labels.get(profile, profile))}</strong></td>",
            f"<td class='num hot'>{fmt(tps)}</td>",
            f"<td class='num'>{fmt(row.get('prompt_tokens_per_second'))}</td>",
            f"<td class='num'>{fmt(gain, 1) + '%' if gain is not None else '—'}</td>",
            (f"<td class='num'>{fmt(row.get('benchmark_mean_score') * 100, 1)}%</td>"
             if row.get("benchmark_mean_score") is not None else "<td class='num'>—</td>"),
            f"<td class='num'>{html.escape(memory)}</td>",
            f"<td class='num'>{html.escape(utilization)}</td>",
        ]) + "</tr>")
    return f"""
<h2>Zelfde model, andere hardware</h2>
<p>Qwen3.6-27B IQ3_XXS draait hier met exact dezelfde weights, prompt, context en
256-token decode op iedere configuratie. Hierdoor meet deze tabel het hardware-/split-effect;
anders dan in de modelranglijst verandert de modelkwaliteit niet.</p>
<div class="tablewrap"><table class="sortable"><thead><tr><th>Configuratie</th>
<th data-sort-dir="desc">Decode t/s</th><th>Prompt t/s</th><th>vs. A4000</th>
<th>8-task score</th><th>VRAM GiB</th><th>GPU util.</th></tr></thead>
<tbody>{''.join(body)}</tbody></table></div>
"""


def glm53_hardware_section() -> str:
    """Render the large GLM-5.3-Flash CPU/GPU-offload matrix as it progresses."""
    data = load("hardware_scaling_glm53_20260921.json")
    results = data.get("results", [])
    if not results:
        return ""
    labels = {
        "single-a4000": "RTX A4000 15GB",
        "single-ada": "RTX 4000 Ada 20GB",
        "dual-a4000ada": "A4000 + RTX 4000 Ada (PCIe)",
        "single-v100": "V100 #1 32GB (zelfstandig)",
        "single-v100-2": "V100 #2 32GB (zelfstandig)",
        "dual-layer": "2× V100-SXM2 (NVLink)",
        "all-four-layer": "Alle 4 GPU’s (99GB VRAM)",
    }
    good = [row for row in results if not row.get("error")]
    baseline = next((row.get("completion_tokens_per_second") for row in good
                     if row.get("profile") == "single-v100"), None)
    body = []
    for row in results:
        profile = row.get("profile", "—")
        if row.get("error"):
            body.append("<tr>" + "".join([
                f"<td><strong>{html.escape(labels.get(profile, profile))}</strong></td>",
                "<td class='num'>—</td><td class='num'>—</td><td class='num'>—</td>",
                "<td class='num'>—</td><td class='num'>—</td>",
                f"<td>{html.escape(row['error'])}</td>",
            ]) + "</tr>")
            continue
        tps = row.get("completion_tokens_per_second")
        versus = tps / baseline if baseline and tps is not None else None
        telemetry = row.get("gpu_decode_telemetry") or {}
        memory = " / ".join(
            fmt(device.get("max_memory_mib") / 1024)
            for device in telemetry.values() if device.get("max_memory_mib") is not None
        )
        utilization = " / ".join(
            fmt(device.get("mean_util_pct"), 1) + "%"
            for device in telemetry.values() if device.get("mean_util_pct") is not None
        )
        body.append("<tr>" + "".join([
            f"<td><strong>{html.escape(labels.get(profile, profile))}</strong></td>",
            f"<td class='num hot'>{fmt(tps)}</td>",
            f"<td class='num'>{fmt(row.get('prompt_tokens_per_second'))}</td>",
            f"<td class='num'>{fmt(versus, 2) + '×' if versus is not None else '—'}</td>",
            (f"<td class='num'>{fmt(row.get('benchmark_mean_score') * 100, 1)}%</td>"
             if row.get("benchmark_mean_score") is not None else "<td class='num'>—</td>"),
            f"<td class='num'>{html.escape(memory)}</td>",
            f"<td>{html.escape(utilization)}</td>",
        ]) + "</tr>")
    provenance = data.get("provenance") or {}
    source = provenance.get("source_url", "https://huggingface.co/aj9o9/GLM-5.3-Flash-GGUF")
    revision = provenance.get("source_revision", "07c62fcdeaf1c05d22bd123c3da8058a1b1e63e2")[:12]
    return f"""
<h2>GLM-5.3-Flash: volledige hardwarematrix</h2>
<p>Dezelfde 313,33B/17,3B-actieve inference-trunk van GLM-5.3-Flash draait met automatische
CPU-offload op elke configuratie. Daardoor toont deze tabel niet alleen ruwe GPU-snelheid,
maar ook hoeveel extra VRAM en NVLink de decode versnellen. Beide V100’s zijn bovendien
afzonderlijk gemeten. Dit is een <em>layer-split</em>-matrix: GLM-5.3 draait via een
experimentele runtime met automatische CPU-offload en krijgt daarom bewust geen
onbewaakte tensor-split-duurtest. Eerdere llama.cpp-GGUF tensor-split-runs op de
V100's eindigden tweemaal in een driverhang; de onderliggende CUDA-kernel is niet
bewezen. Dat incident staat los van vLLM tensor parallel, dat per model apart wordt
vooraf gecontroleerd.</p>
<div class="tablewrap"><table class="sortable"><thead><tr><th>Configuratie</th>
<th data-sort-dir="desc">Decode t/s</th><th>Prompt t/s</th><th>vs. V100 #1</th>
<th>8-task score</th><th>VRAM GiB</th><th>GPU util.</th></tr></thead>
<tbody>{''.join(body)}</tbody></table></div>
<div class="legend"><span>Bron: <a href="{html.escape(source, quote=True)}">aj9o9/GLM-5.3-Flash-GGUF</a></span>
<span>publicatie 2026-09-01</span><span>revisie <code>{html.escape(revision)}</code></span>
<span>AJ-IQ2_XXS · 2,23 bpw · 87,35 GB</span></div>
"""


def glm53_completion_summary() -> str:
    """Keep the completed GLM result visible above the long sortable tables."""
    suite = next((row for row in load("well_known_suite_20260917.json").get("results", [])
                  if row.get("model") == "glm53-flash-aj-iq2xxs" and not row.get("error")), None)
    hardware = [row for row in load("hardware_scaling_glm53_20260921.json").get("results", [])
                if not row.get("error") and isinstance(row.get("completion_tokens_per_second"), (int, float))]
    if not suite or not hardware:
        return ""
    best = max(hardware, key=lambda row: row["completion_tokens_per_second"])
    score_parts = [
        rm.gsm8k_score(suite),
        (suite.get("bbh") or {}).get("mean_accuracy"),
        (suite.get("mmlu_sample") or {}).get("mean_accuracy"),
        (suite.get("humaneval") or {}).get("pass_at_1"),
    ]
    composite = sum(v for v in score_parts if isinstance(v, (int, float))) / 4
    return f"""
<section class="callout"><strong>✓ GLM-5.3-Flash AJ-IQ2_XXS voltooid.</strong>
De volledige huidige sandbox-suite staat in de tabel <em>Bekende benchmarks</em>: composite
{composite * 100:.0f}% · {fmt(suite.get('completion_tokens_per_second'))} t/s op de algemene run.
De zeven hardwaremetingen staan hieronder in <em>GLM-5.3-Flash: volledige hardwarematrix</em>;
de snelste is <strong>{html.escape(best.get('profile', '—'))}</strong> met
{fmt(best['completion_tokens_per_second'])} t/s. Dit betreft de werkende AJ-IQ2_XXS-quant;
REAP50 is afzonderlijk mislukt en is geen voltooide score.</section>
"""


def additional_gpu_serving_section() -> str:
    data = load("additional_gpu_serving_matrix_20260921.json")
    serial = data.get("serial_mean") or {}
    concurrent = data.get("concurrent_aggregate_mean_tokens_per_second")
    efficiency = data.get("concurrent_efficiency_vs_serial_capacity_sum")
    if not serial or concurrent is None:
        return ""
    a4000 = serial.get("a4000_qwen36", {})
    ada = serial.get("ada_qwen38", {})
    best_single = max(
        value for value in (a4000.get("wall_tokens_per_second"), ada.get("wall_tokens_per_second"))
        if isinstance(value, (int, float))
    )
    gain = 100 * (concurrent / best_single - 1)
    return f"""
<h2>Twee modellen parallel op de extra RTX-kaarten</h2>
<p>Drie herhalingen van identieke geforceerde 256-tokenrequests. Qwen3.6 draait uitsluitend
op fysieke GPU0 (A4000); Qwen3.8 uitsluitend op GPU3 (RTX 4000 Ada). Dit is een
parallel-servingcapaciteitsmeting, geen gecombineerde accuracy- of ensemblescore.</p>
<div class="grid"><div class="card"><small>A4000 · Qwen3.6</small>
<div class="metric">{fmt(a4000.get('wall_tokens_per_second'))} t/s</div><small>serial wall throughput</small></div>
<div class="card"><small>RTX 4000 Ada · Qwen3.8</small>
<div class="metric">{fmt(ada.get('wall_tokens_per_second'))} t/s</div><small>serial wall throughput</small></div>
<div class="card"><small>Beide modellen tegelijk</small><div class="metric">{fmt(concurrent)} t/s</div>
<small>aggregate · +{fmt(gain, 1)}% versus snelste losse endpoint</small></div>
<div class="card"><small>Parallel efficiency</small><div class="metric">{fmt(100 * efficiency, 1)}%</div>
<small>versus som van beide serial capaciteiten</small></div></div>
<div class="callout"><strong>Interpretatie.</strong> De extra kaarten leveren {fmt(concurrent)} t/s
gezamenlijke capaciteit voor twee gelijktijdige modellen bij {fmt(100 * efficiency, 1)}% van hun
opgetelde losse capaciteit. De kaarten hebben onderling geen NVLink; zet daarom bij voorkeur één
model per kaart. Gebruik A4000+Ada layer split alleen om een groter model passend te maken.</div>
"""


def benchmark_section(rows: list[dict]) -> str:
    if not rows:
        return ""
    header_cells = "".join(f"<th>{html.escape(label)}</th>" for label in TASK_LABELS.values())
    legend = "".join(
        f"<span><strong>{html.escape(label)}</strong>: {html.escape(desc)}</span>"
        for label, desc in (
            ("RIV", "RIV AU lifecycle JSON extraction, 6 fields, no hallucination"),
            ("Arith", "deterministic multi-step payout calculation"),
            ("Code", "generated max_drawdown() executed and checked vs. a reference implementation"),
            ("JSON", "strict schema + factual grounding (AAPL/US/not delisted)"),
            ("GICS", "exactly 5 lines, 5 real GICS sector names, no extra text"),
            ("Needle", "needle-in-haystack date recall inside a real repo doc"),
            ("Summary", "2-3 sentence summary of a real docstring, key numbers retained"),
            ("Refusal", "must abstain (not fabricate a date) when the passage is silent"),
        )
    )
    return f"""
<h2>Benchmarkscores (8 taken, deterministisch gescoord)</h2>
<p>Elke taak levert een objectieve, programmatisch gecontroleerde score 0-100% op -- geen LLM-jury.
Alle 7 lokale modellen zijn hierop getest op de dual-tensor 2xV100-configuratie (GPU 0/3 uitgesloten).</p>
<div class="tablewrap"><table><thead><tr><th>Model</th>{header_cells}<th>Gemiddeld</th></tr></thead>
<tbody>{benchmark_table(rows)}</tbody></table></div>
<div class="legend">{legend}</div>
<div class="callout"><strong>Opschoonregel (modellen &lt;30B die slechter scoren dan Qwen3.8-27B worden verwijderd):</strong>
geen enkel getest model &lt;30B parameters scoort lager dan de Qwen3.8-27B-referentie (85%) -- Qwen3.6-27B,
Gemma4-26B-A4B en Devstral Small 2 scoren allemaal hoger (88%), Qwen3.5-27B scoort exact gelijk (85%).
DeepSeek-R1-Qwen (32B) en Qwen2.5-72B (72,7B) vallen buiten de &lt;30B-regel. Er is dus niets verwijderd.
De <strong>Arith</strong>-kolom staat overal op 0%: de prompt verbiedt zichtbare redenering ("return only
the final numeric answer") en schakelt thinking-mode uit, wat exacte meerstaps mentale rekenkunde voor elk
lokaal getest model onmogelijk maakt zonder scratchpad -- een reëel, verwacht model-limiet, geen scoringsbug.</div>
"""


def main() -> int:
    rows = gguf_rows() + onecat_rows()
    throughput = throughput_rows(rows)
    bench_rows = benchmark_score_rows()
    wk_rows = well_known_rows()
    q = {row["profile"]: row for row in rows if row["model"] == "Qwen3.8-27B"}
    q_single = q.get("single-v100", {}).get("tps")
    q_tensor = q.get("dual-tensor", {}).get("tps")
    q_gain = 100 * (q_tensor / q_single - 1) if q_single and q_tensor else None
    dense72 = next((row for row in rows if row["model"] == "Qwen2.5-72B" and row["profile"] == "dual-tensor"), {})
    disk_root = Path(os.environ.get("VIRTUALV_DATA_ROOT", "/media/knight2/EDS2"))
    free_override = os.environ.get("VIRTUALV_DISK_FREE_GB")
    free_gb = (float(free_override) if free_override is not None else
               shutil.disk_usage(disk_root).free / 1_000_000_000 if disk_root.exists() else None)
    created_override = os.environ.get("VIRTUALV_DASHBOARD_CREATED")
    created = created_override or datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    document = f"""<!doctype html>
<html lang="nl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Node2 LLM hardware benchmark · V100 NVLink + RTX</title>
<style>
:root{{--bg:#071018;--panel:#0c1823;--panel2:#102332;--ink:#eaf2f8;--muted:#8ea5b5;--line:#1d3a4d;--cyan:#33d1c6;--orange:#ff9f43;--green:#7ee787}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 80% 0,#153447 0,transparent 34%),var(--bg);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif}}
main{{max-width:1240px;margin:auto;padding:54px 28px 80px}} h1{{font-size:clamp(36px,6vw,68px);line-height:1.02;letter-spacing:-.045em;margin:14px 0}} h2{{margin:42px 0 14px;font-size:25px;letter-spacing:-.02em}} h3{{margin:0 0 4px}} p{{color:var(--muted);max-width:900px}} a{{color:var(--cyan)}} .eyebrow{{color:var(--cyan);text-transform:uppercase;letter-spacing:.18em;font-size:12px;font-weight:800}}
.badges{{display:flex;flex-wrap:wrap;gap:8px;margin:22px 0 30px}} .badge,.pill{{border:1px solid var(--line);background:#0a1c28;border-radius:999px;padding:5px 10px;color:#bcd0dc;font-size:12px}} .badge.ok{{border-color:#246f67;color:var(--green)}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}} .card{{background:linear-gradient(145deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 12px 28px #0005}} .metric{{font-size:29px;font-weight:800;color:var(--orange);letter-spacing:-.03em}} .card small,td small,.barlabel small{{display:block;color:var(--muted);font-size:12px}}
.callout{{border-left:3px solid var(--cyan);padding:15px 18px;background:#0b202b;border-radius:0 12px 12px 0;margin:18px 0;color:#cce0e9}}
/* 2026-09-22: th{{position:sticky;top:0}} only sticks within ITS OWN scrolling
   ancestor. .tablewrap previously had overflow:auto but no bounded height, so
   it never scrolled internally -- the whole page scrolled past it instead,
   and the header scrolled away with everything else (user report: "header
   loopt boven uit het scherm"). Bounding the height turns .tablewrap into the
   actual scroll container, so the sticky header now works, and the resulting
   scrollbar (styled below) doubles as the horizontal "slider" the user asked
   for instead of an easy-to-miss thin native one. */
.tablewrap{{overflow:auto;max-height:74vh;border:1px solid var(--line);border-radius:16px;background:#091722;scrollbar-color:#3a6580 #0c1c28;scrollbar-width:thin}}
.tablewrap::-webkit-scrollbar{{width:12px;height:12px}}
.tablewrap::-webkit-scrollbar-track{{background:#0c1c28}}
.tablewrap::-webkit-scrollbar-thumb{{background:#3a6580;border-radius:8px;border:2px solid #0c1c28}}
.tablewrap::-webkit-scrollbar-thumb:hover{{background:var(--cyan)}}
table{{border-collapse:collapse;width:100%;min-width:980px;table-layout:fixed}} th,td{{padding:12px 13px;border-bottom:1px solid #173345;text-align:left;white-space:nowrap}} th{{position:sticky;top:0;background:#102737;color:#9fc0cf;font-size:11px;text-transform:uppercase;letter-spacing:.08em;z-index:2}} tr:hover td{{background:#0d2130}} .num{{font-variant-numeric:tabular-nums}} .hot{{color:var(--orange);font-weight:800}}
/* 2026-09-19: a long "Mixture-of-models -- method (members...)" label was
   forcing the whole table wider than the viewport (white-space:nowrap on
   every cell), pushing t/s/Hardware/Status off-screen without scrolling far
   right (user report + screenshot). Cap the model column and let ITS text
   wrap; every other column stays nowrap/compact so numbers stay aligned. */
/* Give the long model/ensemble descriptions a usable default. The right-edge
   grips on every column let the reader trade this space against score columns. */
/* 2026-09-22: fixed 440px let the model-name column eat well over 25% of a
   1240px main on any screen size and didn't respond to viewport width at all
   on narrower ones (user report). max-width:25vw caps it relative to the
   actual screen; the table's min-width:980px plus every other column staying
   nowrap is what makes the rest of the row overflow .tablewrap horizontally
   (scrollable, per the .tablewrap rule above) instead of squeezing. */
.tablewrap table thead th:first-child,.tablewrap table tbody td:first-child{{width:300px;min-width:220px;max-width:25vw;white-space:normal;word-break:break-word}}
.col-resizer{{position:absolute;z-index:3;top:0;right:-6px;width:12px;height:100%;cursor:col-resize;touch-action:none}}
.col-resizer::after{{content:'';position:absolute;top:28%;bottom:28%;left:5px;border-left:1px solid #4b7892;opacity:.8}}
.col-resizer:hover::after,.col-resizer:focus-visible::after{{border-color:var(--cyan);border-left-width:2px;opacity:1}}
body.column-resizing{{cursor:col-resize;user-select:none}}
.score-hi{{color:var(--green);font-weight:700}} .score-mid{{color:var(--orange);font-weight:700}} .score-lo{{color:#ff6b6b;font-weight:700}}
.legend{{display:flex;flex-wrap:wrap;gap:10px;margin:10px 0 0;color:var(--muted);font-size:12px}}
.chart{{background:#091722;border:1px solid var(--line);border-radius:16px;padding:20px}} .barrow{{display:grid;grid-template-columns:190px 1fr 65px;gap:12px;align-items:center;margin:11px 0}} .track{{height:14px;background:#122b3a;border-radius:20px;overflow:hidden}} .bar{{height:100%;background:linear-gradient(90deg,var(--cyan),var(--orange));border-radius:20px}} .barvalue{{font-variant-numeric:tabular-nums;text-align:right;font-weight:750}} .barlabel{{font-size:13px}}
.twocol{{display:grid;grid-template-columns:1fr 1fr;gap:14px}} code{{color:#bfe9e5;background:#08141c;padding:2px 6px;border-radius:5px}} footer{{margin-top:54px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}}
table.sortable th{{user-select:none}} .th-content{{display:inline-flex;align-items:center;gap:7px}}
.benchmark-link{{color:#9fc0cf;text-decoration:underline;text-decoration-style:dotted;text-underline-offset:3px}}
.benchmark-link:hover,.benchmark-link:focus{{color:var(--ink);text-decoration-style:solid}}
.sort-control{{width:23px;height:23px;min-width:23px;padding:0;border:1px solid #386079;border-radius:4px;background:#091722;color:#9fc0cf;cursor:pointer;font:11px/1 ui-sans-serif,system-ui,sans-serif}}
.sort-control::after{{content:'↕'}} .sort-control[data-sort-dir="asc"]::after{{content:'▲'}} .sort-control[data-sort-dir="desc"]::after{{content:'▼'}}
.sort-control:hover,.sort-control:focus-visible{{color:var(--ink);border-color:var(--cyan);outline:none}}
.benchmark-explanations{{scroll-margin-top:22px}} .explanation-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}
.explanation-grid article{{scroll-margin-top:22px;background:#091722;border:1px solid var(--line);border-radius:12px;padding:16px}}
.explanation-grid article:target{{border-color:var(--cyan);box-shadow:0 0 0 2px #33d1c633}}
.explanation-grid h4{{margin:0 0 5px;color:var(--cyan)}} .explanation-grid p{{margin:0 0 8px}}
@media(max-width:850px){{.grid,.twocol,.explanation-grid{{grid-template-columns:1fr 1fr}}.barrow{{grid-template-columns:130px 1fr 55px}}}} @media(max-width:560px){{main{{padding:32px 16px}}.grid,.twocol,.explanation-grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div class="eyebrow">node2 · gecontroleerde lokale meting</div><h1>LLM Hardware<br>Lab</h1>
<p>Een reproduceerbare vergelijking van lokale modellen en hardwareconfiguraties: het
2×V100-SXM2 NVLink-paar, een RTX A4000, een RTX 4000 Ada en beide RTX-kaarten als
onafhankelijke parallel-servinglaag. Iedere rij bewaart de werkelijk zichtbare fysieke GPU’s.</p>
<div class="badges"><span class="badge ok">✓ fysieke GPU-provenance</span><span class="badge ok">✓ PCI_BUS_ID vastgezet</span><span class="badge ok">✓ per-profiel telemetry</span><span class="badge">V100 NV6 · 6 links</span><span class="badge">64 GB HBM2 + 35 GB RTX</span><span class="badge">{created}</span></div>
<section class="grid"><div class="card"><small>Qwen3.8 tensor split</small><div class="metric">{fmt(q_tensor)} t/s</div><small>{fmt(q_gain,1)}% boven single V100</small></div>
<div class="card"><small>Dense schaaltest</small><div class="metric">{fmt(dense72.get('tps'))} t/s</div><small>Qwen2.5 72,7B · RIV {dense72.get('quality','—')}</small></div>
<div class="card"><small>1Cat DFlash2 B1</small><div class="metric">{fmt(next((r['tps'] for r in rows if r['model']=='Qwen3.8 + DFlash2' and 'B1' in r['profile']),None))} t/s</div><small>wall output throughput</small></div>
<div class="card"><small>Vrije modelopslag</small><div class="metric">{fmt(free_gb, 1)} GB</div><small>live bij lokale generatie; niet beschikbaar op CI</small></div></section>
<div class="callout"><strong>Hoofdconclusie.</strong> Layer split vergroot vooral capaciteit. Tensor split gebruikt beide V100’s echt parallel: Qwen3.8 wint {fmt(q_gain,1)}%, terwijl de 72,7B dense Qwen van 14,84 naar 24,29 t/s gaat (+63,7%). Voor Qwen3.8 single-stream latency blijft 1Cat + DFlash2 de snelste route; bij batch 4 wint target-only.</div>
<h2>Alle lokale resultaten</h2><p>Decode-t/s van llama.cpp en wall-output-t/s van 1Cat zijn apart gemeten; batch-4 is aggregaat. RIV is een brongebonden zesveldentest, geen algemene modelaccuracy.</p>
<div class="tablewrap"><table class="sortable"><thead><tr><th>Model</th><th>Engine</th><th>Profiel</th><th data-sort-dir="desc">Output t/s</th><th>Prompt t/s</th><th>Qwen3.8 speedup</th><th>RIV</th><th>VRAM GiB</th><th>GPU util.</th></tr></thead><tbody>{table(rows)}</tbody></table></div>
<section id="onecat-uitleg" class="benchmark-explanations" aria-labelledby="onecat-uitleg-title">
<h3 id="onecat-uitleg-title">Uitleg van de 1Cat-vLLM-rijen</h3>
<div class="explanation-grid">
<article id="onecat-target"><h4>Qwen3.8 target</h4><p>Het model draait alleen, zonder
hulpmodel. Elke token komt rechtstreeks uit één forward pass van Qwen3.8 zelf.</p>
<a href="#onecat-uitleg-title">Terug naar boven</a></article>
<article id="onecat-dflash2"><h4>Qwen3.8 + DFlash2</h4><p>Speculative decoding: een klein
draft-model (DFlash2) stelt meerdere tokens tegelijk voor, die Qwen3.8 in één stap
verifieert/accepteert. Bij lage gelijktijdigheid (B1) verlaagt dit de latency per
request merkbaar; bij hoge gelijktijdigheid (B4) verdwijnt dat voordeel omdat de
GPU dan toch al vol staat met werk van andere requests.</p>
<a href="#onecat-uitleg-title">Terug naar boven</a></article>
<article id="onecat-b1"><h4>B1 (batchgrootte 1)</h4><p>Eén gelijktijdig verzoek: meet
single-stream latency/doorvoer, het scenario van één interactieve gebruiker die op
antwoord wacht.</p><a href="#onecat-uitleg-title">Terug naar boven</a></article>
<article id="onecat-b4"><h4>B4 (batchgrootte 4)</h4><p>Vier gelijktijdige verzoeken:
meet aggregate doorvoer onder concurrency, het scenario van meerdere gebruikers of
achtergrondtaken tegelijk.</p><a href="#onecat-uitleg-title">Terug naar boven</a></article>
</div></section>
<h2>Throughputprofiel: alle gemeten modellen en configuraties</h2>
<p>Elke balk is een werkelijk gemeten configuratie, niet een afgeleide schatting. Het onderschrift
onder de modelnaam vermeldt steeds de fysieke GPU’s. <strong>Aggregate 2 requests</strong> telt
twee gelijktijdige endpoints op en is dus capaciteit, geen snelheid van één model. De brede
tabel erboven bevat de bijbehorende prompt-snelheid, VRAM en GPU-utilisatie.</p>
<div class="chart">{bars(throughput)}</div>
{benchmark_section(bench_rows)}
{glm53_completion_summary()}
{well_known_section(wk_rows)}
{specialist_section()}
{contamination_section()}
{matrix_gate_section()}
{hardware_scaling_section()}
{glm53_hardware_section()}
{additional_gpu_serving_section()}
{large_model_provenance_section()}
{qwen38_flash_next_vllm_section()}
{candidate_research_section()}
<h2>Wat gebruikt NVLink het best?</h2><div class="twocol"><div class="card"><h3>Tensor split</h3><p>Beste llama.cpp-topologie. Qwen3.8 bereikt circa 85–86% gemiddelde utiliteit per V100; Qwen-72B circa 95–97%. De zes NVLinks maken de benodigde tensorcollectives praktisch.</p></div><div class="card"><h3>Layer split</h3><p>Goed om modellen te laten passen, maar geen decodeversnelling voor 27B: 33,33 versus 33,53 t/s single. De lagen worden grotendeels na elkaar uitgevoerd.</p></div><div class="card"><h3>1Cat TP2</h3><p>Qwen3.8 NVFP4 gebruikt beide V100’s op 100%. DFlash2 is sterk bij B1; target-only schaalt beter bij vier gelijktijdige requests.</p></div><div class="card"><h3>72B–100B envelope</h3><p>72,7B dense Q4_K_M gebruikt circa 22,27 GiB per kaart en is stabiel op 8K. Een dense 100B Q4 zou te weinig allocator- en KV-marge laten; de betrouwbare grens ligt op deze machine daarom rond 70–80B.</p></div></div>
<h2>Meetintegriteit</h2><div class="tablewrap"><table><tbody><tr><th>V100-modelranglijst</th><td>Fysieke GPU 1 + 2: Tesla V100-SXM2-32GB; A4000/Ada zijn onzichtbaar en uitgesloten</td></tr><tr><th>RTX-profielen</th><td>GPU0 = RTX A4000 15GB; GPU3 = RTX 4000 Ada 20GB. Single-card, PCIe layer split en parallel serving worden als verschillende configuraties gerapporteerd.</td></tr><tr><th>Device contract</th><td><code>CUDA_DEVICE_ORDER=PCI_BUS_ID</code> plus een expliciete <code>CUDA_VISIBLE_DEVICES</code>-set; de llama.cpp device probe en telemetry moeten exact met het profiel overeenkomen</td></tr><tr><th>Interconnect</th><td>NV6; 6 actieve links per V100, elk 25,781 GB/s gerapporteerd. Tussen A4000 en Ada bestaat geen NVLink.</td></tr><tr><th>Temperatuur</th><td>Maximaal 65°C in de 72B tensor-run</td></tr><tr><th>Brondata</th><td>JSON-artefacten in <code>reports/</code>; pagina wordt daar rechtstreeks uit gegenereerd</td></tr></tbody></table></div>
<footer>Gegenereerd door <code>scripts/benchmarks/build_dual_v100_html.py</code>. Officiële modelbron: <a href="https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GGUF">Qwen/Qwen2.5-72B-Instruct-GGUF</a>. 1Cat-referentie: <a href="https://github.com/1CatAI/1Cat-vLLM/blob/main/RELEASE.md">1Cat-vLLM 1.5 release</a>.</footer>
<script>
(function(){{
  function cellValue(td, forceText){{
    if (td.dataset.sort !== undefined && td.dataset.sort !== '') return parseFloat(td.dataset.sort);
    var t = td.textContent.trim();
    if (t === '—' || t === '') return null;
    // Hardware values can start with a digit ("2× V100"), but that is a
    // label, not a numeric quantity. Keep all descriptor columns alphabetical.
    if (forceText) return t.toLocaleLowerCase('nl-NL');
    var n = parseFloat(t.replace('%','').replace('×',''));
    return isNaN(n) ? t.toLowerCase() : n;
  }}
  document.querySelectorAll('table.sortable').forEach(function(table){{
    var headers = table.querySelectorAll('thead th');
    headers.forEach(function(th, idx){{
      var button = th.querySelector('.sort-control');
      if (!button) {{
        var initial = th.getAttribute('data-sort-dir');
        th.removeAttribute('data-sort-dir');
        var content = document.createElement('span');
        content.className = 'th-content';
        while (th.firstChild) content.appendChild(th.firstChild);
        button = document.createElement('button');
        button.className = 'sort-control';
        button.type = 'button';
        button.setAttribute('aria-label', 'Sorteer op ' + content.textContent.trim());
        button.title = button.getAttribute('aria-label');
        if (initial) button.setAttribute('data-sort-dir', initial);
        content.appendChild(button);
        th.appendChild(content);
      }}
      button.addEventListener('click', function(event){{
        event.preventDefault();
        event.stopPropagation();
        var dir = button.getAttribute('data-sort-dir') === 'asc' ? 'desc' : 'asc';
        headers.forEach(function(h){{
          var control = h.querySelector('.sort-control');
          if (control) control.removeAttribute('data-sort-dir');
        }});
        button.setAttribute('data-sort-dir', dir);
        var tbody = table.querySelector('tbody');
        var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));
        var headerText = headers[idx].textContent.trim().toLocaleLowerCase('nl-NL');
        var textColumn = /^(model|engine|profiel|hardware|status|configuratie)$/.test(headerText);
        rows.sort(function(a, b){{
          var av = cellValue(a.children[idx], textColumn);
          var bv = cellValue(b.children[idx], textColumn);
          if (av === null) return 1;
          if (bv === null) return -1;
          if (av < bv) return dir === 'asc' ? -1 : 1;
          if (av > bv) return dir === 'asc' ? 1 : -1;
          return 0;
        }});
        rows.forEach(function(row){{ tbody.appendChild(row); }});
      }});
    }});
  }});
  // Resize is deliberately separate from sorting: drag the thin grip on the
  // right of a header; click the square button to sort. Widths are per-table
  // and persisted locally, so a dashboard refresh keeps the chosen layout.
  // 2026-09-22 fix: the storage key used to be the table's raw DOM position
  // (tableIndex). Every dashboard rebuild can add/remove/reorder sections
  // (new models, new tables), so "the 3rd table" stops being the same table
  // -- a saved narrow width from resizing one table's columns could silently
  // get applied to a DIFFERENT table's columns next load, squeezing a column
  // down toward the 110px floor and cramming its sort button into unusable
  // space (user report: sort square "hidden" on the specialist table).
  // Keying on the table's own header text is stable across rebuilds as long
  // as that table's columns don't change.
  document.querySelectorAll('.tablewrap table').forEach(function(table){{
    var headers = Array.prototype.slice.call(table.querySelectorAll('thead th'));
    if (!headers.length) return;
    var signature = headers.map(function(th){{ return th.textContent.trim(); }}).join('|');
    var storageKey = 'llm-hardware-column-widths-v2:' + signature;
    var saved = {{}};
    try {{ saved = JSON.parse(localStorage.getItem(storageKey) || '{{}}'); }} catch (ignore) {{}}
    headers.forEach(function(th, index){{
      if (saved[index]) th.style.width = saved[index] + 'px';
      var grip = document.createElement('span');
      grip.className = 'col-resizer';
      grip.tabIndex = 0;
      grip.setAttribute('role', 'separator');
      grip.setAttribute('aria-orientation', 'vertical');
      grip.setAttribute('aria-label', 'Wijzig breedte van ' + th.textContent.trim());
      grip.title = 'Sleep om deze kolom breder of smaller te maken';
      th.appendChild(grip);
      function begin(clientX){{
        var startX = clientX;
        var startColumn = th.getBoundingClientRect().width;
        var startTable = table.getBoundingClientRect().width;
        document.body.classList.add('column-resizing');
        function move(event){{
          var width = Math.max(110, Math.min(900, startColumn + event.clientX - startX));
          th.style.width = width + 'px';
          // Keep other cells aligned and expand the scrollable table only when
          // necessary; shrinking never makes it narrower than the CSS minimum.
          table.style.width = Math.max(980, startTable + width - startColumn) + 'px';
        }}
        function end(){{
          document.body.classList.remove('column-resizing');
          document.removeEventListener('pointermove', move);
          document.removeEventListener('pointerup', end);
          saved[index] = Math.round(th.getBoundingClientRect().width);
          try {{ localStorage.setItem(storageKey, JSON.stringify(saved)); }} catch (ignore) {{}}
        }}
        document.addEventListener('pointermove', move);
        document.addEventListener('pointerup', end, {{once:true}});
      }}
      grip.addEventListener('pointerdown', function(event){{
        event.preventDefault(); event.stopPropagation(); begin(event.clientX);
      }});
      grip.addEventListener('keydown', function(event){{
        if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
        event.preventDefault();
        var width = Math.max(110, Math.min(900, th.getBoundingClientRect().width +
          (event.key === 'ArrowRight' ? 24 : -24)));
        th.style.width = width + 'px';
        saved[index] = Math.round(width);
        try {{ localStorage.setItem(storageKey, JSON.stringify(saved)); }} catch (ignore) {{}}
      }});
    }});
  }});
}})();
</script>
</main></body></html>"""
    OUT.write_text(document)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
