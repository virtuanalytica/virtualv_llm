#!/usr/bin/env python3
"""Optional high-difficulty specialist benchmarks.

This is intentionally separate from the fast, general-purpose battery.  It
records a result only for models evaluated after this protocol was introduced;
old rows therefore remain blank rather than being mistaken for zeroes.

Every lane reads a local, editable question pack. Chemistry/physics use GPQA
Diamond only as a difficulty reference, while MMMU-Pro and VBench are analogous
references for vision and video. Scores are therefore less exposed to public
benchmark contamination. Native diffusion-video remains a separate model type.
"""
from __future__ import annotations

import json
import csv
import hashlib
import shutil
import subprocess
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
OUT = REPORTS / "specialist_suite_20260922.json"
PRIVATE_PACK_DIR = ROOT / "data" / "eval_cache" / "specialist_packs"
PACK_DIR = ROOT / "config" / "specialist_benchmarks"
PROTOCOL = "v2-private-specialist-packs-vision-video-iq-eq-fq-qq-20260922"
SPECIALISTS = ("chemistry", "physics", "vision", "video", "iq", "eq", "fq", "qq")

# Sources and publication dates live in the result as well as in this code, so
# a detached JSON/HTML artifact always preserves its provenance.
SOURCES = {
    "chemistry": {"name": "EDS editable private chemistry holdout v1; difficulty target GPQA Diamond",
                  "publication_date": "2026-09-22", "url": "https://arxiv.org/abs/2311.12022", "samples": "enabled CSV rows"},
    "physics": {"name": "EDS editable private physics holdout v1; difficulty target GPQA Diamond",
                "publication_date": "2026-09-22", "url": "https://arxiv.org/abs/2311.12022", "samples": "enabled CSV rows"},
    "vision": {"name": "EDS synthetic object recognition + spatial reasoning holdout; MMMU-Pro difficulty reference", "publication_date": "2026-09-22", "url": "https://arxiv.org/abs/2409.02813", "samples": "enabled CSV rows", "max_wall_sec": 300},
    "video": {"name": "EDS programmatic MP4 artifact holdout; VBench metric reference", "publication_date": str(date.today()), "url": "https://arxiv.org/abs/2408.06072", "samples": "enabled CSV rows", "max_wall_sec": 300},
    "iq": {"name": "EDS abstract/rule reasoning holdout (not a clinical IQ test)", "publication_date": "2026-09-22", "url": "", "samples": "enabled CSV rows"},
    "eq": {"name": "EDS social-emotional reasoning holdout (not a clinical EQ test)", "publication_date": "2026-09-22", "url": "", "samples": "enabled CSV rows"},
    "fq": {"name": "EDS actuator/robot physics simulation holdout", "publication_date": "2026-09-22", "url": "", "samples": "enabled CSV rows"},
    "qq": {"name": "EDS quantum systems and quantum chemistry holdout", "publication_date": "2026-09-22", "url": "", "samples": "enabled CSV rows"},
}


def _answer_letter(text: str) -> str | None:
    import re
    match = re.search(r"\b([ABCD])\b", text.upper())
    return match.group(1) if match else None


def custom_questions_path(area: str) -> Path:
    """Prefer an ignored private pack; use the tracked starter only initially."""
    private = PRIVATE_PACK_DIR / f"{area}.csv"
    if private.exists():
        return private
    if area in {"chemistry", "physics"}:
        return ROOT / "config" / "benchmark_specialist_holdout_v1.csv"
    return PACK_DIR / f"{area}.csv"


def _custom_rows(area: str, require_mcq: bool = True) -> list[dict[str, str]]:
    """Read a user-editable blind pack; answer keys never enter prompts."""
    CUSTOM_QUESTIONS = custom_questions_path(area)
    if not CUSTOM_QUESTIONS.exists():
        raise RuntimeError(f"missing editable question CSV: {CUSTOM_QUESTIONS}")
    with CUSTOM_QUESTIONS.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"id", "discipline", "enabled", "prompt"}
    if require_mcq:
        required |= {"option_a", "option_b", "option_c", "option_d", "answer"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise RuntimeError(f"CSV missing columns: {sorted(missing)}")
    selected = [row for row in rows if row["discipline"].strip().lower() == area and row["enabled"].strip().lower() in {"1", "true", "yes"}]
    bad = [row.get("id", "?") for row in selected if require_mcq and row["answer"].strip().upper() not in "ABCD"]
    if bad:
        raise RuntimeError(f"CSV invalid answer letter for: {bad}")
    if not selected:
        raise RuntimeError(f"CSV contains no enabled {area} questions")
    return selected


def _run_custom(area: str, complete: Callable[[str, int], str]) -> dict[str, Any]:
    rows = _custom_rows(area)
    pack = custom_questions_path(area)
    correct, details = 0, []
    for row in rows:
        prompt = ("You are answering an original, expert-level " + area + " holdout question.\n\n" + row["prompt"] + "\n\n" +
                  "\n".join(f"{letter}. {row[f'option_{letter.lower()}']}" for letter in "ABCD") +
                  "\n\nReason privately, then output only the single answer letter.")
        predicted, target = _answer_letter(complete(prompt, 768)), row["answer"].strip().upper()
        correct += int(predicted == target)
        details.append({"id": row["id"], "correct": predicted == target, "target": target, "prediction": predicted})
    return {"status": "complete", "accuracy": round(correct / len(rows), 4), "n_samples": len(rows),
            "question_pack": str(pack.relative_to(ROOT)), "pack_version": pack.stem,
            "pack_sha256": hashlib.sha256(pack.read_bytes()).hexdigest(), "samples": details}


def _run_vision(vision_complete: Callable[[str, Path, int], str] | None) -> dict[str, Any]:
    if vision_complete is None:
        return {"status": "unsupported", "reason": "text-only endpoint; requires --vision-capable and a VLM adapter", "n_samples": 0}
    rows, correct, details = _custom_rows("vision"), 0, []
    pack = custom_questions_path("vision")
    for row in rows:
        asset = ROOT / row.get("asset_path", "")
        if not asset.exists():
            raise RuntimeError(f"vision asset missing for {row['id']}: {asset}")
        prompt = (row["prompt"] + "\n\n" + "\n".join(f"{letter}. {row[f'option_{letter.lower()}']}" for letter in "ABCD") +
                  "\n\nInspect the image. Reason privately, then output only the answer letter.")
        predicted, target = _answer_letter(vision_complete(prompt, asset, 768)), row["answer"].strip().upper()
        correct += int(predicted == target)
        details.append({"id": row["id"], "correct": predicted == target, "target": target, "prediction": predicted})
    return {"status": "complete", "accuracy": round(correct / len(rows), 4), "n_samples": len(rows),
            "question_pack": str(pack.relative_to(ROOT)), "pack_version": pack.stem,
            "pack_sha256": hashlib.sha256(pack.read_bytes()).hexdigest(), "samples": details}


def _run_video(complete: Callable[[str, int], str]) -> dict[str, Any]:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        return {"status": "unavailable", "reason": "ffmpeg/ffprobe not installed", "n_samples": 0}
    started, passed, details = time.monotonic(), 0, []
    with tempfile.TemporaryDirectory(prefix="specialist-video-") as tmp:
        rows = _custom_rows("video", require_mcq=False)
        for index, row in enumerate(rows):
            if time.monotonic() - started > SOURCES["video"]["max_wall_sec"]:
                break
            graph = complete(row["prompt"], 512).strip().replace("```", "").strip()
            output = Path(tmp) / f"task-{index}.mp4"
            # The model supplies data, never an executable command.  ffmpeg is
            # invoked with a fixed argv and controlled output path.
            if len(graph) > 8_000 or any(token in graph for token in ("\n", "\r", "`", "$", "../")):
                details.append({"passed": False, "reason": "unsafe/oversized filtergraph"}); continue
            try:
                proc = subprocess.run([
                    "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=1280x720:r=24:d=7",
                    "-filter_complex", graph, "-t", "7", "-pix_fmt", "yuv420p", str(output),
                ], cwd=tmp, capture_output=True, text=True, timeout=70)
                probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", str(output)], capture_output=True, text=True, timeout=15)
                duration = float(probe.stdout.strip()) if probe.returncode == 0 else 0.0
                ok = proc.returncode == 0 and output.exists() and output.stat().st_size > 20_000 and 4.5 <= duration <= 7.0
                passed += int(ok); details.append({"passed": ok, "duration_sec": round(duration, 2)})
            except (subprocess.TimeoutExpired, ValueError) as exc:
                details.append({"passed": False, "reason": type(exc).__name__})
    return {"status": "complete", "accuracy": round(passed / len(rows), 4), "n_samples": len(rows),
            "elapsed_sec": round(time.monotonic() - started, 1), "samples": details}


def _run_fq(complete: Callable[[str, int], str]) -> dict[str, Any]:
    """Score robot actuator commands through a deterministic differential-drive model."""
    rows, correct, details = _custom_rows("fq", require_mcq=False), 0, []
    for row in rows:
        raw = complete(row["prompt"] + "\nOutput only JSON with left_rad_s, right_rad_s and duration_s.", 256)
        try:
            data = json.loads(raw.strip().replace("```json", "").replace("```", ""))
            left, right, duration = (float(data[key]) for key in ("left_rad_s", "right_rad_s", "duration_s"))
            radius, base = float(row["wheel_radius_m"]), float(row["wheelbase_m"])
            omega, speed = radius * (right - left) / base, radius * (right + left) / 2
            if abs(omega) < 1e-9:
                x, y = speed * duration, 0.0
            else:
                x, y = speed / omega * __import__("math").sin(omega * duration), speed / omega * (1 - __import__("math").cos(omega * duration))
            heading = omega * duration * 180 / __import__("math").pi
            error = max(abs(x - float(row["target_x_m"])), abs(y - float(row["target_y_m"])), abs(heading - float(row["target_heading_deg"])) / 180)
            ok = duration > 0 and duration <= float(row.get("max_duration_s") or 10) and error <= float(row.get("tolerance") or .03)
            correct += int(ok); details.append({"id": row["id"], "correct": ok, "max_normalized_error": round(error, 4)})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            details.append({"id": row["id"], "correct": False, "reason": type(exc).__name__})
    return {"status": "complete", "accuracy": round(correct / len(rows), 4), "n_samples": len(rows), "samples": details,
            "simulator": "differential_drive_v1"}


def run_specialists(model: str, complete: Callable[[str, int], str], selected: tuple[str, ...] = SPECIALISTS,
                    vision_complete: Callable[[str, Path, int], str] | None = None, out: Path = OUT,
                    access_profile: str = "sandbox") -> dict[str, Any]:
    unknown = set(selected) - set(SPECIALISTS)
    if unknown:
        raise ValueError(f"unknown specialists: {sorted(unknown)}")
    result: dict[str, Any] = {"model": model, "protocol": PROTOCOL, "sources": SOURCES,
                              "access_profile": access_profile, "results": {}}
    for specialist in selected:
        try:
            if specialist in ("chemistry", "physics", "iq", "eq", "qq"):
                score = _run_custom(specialist, complete)
            elif specialist == "vision":
                score = _run_vision(vision_complete)
            elif specialist == "video":
                score = _run_video(complete)
            else:
                score = _run_fq(complete)
        except Exception as exc:
            score = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}", "n_samples": 0}
        result["results"][specialist] = score
    payload = json.loads(out.read_text()) if out.exists() else {"suite": "specialist", "results": []}
    payload["protocol"] = PROTOCOL
    payload["results"] = [r for r in payload.get("results", []) if r.get("model") != model] + [result]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp"); tmp.write_text(json.dumps(payload, indent=2) + "\n"); tmp.replace(out)
    return result
