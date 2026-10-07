#!/usr/bin/env python3
"""Tool experts for the specialist lanes where a bare model scores near zero.

An expert is the same model plus one deterministic tool and a bounded
check-and-repair loop:

* FQ: the model proposes wheel speeds, a forward-kinematics simulator reports
  where the robot actually ends, and the model corrects its own command.
* Video: the model's FFmpeg filtergraph is rendered in the suite's fixed
  sandbox; on failure it gets FFmpeg's error and repairs the graph.

Neither tool sees the answer key. The simulator is physics and the target is
in the task text; FFmpeg only says whether a graph renders. Expert rows are
written to their own report and are never mixed with bare-model rows: the
bare model is scored in the same run on the same items for a paired view.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen

import specialist_suite as suite
from result_store import upsert_result

OUT = suite.REPORTS / "specialist_experts.json"
Complete = Callable[[str, int], str]
MAX_ROUNDS = 3


def _command(raw: str) -> dict | None:
    try:
        data = json.loads(raw.strip().replace("```json", "").replace("```", ""))
        return {key: float(data[key]) for key in
                ("left_rad_s", "right_rad_s", "duration_s", "wheel_radius_m", "wheelbase_m")}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def fq_expert(complete: Complete, trace: list | None = None) -> Complete:
    """Wrap ``complete`` so an actuator command is simulated and self-corrected."""
    def expert(prompt: str, max_tokens: int) -> str:
        task = (prompt + " Also include wheel_radius_m and wheelbase_m exactly as the task states them.")
        raw, rounds = complete(task, max_tokens), 1
        while rounds < MAX_ROUNDS:
            command = _command(raw)
            if command is None:
                feedback = "Your previous output was not the requested JSON object."
            else:
                x, y, heading = suite.differential_drive(
                    command["left_rad_s"], command["right_rad_s"], command["duration_s"],
                    command["wheel_radius_m"], command["wheelbase_m"])
                feedback = (f"A forward-kinematics simulation of your command {raw.strip()} ends at "
                            f"x={x:.3f} m, y={y:.3f} m, heading={heading:.1f} degrees.")
            revised = complete(f"{task}\n\n{feedback}\nCompare that with the target in the task. If it matches, "
                               "output the same JSON again; otherwise output corrected JSON. Output only JSON.",
                               max_tokens)
            rounds += 1
            if _command(revised) == command and command is not None:
                raw = revised
                break
            raw = revised
        if trace is not None:
            trace.append(rounds)
        return raw
    return expert


def video_expert(complete: Complete, trace: list | None = None) -> Complete:
    """Wrap ``complete`` so a filtergraph is rendered and repaired from FFmpeg's error."""
    def expert(prompt: str, max_tokens: int) -> str:
        raw, rounds = complete(prompt, max_tokens), 1
        with tempfile.TemporaryDirectory(prefix="video-expert-") as tmp:
            while True:
                graph = suite.clean_filtergraph(raw)
                if suite.unsafe_filtergraph(graph):
                    problem = "It contains a newline, backtick, dollar sign or parent path, or is too long."
                else:
                    try:
                        ok, duration, stderr = suite.render_filtergraph(graph, Path(tmp) / f"try-{rounds}.mp4")
                    except Exception as exc:  # timeout or unreadable probe: treat as a failed render
                        ok, duration, stderr = False, 0.0, type(exc).__name__
                    if ok:
                        break
                    problem = (f"FFmpeg output (duration {duration:.1f}s): {stderr.strip()[-500:]}"
                               if stderr.strip() else f"The file was missing, too small or {duration:.1f}s long.")
                if rounds >= MAX_ROUNDS:
                    break
                raw = complete(f"{prompt}\n\nYour previous filter_complex was:\n{graph[:1500]}\n\nIt did not "
                               f"render an acceptable video. {problem}\nReturn only a corrected filter_complex "
                               "expression on a single line.", max_tokens)
                rounds += 1
        if trace is not None:
            trace.append(rounds)
        return raw
    return expert


def http_complete(base_url: str, model: str, timeout: int = 600, attempts: int = 4,
                  backoff_sec: float = 20.0) -> Complete:
    """Chat completion with retries: one transient provider error (a 502 on
    2026-10-07) must not discard a run of several hundred requests. A request
    that keeps failing still raises; it is never scored as a wrong answer."""
    def complete(prompt: str, max_tokens: int) -> str:
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": max_tokens, "temperature": 0}).encode()
        for attempt in range(1, attempts + 1):
            request = Request(f"{base_url.rstrip('/')}/v1/chat/completions", data=body,
                              headers={"Content-Type": "application/json"})
            try:
                with urlopen(request, timeout=timeout) as response:
                    return json.loads(response.read())["choices"][0]["message"]["content"] or ""
            except OSError:  # URLError, HTTPError and socket timeouts
                if attempt == attempts:
                    raise
                time.sleep(backoff_sec * attempt)
        raise AssertionError("unreachable")
    return complete


FQ_PACK_V2 = suite.PACK_DIR / "fq_v2_parametric.csv"
FQ_COLUMNS = ("id", "discipline", "enabled", "prompt", "wheel_radius_m", "wheelbase_m", "target_x_m",
              "target_y_m", "target_heading_deg", "max_duration_s", "tolerance", "author", "created_date", "notes")


FQ_PACK_V3 = suite.PACK_DIR / "fq_v3_parametric.csv"
FQ_LIMIT_SENTENCE = " The whole manoeuvre must take at most 10 seconds."


def fq_parametric_rows_v3() -> list[dict[str, str]]:
    """The v2 targets with the duration limit stated in the task text.

    v2 scored a command as wrong when it ran longer than 10 s although the
    prompt never said so; 6 of the expert's 8 misses on v2 were that.
    """
    return [{**row, "id": row["id"].replace("fq2-", "fq3-"), "prompt": row["prompt"] + FQ_LIMIT_SENTENCE}
            for row in fq_parametric_rows()]


def fq_parametric_rows(count: int = 73, seed: int = 20261007) -> list[dict[str, str]]:
    """Reachable differential-drive targets with exact ground truth.

    Each target is one straight segment or one circular arc from the origin,
    so a single (left, right, duration) command reaches it; the scorer's own
    simulator decides. 73 is the smallest pack on which a perfect score has a
    95% Wilson lower bound of 0.95. The seed fixes the pack.
    """
    import math
    import random

    rng, rows, seen = random.Random(seed), [], set()
    while len(rows) < count:
        radius, base = rng.choice((0.04, 0.05, 0.06, 0.08)), rng.choice((0.25, 0.30, 0.40))
        if rng.random() < 0.2:
            x, y, heading = round(rng.uniform(0.3, 2.0), 2), 0.0, 0
        else:
            arc, heading = round(rng.uniform(0.3, 1.2), 2), rng.choice((-150, -120, -90, -60, -30, 30, 60, 90, 120, 150))
            theta = math.radians(heading)
            x, y = round(arc * math.sin(abs(theta)), 2), round(math.copysign(arc * (1 - math.cos(theta)), heading), 2)
        if (x, y, heading, radius, base) in seen:
            continue
        seen.add((x, y, heading, radius, base))
        rows.append(dict(zip(FQ_COLUMNS, (
            f"fq2-{len(rows) + 1:03d}", "fq", "true",
            f"A differential-drive robot starts at (0 0) heading 0 degrees. Wheel radius is {radius:.2f} m and "
            f"wheelbase is {base:.2f} m. Select actuator velocities and duration to reach approximately "
            f"({x:.2f} {y:.2f}) at heading {heading} degrees.",
            f"{radius:.2f}", f"{base:.2f}", f"{x:.2f}", f"{y:.2f}", str(heading), "10", "0.04",
            "EDS generator", "2026-10-07", f"parametric, seed {seed}"))))
    return rows


def write_fq_pack(path: Path = FQ_PACK_V2, rows: list[dict[str, str]] | None = None) -> None:
    import csv
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FQ_COLUMNS)
        writer.writeheader()
        writer.writerows(rows or fq_parametric_rows())


def read_pack(path: Path) -> list[dict[str, str]]:
    import csv
    with path.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["enabled"].strip().lower() == "true"]


def run(name: str, complete: Complete, out: Path = OUT, fq_pack: Path | None = None, video: bool = True) -> dict:
    """Score bare model and expert on the same FQ and video items."""
    import hashlib
    fq_rows = read_pack(fq_pack) if fq_pack else None
    fq_rounds: list[int] = []
    video_rounds: list[int] = []
    result = {
        "model": name if not fq_pack else f"{name}@{fq_pack.stem}", "protocol": suite.PROTOCOL,
        "max_rounds": MAX_ROUNDS,
        "bare": {"fq": suite._run_fq(complete, fq_rows)},
        "expert": {"fq": suite._run_fq(fq_expert(complete, fq_rounds), fq_rows)},
        # Rounds actually used per item: the cost side of the expert's score.
        "expert_rounds": {"fq": fq_rounds},
    }
    if fq_pack:
        result["fq_pack"] = {"file": fq_pack.name, "sha256": hashlib.sha256(fq_pack.read_bytes()).hexdigest()}
    if video:
        result["bare"]["video"] = suite._run_video(complete)
        result["expert"]["video"] = suite._run_video(video_expert(complete, video_rounds))
        result["expert_rounds"]["video"] = video_rounds
    upsert_result(out, result, {"suite": "specialist tool experts (bare vs expert, same items)", "results": []})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="row name, normally the model's identifier in the main report")
    parser.add_argument("--url", required=True, help="server root of an OpenAI-compatible endpoint")
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--fq-pack", type=Path, help="score FQ on this pack instead of the starter pack")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    result = run(args.name, http_complete(args.url, args.served_model), args.out, args.fq_pack, not args.no_video)
    for lane in result["bare"]:
        bare, expert = result["bare"][lane], result["expert"][lane]
        print(f"{lane}: bare {bare.get('accuracy')} -> expert {expert.get('accuracy')} "
              f"(n={expert.get('n_samples')}, rounds {result['expert_rounds'][lane]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
