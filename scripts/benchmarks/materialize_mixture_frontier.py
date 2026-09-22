#!/usr/bin/env python3
"""Materialize the best measured post-hoc ensemble for sizes 2 through 6.

Selection uses the existing four-task benchmark samples and is explicitly
reported as selection-set evidence.  TruthfulQA is routed to the strongest
measured member because its free-text answers cannot be majority-voted.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/well_known_suite_20260917.json"
OUT = ROOT / "reports/optimized_mixture_frontier_20260921.json"
OPTIMIZER = ROOT / "scripts/benchmarks/optimize_model_mixture.py"
MATERIALIZER = ROOT / "scripts/benchmarks/mixture_of_models.py"


def truthfulqa_score(row: dict) -> float:
    metrics = row.get("truthfulqa_gen") or {}
    value = metrics.get("rougeL_acc,none")
    return float(value) if isinstance(value, (int, float)) else -1.0


def main() -> int:
    report = json.loads(REPORT.read_text())
    rows = {row.get("model"): row for row in report.get("results", [])}
    frontier = []
    for size in range(2, 7):
        run = subprocess.run(
            ["python3", str(OPTIMIZER), "--report", str(REPORT),
             "--size", str(size), "--top", "1"],
            cwd=ROOT, check=True, text=True, capture_output=True,
        )
        search = json.loads(run.stdout)
        best = search["results"][0]
        members = best["members"]
        specialist = max(members, key=lambda name: truthfulqa_score(rows[name]))
        name = f"mixture-optimized-{size}"
        subprocess.run(
            ["python3", str(MATERIALIZER), *members, "--out", str(REPORT),
             "--name", name, "--truthfulqa-specialist", specialist],
            cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
        )
        frontier.append({
            **best,
            "name": name,
            "truthfulqa_specialist": specialist,
            "selection_warning": (
                "Optimized on these same samples; validate on a fresh held-out run "
                "before treating the score as a production estimate."
            ),
        })
    payload = {
        "method": "exhaustive best combination per ensemble size",
        "sizes": [2, 3, 4, 5, 6],
        "score_used_for_selection": "unweighted GSM8K + BBH + MMLU + HumanEval mean",
        "results": frontier,
    }
    temporary = OUT.with_suffix(OUT.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(OUT)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
