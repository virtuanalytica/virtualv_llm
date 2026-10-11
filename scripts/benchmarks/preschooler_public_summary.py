#!/usr/bin/env python3
"""Export only audited scalar results from local private Preschooler runs.

Raw prompts, predictions, diagnostic examples and file paths remain in the
owner-only benchmark directory. This summary is safe to publish after review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROSTER = ROOT / "config/preschooler_teacher_roster.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_profile(folder: Path, model: dict) -> tuple[dict, dict]:
    status = json.loads((folder / "status.json").read_text())
    report_path = folder / "private" / "private-result.json"
    report = json.loads(report_path.read_text())
    rows = report.get("results", [])
    if (status.get("status") != "complete_private" or status.get("alias") != model["alias"]
            or status.get("revision") != model["revision"]
            or status.get("canary", {}).get("status") != "passed"
            or report.get("suite") != "private" or len(rows) != 1):
        raise ValueError(f"{model['alias']}: incomplete or mismatched private profile")
    row = rows[0]
    if (row.get("status") != "complete" or row.get("model") != model["alias"]
            or row.get("promotion_eligible") is not False
            or row.get("specialist", {}).get("complete") is not True
            or row.get("specialist", {}).get("items", 0) < 20
            or row.get("throughput_probe", {}).get("completion_tokens") != 256
            or row.get("gpu_board_wh_per_answer", 0) <= 0
            or row.get("requests", 0) <= 0):
        raise ValueError(f"{model['alias']}: private score or energy gate incomplete")
    provenance = row.get("provenance", {})
    commitments = {"specialist_pack_sha256": provenance.get("specialist_pack_sha256"),
                   "holdout_pack_sha256": provenance.get("holdout_pack_sha256")}
    if (not isinstance(commitments["specialist_pack_sha256"], dict)
            or not commitments["specialist_pack_sha256"]
            or not isinstance(commitments["holdout_pack_sha256"], str)):
        raise ValueError(f"{model['alias']}: pack commitments missing")
    specialist = row["specialist"]
    rate = row["throughput_probe"]["completion_tokens_per_second"]
    if not isinstance(rate, (int, float)) or rate <= 0:
        raise ValueError(f"{model['alias']}: invalid 256-token rate")
    anti = row["anti_contamination"]
    public = {
        "model": model["alias"], "family": model["family"],
        "revision": model["revision"], "engine": status["engine"],
        "physical_gpu_index": status["gpu_index"], "protocol": row["protocol"],
        "finished_at": status["finished_at"], "private_report_sha256": _sha(report_path),
        "eight_task_mean": row["eight_task_mean"],
        "specialist_composite": specialist["value"],
        "specialist_ci95": specialist["ci95"], "specialist_items": specialist["items"],
        "specialist_lanes": specialist["lanes"],
        "output_tokens_per_second_256": round(rate, 3),
        "throughput_scope": ("request wall time including prefill" if status["engine"] == "transformers"
                             else row["throughput_probe"].get("timing_source")),
        "gpu_board_wh_per_answer": row["gpu_board_wh_per_answer"],
        "gpu_board_wh_total": row["gpu_board_wh"], "requests": row["requests"],
        "canary_recall_rate": anti["canary_recall"]["recall_rate"],
        "reordered_option_original_accuracy": anti["paraphrase_invariance"]["original_accuracy"],
        "reordered_option_accuracy": anti["paraphrase_invariance"]["paraphrase_accuracy"],
        "fresh_arithmetic_holdout_accuracy": anti["post_cutoff_holdout"]["accuracy"],
        "promotion_eligible": False,
    }
    if status.get("runtime_compat"):
        compat = status["runtime_compat"]
        public["runtime_compat"] = {key: compat[key] for key in ("transformers", "accelerate", "mode")
                                    if key in compat}
    return public, commitments


def build(root: Path, roster: dict) -> dict:
    rows, frozen = [], None
    for model in roster["models"]:
        folder_name = model.get("private_run_folder", f"{model['alias']}-private")
        if Path(folder_name).name != folder_name:
            raise ValueError("private run folder must be a basename inside runs root")
        folder = root / folder_name
        if not folder.exists():
            continue
        status = json.loads((folder / "status.json").read_text())
        if status.get("status") != "complete_private":
            continue
        public, commitments = _read_profile(folder, model)
        if frozen is None:
            frozen = commitments
        elif commitments != frozen:
            raise ValueError("private models used different specialist or holdout packs")
        rows.append(public)
    if len({row["protocol"] for row in rows}) > 1:
        raise ValueError("private models used different protocols")
    return {"schema": "preschooler-public-profile-summary/v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "roster_count": len(roster["models"]), "complete_profiles": len(rows),
            "pack_commitments": frozen, "rows": rows,
            "interpretation": "Untrained checkpoint comparisons only; no model or mixture promotion. GPU-board Wh excludes CPU, memory, storage and PSU. Raw private items stay outside Git."}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--roster", type=Path, default=ROSTER)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = build(args.runs_root, json.loads(args.roster.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"complete_profiles": report["complete_profiles"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
