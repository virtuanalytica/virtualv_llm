#!/usr/bin/env python3
"""Score a paired ClaudeClaw agent trial; never infer proof from model rows."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from software_data_suite import ROOT, compare_software

OUT = ROOT / "reports/software_agent_comparison.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path,
                        help="independently verified Toddler + Teacher + agent on ClaudeClaw report")
    parser.add_argument("--baseline", required=True, type=Path,
                        help="independently verified ordinary ClaudeClaw worker report")
    parser.add_argument("--commitment", required=True, type=Path,
                        help="pre-registered independent-evaluator pack commitment")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text())
    baseline = json.loads(args.baseline.read_text())
    commitment = json.loads(args.commitment.read_text())
    if (commitment.get("status") != "sealed-independent" or
            commitment.get("sha256") != candidate.get("pack_sha256") or
            commitment.get("sha256") != baseline.get("pack_sha256") or
            not commitment.get("sealed_at") or
            not all(r.get("started_at", "") > commitment["sealed_at"] for r in (candidate, baseline))):
        proof = {"better_proven": False, "reason": "independent_preregistered_pack_missing"}
    else:
        proof = compare_software(candidate, baseline)
    report = {"question": "Is Toddler + Teacher + agent on ClaudeClaw better on software tasks?",
              "answer": "ja, intern aangetoond" if proof["better_proven"] else "nee, nog niet",
              "candidate": candidate.get("run_id"), "baseline": baseline.get("run_id"),
              "protocol": candidate.get("protocol"), "pack_sha256": candidate.get("pack_sha256"),
              "at": datetime.now(timezone.utc).isoformat(), "proof": proof}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(args.out)
    print(json.dumps({"answer": report["answer"], "reason": proof["reason"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
