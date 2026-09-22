#!/usr/bin/env python3
"""Hard gate for the expensive three-access-profile benchmark matrix."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config" / "benchmark_execution_policy.json"


def matrix_gate() -> dict[str, Any]:
    policy = json.loads(POLICY.read_text())
    unlock = policy["matrix_unlock"]
    report = ROOT / unlock["required_artifact"]
    data = json.loads(report.read_text()) if report.exists() else {"results": []}
    candidates = [row for row in data.get("results", []) if row.get("model") in unlock["model_aliases"]]
    accepted = []
    for row in candidates:
        complete = (row.get("eval_protocol") == unlock["required_protocol"] and not row.get("error") and
                    all(row.get(key) is not None for key in unlock["required_fields"]))
        if complete:
            accepted.append(row.get("model"))
    if not accepted:
        reason = "GLM-5.3 current-protocol sandbox suite incomplete"
    elif not policy.get("matrix_enabled"):
        reason = "GLM-5.3 gate passed; matrix awaits explicit policy enablement"
    else:
        reason = "GLM-5.3 gate passed"
    return {
        "phase": policy["current_phase"],
        "matrix_enabled": bool(policy.get("matrix_enabled")) and bool(accepted),
        "glm53_sandbox_winner": accepted[0] if accepted else None,
        "reason": reason,
        "policy": str(POLICY.relative_to(ROOT)),
    }


if __name__ == "__main__":
    result = matrix_gate()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["matrix_enabled"] else 1)
