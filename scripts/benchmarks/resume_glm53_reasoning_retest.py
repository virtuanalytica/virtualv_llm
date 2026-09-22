#!/usr/bin/env python3
"""Resume the current forced GLM re-test only when its attempt is incomplete.

The systemd unit acquires the shared GPU lock before invoking this script. This
allows an already-running interactive attempt to finish first without duplicate
GPU work, while still recovering automatically if that process disappears.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "reports/glm53_reap50_cascade_20260922.json"
RUNNER = ROOT / "scripts/benchmarks/run_glm53_reap50_cascade.py"
REQUIRED_MODELS = {
    "glm53-reap50-iq3m-v100",
    "glm53-reap50-iq3m-allfour",
}


def current_attempt_complete() -> bool:
    if not STATE.exists():
        return False
    events = json.loads(STATE.read_text()).get("events", [])
    starts = [event for event in events
              if event.get("quant") == "iq3m" and event.get("event") == "download_started"]
    if not starts:
        return False
    started_at = starts[-1].get("at", "")
    completed = {
        event.get("model") for event in events
        if event.get("quant") == "iq3m"
        and event.get("event") == "benchmark_complete"
        and event.get("at", "") >= started_at
    }
    return REQUIRED_MODELS <= completed


def main() -> int:
    if current_attempt_complete():
        print("current forced GLM reasoning re-test already complete; no rerun needed")
        return 0
    command = ["python3", str(RUNNER), "--only", "iq3m", "--force"]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
