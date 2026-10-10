"""No direct or scheduled entry point may start the archived public battery."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_suite_cli_refuses_before_model_access():
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/benchmarks/well_known_suite.py"),
                           "obsolete-model", "--external-url", "http://127.0.0.1:1"],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 2
    assert "retired historical archive" in proc.stderr


def test_cron_runner_refuses_before_selecting_a_model():
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/benchmarks/run_next_benchmark.py")],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 2
    assert "retired" in proc.stdout
