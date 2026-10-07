"""Uncertainty maths behind the dashboard's error bars."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/benchmarks"))
import score_confidence as sc  # noqa: E402


def test_wilson_matches_reference_values():
    low, high = sc.wilson(3, 3)
    assert round(low, 4) == 0.4385 and high == 1.0
    low, high = sc.wilson(45, 50)
    assert round(low, 3) == 0.786 and round(high, 3) == 0.957


def test_perfect_small_sample_keeps_uncertainty():
    assert sc.variance(1.0, 3) > 0
    assert sc.min_items_for_lower_bound(0.95) == 73
    assert sc.wilson(73, 73)[0] >= 0.95 > sc.wilson(72, 72)[0]


def test_curated_report_yields_ranked_rows_with_intervals():
    payload = json.loads(sc.WELL_KNOWN.read_text())
    profiles = sc.well_known_confidence(payload, sc.current_protocol())
    entries = profiles["sandbox"]
    assert entries[0]["tied_with_leader"] and entries[0]["rank"] == 1
    assert [e["composite"] for e in entries] == sorted((e["composite"] for e in entries), reverse=True)
    for entry in entries:
        low, high = entry["ci95"]
        assert 0 <= low <= entry["composite"] <= high <= 1
        assert entry["standard_error"] > 0
    assert not entries[-1]["tied_with_leader"]
