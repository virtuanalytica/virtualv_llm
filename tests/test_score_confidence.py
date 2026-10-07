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


def test_mcnemar_exact_reference_values():
    assert sc.mcnemar_exact(0, 0) == 1.0
    assert sc.mcnemar_exact(5, 5) == 1.0
    assert round(sc.mcnemar_exact(6, 0), 5) == 0.03125   # 2 * 0.5**6
    assert round(sc.mcnemar_exact(9, 1), 4) == 0.0215


def test_paired_humaneval_uses_only_shared_items_of_curated_rows():
    payload = json.loads(sc.WELL_KNOWN.read_text())
    paired = sc.humaneval_paired(payload, sc.current_protocol())["sandbox"]
    assert paired and all(0 <= e["p_value"] <= 1 for e in paired)
    for entry in paired:
        assert entry["leader_only_pass"] + entry["model_only_pass"] <= entry["shared_items"] <= 40
        assert entry["differs_from_leader"] == (entry["p_value"] < 0.05)


def lane(accuracy, n, status="complete"):
    return {"status": status, "accuracy": accuracy, "n_samples": n}


def test_specialist_composite_counts_only_complete_text_lanes():
    row = {"results": {name: lane(0.5, 4) for name in sc.SPECIALIST_LANES}}
    full = sc.specialist_composite(row)
    assert full["complete"] and full["lanes"] == 8 and full["value"] == 0.5 and full["items"] == 32
    row["results"]["vision"] = lane(1.0, 3)                      # never counted
    row["results"]["video"] = {"status": "unavailable", "n_samples": 0}
    partial = sc.specialist_composite(row)
    assert not partial["complete"] and partial["lanes"] == 7 and partial["value"] == 0.5
    assert sc.specialist_composite({"results": {}}) is None


def test_resistant_composite_keeps_memorisation_signals_out_of_the_number():
    row = {"results": {
        "post_cutoff_holdout": {"status": "complete", "accuracy": 0.5, "n_samples": 12},
        "paraphrase_invariance": {"status": "complete", "paraphrase_accuracy": 0.7, "accuracy_gap": 0.3,
                                  "n_samples": 15},
        "canary_recall": {"status": "complete", "recall_rate": 0.5, "n_samples": 10}}}
    result = sc.resistant_composite(row)
    assert result["value"] == 0.6 and result["items"] == 27
    assert result["flags"] == ["canary-recall", "form-sensitive"]
    row["results"]["canary_recall"]["recall_rate"] = 0.0
    assert sc.resistant_composite(row)["value"] == 0.6
    row["results"]["post_cutoff_holdout"]["status"] = "unavailable"
    assert sc.resistant_composite(row) is None


def test_resistant_ranking_on_curated_evidence_is_sorted_and_bounded():
    ranking = sc.resistant_ranking(json.loads(sc.WELL_KNOWN.read_text()), json.loads(sc.SPECIALIST.read_text()),
                                   json.loads(sc.AUDIT.read_text()), sc.current_protocol())
    scored = [e["resistant"]["value"] for e in ranking if e["resistant"]]
    assert scored and scored == sorted(scored, reverse=True)
    for entry in ranking:
        for cell in (entry["resistant"], entry["specialist"]):
            if cell:
                assert 0 <= cell["ci95"][0] <= cell["value"] <= cell["ci95"][1] <= 1
