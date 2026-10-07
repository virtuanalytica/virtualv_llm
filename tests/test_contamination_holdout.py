"""The holdout answer key is checked by arithmetic, and old rows are re-scored from stored predictions."""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/benchmarks"))
import contamination_audit as ca  # noqa: E402

PACK = ROOT / "config/contamination_audit/post_cutoff_holdout.csv"
# Independent solutions of every arithmetic item, from the numbers in the task text.
EXPECTED = {
    "pch-001": 17 + 6 * (4 - 2),
    "pch-002": 22 * 3 - (15 + 9),
    "pch-003": 5 + 10 + 20 + 40,
    "pch-005": (14 + (2 * 14 - 5)) // 2,
    "pch-006": (316 - 4) // 12,
    "pch-008": 6 * 2 ** (20 // 5),
    "pch-011": sum(7 + 3 * day for day in range(5)),
    "pch-012": 540 * 4 // (2 + 3 + 4),
}


def rows():
    with PACK.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_every_enabled_arithmetic_key_matches_the_computed_answer():
    enabled = {r["id"]: r for r in rows() if r["enabled"] == "true"}
    assert len(enabled) == 11 and "pch-009" not in enabled
    for item, value in EXPECTED.items():
        row = enabled[item]
        assert row[f"option_{row['answer'].lower()}"] == str(value), item


def test_unsolvable_item_has_no_correct_option():
    row = next(r for r in rows() if r["id"] == "pch-009")
    assert (264 - 18) / 5 == 49.2
    assert "49.2" not in (row[f"option_{c}"] for c in "abcd") and row["enabled"] == "false"


def test_rescoring_uses_stored_predictions_and_is_idempotent(tmp_path):
    report = tmp_path / "audit.json"
    samples = [{"id": f"pch-{i:03d}", "prediction": "A", "target": "X", "correct": False} for i in range(1, 13)]
    report.write_text(json.dumps({"results": [{"model": "m", "results": {"post_cutoff_holdout": {
        "status": "complete", "accuracy": 0.0, "n_samples": 12, "pack_sha256": "old", "samples": samples}}}]}))
    assert ca.rescore_post_cutoff_holdout(report) == 1
    holdout = json.loads(report.read_text())["results"][0]["results"]["post_cutoff_holdout"]
    key_a = sum(r["answer"] == "A" and r["enabled"] == "true" for r in rows())
    assert holdout["n_samples"] == 11 and holdout["accuracy"] == round(key_a / 11, 4)
    assert all(s["id"] != "pch-009" for s in holdout["samples"])
    assert ca.rescore_post_cutoff_holdout(report) == 0
