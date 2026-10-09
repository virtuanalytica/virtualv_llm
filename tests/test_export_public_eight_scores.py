"""The public eight-task export must never include raw model answers."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/benchmarks"))
from export_public_eight_scores import TASK_IDS, export_one  # noqa: E402


def test_export_contains_only_scores_and_source_hash(tmp_path):
    secret = "PRIVATE_RAW_RESPONSE_DO_NOT_PUBLISH"
    path = tmp_path / "eight-tasks.jsonl"
    rows = [{"task": task, "score": 0.5, "detail": {"answer": secret},
             "request": {"prompt": secret}} for task in TASK_IDS]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    exported = export_one("model", path)
    encoded = json.dumps(exported)
    assert secret not in encoded
    assert len(exported["benchmarks"]) == len(TASK_IDS) + 1
    assert exported["benchmarks"]["_mean_score"] == 0.5
    assert len(exported["source_sha256"]) == 64


def test_export_rejects_duplicate_or_missing_tasks(tmp_path):
    path = tmp_path / "eight-tasks.jsonl"
    rows = [{"task": task, "score": 1} for task in TASK_IDS[:-1]]
    rows.append({"task": TASK_IDS[0], "score": 1})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        export_one("model", path)
