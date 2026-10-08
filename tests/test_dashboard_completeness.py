"""The well-known table lists every measured row, and a run can skip the shared rebuild."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/benchmarks"))
import build_dual_v100_html as dashboard  # noqa: E402
import well_known_suite as wks  # noqa: E402


def _report_rows() -> list[dict]:
    return json.loads((ROOT / "reports/well_known_suite_20260917.json").read_text())["results"]


def test_every_report_row_reaches_the_well_known_table():
    rows = dashboard.well_known_rows()
    # A model outside WELL_KNOWN_ORDER (cloud rows, later hardware profiles) is shown under its id.
    shown = {row["model"] for row in rows}
    unlisted = [r["model"] for r in _report_rows()
                if r["model"] not in dashboard.WELL_KNOWN_LABELS and not r["model"].startswith("mixture-optimized-")]
    assert unlisted, "the committed report is expected to hold rows outside WELL_KNOWN_ORDER"
    assert not [name for name in unlisted if name not in shown]


def test_superseded_and_retired_rows_are_labelled_not_promised_a_retry():
    by_name = {r["model"]: r for r in _report_rows()}
    status = {row["model"]: row["status"] for row in dashboard.well_known_rows()}
    for name, row in by_name.items():
        if row.get("superseded_by"):
            assert row["superseded_by"] in by_name
            assert status[name].startswith("achterhaald")
        if row.get("retired_reason"):
            assert status[name].startswith("niet vervolgd")


def test_skip_flag_writes_the_row_without_rebuilding(tmp_path, monkeypatch):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"results": []}))
    monkeypatch.setenv("VIRTUALV_SKIP_DASHBOARD", "1")

    def no_rebuild(*args, **kwargs):
        raise AssertionError("dashboard rebuild must be skipped")

    monkeypatch.setattr(wks.subprocess, "run", no_rebuild)
    wks.publish_partial(report, "some-model", "bezig · GSM8K af", gsm8k={"exact_match,strict-match": 1.0})
    row = json.loads(report.read_text())["results"][0]
    assert row["model"] == "some-model" and row["status"] == "running" and "gsm8k" in row
