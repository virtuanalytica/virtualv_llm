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


def test_error_text_decides_the_status_not_a_blanket_retry_promise():
    by_name = {r["model"]: r for r in _report_rows()}
    status = {row["model"]: row["status"] for row in dashboard.well_known_rows()}
    for name, row in by_name.items():
        error = str(row.get("error", "")).lower()
        shown = status.get(dashboard.WELL_KNOWN_LABELS.get(name, name), "")
        if error.startswith("excluded on speed"):
            assert shown.startswith("uitgesloten op snelheid"), name
        if error.startswith("unsupported") and not row.get("superseded_by") and not row.get("retired_reason"):
            assert shown.startswith("niet ondersteund"), name


def test_builder_honours_the_skip_flag(monkeypatch, capsys):
    monkeypatch.setenv("VIRTUALV_SKIP_DASHBOARD", "1")
    monkeypatch.setattr(dashboard, "well_known_rows", lambda: (_ for _ in ()).throw(AssertionError("must not build")))
    assert dashboard.main() == 0
    assert "skipped" in capsys.readouterr().out


def test_placeholders_point_at_measured_rows_and_only_planned_work_stays_queued():
    measured = {r["model"] for r in _report_rows()}
    order = set(dashboard.WELL_KNOWN_ORDER)
    assert set(dashboard.PLACEHOLDER_SUPERSEDED_BY) | set(dashboard.PLACEHOLDER_RETIRED) <= order
    assert set(dashboard.PLACEHOLDER_SUPERSEDED_BY.values()) <= measured
    # A placeholder that has since been measured must not keep a stale disposition.
    assert not (set(dashboard.PLACEHOLDER_SUPERSEDED_BY) | set(dashboard.PLACEHOLDER_RETIRED)) & measured
    queued = [row["model"] for row in dashboard.well_known_rows() if row["status"] == "in benchmarkwachtrij"]
    assert queued == [dashboard.WELL_KNOWN_LABELS["qwen36-27b-iq3"]]
