"""Every publication date on a result row is backed by a source and matches the sourced date file."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS = json.loads((ROOT / "reports/well_known_suite_20260917.json").read_text())["results"]
DATES = json.loads((ROOT / "config/model_release_dates_v1.json").read_text())


def test_every_dated_row_names_its_source():
    dated = [r for r in ROWS if r.get("model_release_date")]
    assert dated
    for row in dated:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["model_release_date"]), row["model"]
        assert row.get("model_release_source", "").startswith("https://"), row["model"]


def test_rows_carry_the_date_recorded_in_the_sourced_file():
    by_name = {r["model"]: r for r in ROWS}
    for name, entry in DATES["rows"].items():
        assert by_name[name]["model_release_date"] == entry["model_release_date"]
        assert by_name[name]["model_release_source"] == entry["model_release_source"]
    # Unmapped models stay undated rather than receiving a guessed date.
    for name in DATES["unmapped"]:
        if name in by_name:
            assert not by_name[name].get("model_release_date"), name
