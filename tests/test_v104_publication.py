"""The v1.0.4 dashboard cannot promote packs exposed in PR #44."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/benchmarks"))
import build_dual_v100_html as dashboard  # noqa: E402


def test_exposed_private_pack_is_historical_in_dashboard():
    section = dashboard.private_aggregate_section()
    assert section.count("historisch · pack blootgesteld") == 4
    assert "historisch · kwaliteitsfilter gefaald" in section
    assert "test compleet · promotie open" not in section


def test_manifest_tracks_redacted_aggregate_and_vision_summary():
    manifest = json.loads((ROOT / "reports/night_runs_manifest_20261009.json").read_text())
    for filename in ("private_benchmark_aggregates_20261009.json",
                     "vision_private_summary_20261009.json"):
        data = (ROOT / "reports" / filename).read_bytes()
        assert manifest["published_files"][filename] == {
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def test_vision_repeat_is_shown_without_invalid_split_score():
    section = dashboard.vision_replication_section()
    assert "herhaling · GPU 2" in section
    assert "12/12" in section
    assert "twee-GPU-split gaf onleesbare uitvoer" in section
