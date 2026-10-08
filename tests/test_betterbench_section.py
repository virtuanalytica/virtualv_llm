"""The suite-quality section reflects the committed self-assessment and the verbatim criteria."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/benchmarks"))
import build_dual_v100_html as dashboard  # noqa: E402

CRITERIA = json.loads((ROOT / "config/betterbench_criteria_v1.json").read_text())["criteria"]
ASSESSMENT = json.loads((ROOT / "reports/betterbench_self_assessment_20261008.json").read_text())


def test_assessment_scores_every_criterion_once_on_the_paper_scale():
    for stage, items in CRITERIA.items():
        scored = ASSESSMENT["stages"][stage]["criteria"]
        assert [row["criterion"] for row in scored] == items
        assert all(row["score"] in (0, 5, 10, 15) and row["evidence"] for row in scored)
        mean = sum(row["score"] for row in scored) / len(scored)
        assert abs(mean - ASSESSMENT["stages"][stage]["mean"]) < 0.01


def test_section_shows_each_stage_and_lists_the_zero_scores():
    section = dashboard.betterbench_section()
    for label in dashboard.BETTERBENCH_STAGES.values():
        assert f"<strong>{label}</strong>" in section
    zero = [row["criterion"] for stage in ASSESSMENT["stages"].values()
            for row in stage["criteria"] if row["score"] == 0]
    assert zero == ASSESSMENT["zero_scores"]
    assert all(item in section for item in zero)
    assert "niet getoetst door de auteurs" in section
