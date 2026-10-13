import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "benchmarks"))
import mom_as_whole as mw  # noqa: E402


def write(dirpath, name, rows, errors=0):
    items = [{"idx": i, "half": h, "correct": c, "reply_tail": f"Answer: {a}"} for i, (h, c, a) in enumerate(rows)]
    (dirpath / name).write_text(json.dumps({"errors": errors, "mean_tokens": 10, "items": items}))


def test_selector_takes_local_answer_on_agreement_and_haiku_otherwise(tmp_path):
    # four test items; KAT and Gemma4 agree on items 0-2, disagree on item 3
    write(tmp_path, "gsm8k_haiku.json", [("test", 1, 5), ("test", 0, 9), ("test", 1, 7), ("test", 1, 3)])
    write(tmp_path, "gsm8k_kat_off.json", [("test", 1, 5), ("test", 1, 6), ("test", 0, 1), ("test", 0, 2)])
    write(tmp_path, "gsm8k_gemma4_off.json", [("test", 1, 5), ("test", 1, 6), ("test", 0, 1), ("test", 1, 3)])
    members, skipped = mw.load_members(tmp_path, "gsm8k")
    s = mw.summarise("gsm8k", members, skipped)
    pre = next(x for x in s["selectors"] if x["preregistered"])
    # items 0,1,2 local (1,1,0), item 3 Haiku (1) -> 3/4; Haiku asked once
    assert pre["score"] == 0.75 and pre["haiku_share"] == 0.25
    assert pre["vs_haiku"] == {"better": 1, "worse": 1, "p_exact": 1.0}
    assert s["oracle_all_members"] == 1.0 and s["haiku_score"] == 0.75


def test_rerun_replaces_failed_file_and_failed_runs_are_listed(tmp_path):
    write(tmp_path, "tqa_mc1_haiku.json", [("test", 1, "A")])
    write(tmp_path, "tqa_mc1_kat_off.json", [("test", 0, "B")], errors=50)
    write(tmp_path, "tqa_mc1_kat_off_r2.json", [("test", 1, "A")])
    write(tmp_path, "tqa_mc1_q36_off.json", [("test", 0, "C")], errors=9)
    members, skipped = mw.load_members(tmp_path, "tqa_mc1")
    assert "kat_off" in members and members["kat_off"]["rows"][0][1] == 1
    assert skipped == ["q36_off: 9 errors"]


def test_dashboard_section_renders_members_oracle_and_selectors(tmp_path, monkeypatch):
    import build_dual_v100_html as b
    report = {"sets": {"gsm8k": {"n_test": 4, "haiku_score": 0.75, "oracle_all_members": 1.0, "skipped": [],
                                 "members": [{"name": "kat_off", "model": "KAT", "thinking": False, "score": 0.5}],
                                 "selectors": [{"members": ["kat_off", "gemma4_off"], "preregistered": True,
                                                "passed": True, "score": 0.75, "haiku_share": 0.25,
                                                "vs_haiku": {"better": 1, "worse": 1, "p_exact": 1.0}}]}}}
    monkeypatch.setattr(b, "load", lambda name: report if name == b.MOM_WHOLE_REPORT else {})
    html_out = b.mom_as_whole_section()
    assert "mom-als-geheel" in html_out and "100.0%" in html_out and "25.0%" in html_out
    assert "vooraf vastgelegd · geslaagd" in html_out and "<td><td" not in html_out
