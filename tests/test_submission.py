"""The contribution protocol must fail closed; fixtures come from curated evidence."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/benchmarks"))
sys.path.insert(0, str(ROOT / "scripts/reporting"))
import score_confidence  # noqa: E402
import submission  # noqa: E402

REPORT = ROOT / "reports/well_known_suite_20260917.json"
PATH = "submissions/octocat/qwen38-flash-next-ap-iq2s.json"


def curated_document() -> dict:
    """A submission derived from a real curated row, so counts are measured ones."""
    rows = json.loads(REPORT.read_text())["results"]
    row = next(r for r in rows if r.get("weight_sha256") and r.get("source_revision")
               and "error" not in r and score_confidence.components(r))
    scores = {name: {"correct": round(acc * n), "n": n}
              for name, (acc, n) in score_confidence.components(row).items()}
    return {
        "schema": submission.CONFIG["schema"], "submission_id": "octocat/qwen38-flash-next-ap-iq2s",
        "status": "community-unverified", "submitted_at": "2026-10-07", "protocol": row["eval_protocol"],
        "submitter": {"github": "octocat"},
        "model": {"name": row["model"], "source_repo": row["source_repo"],
                  "source_revision": row["source_revision"], "quantization": str(row.get("quantization")),
                  "weight_sha256": row["weight_sha256"], "license": "see source repository"},
        "runtime": {"engine": "llama.cpp", "engine_version": "b1", "context_tokens": 8192,
                    "suite_commit": "c228fe5" + "0" * 33},
        "hardware": {"gpus": ["Tesla V100-SXM2-32GB", "Tesla V100-SXM2-32GB"], "topology": "NVLink layer split",
                     "cuda_device_order": "PCI_BUS_ID"},
        "scores": scores,
        "evidence": {"url": "https://github.com/virtuanalytica/virtualv_llm/releases",
                     "sha256": hashlib.sha256(REPORT.read_bytes()).hexdigest(), "bytes": REPORT.stat().st_size},
        "attestation": {"no_manual_score_edits": True, "protocol_unmodified": True,
                        "weights_license_respected": True},
    }


def check(doc: dict, path: str = PATH, external: bool = True) -> list[str]:
    return submission.validate_document(json.dumps(doc), path, external=external)


def test_curated_row_is_a_valid_submission():
    doc = curated_document()
    assert check(doc) == []
    assert 0 < submission.composite(doc) <= 1


@pytest.mark.parametrize("mutate, fragment", [
    (lambda d: d["scores"]["gsm8k"].update(n=49), "scores.gsm8k.n must be 50"),
    (lambda d: d["scores"]["mmlu"].update(correct=161), "between 0 and n"),
    (lambda d: d["scores"]["bbh"].update(correct=True), "wrong type"),
    (lambda d: d.update(protocol="v3-old"), "is not accepted"),
    (lambda d: d.update(status="verified-reproduced"), "status must be one of"),
    (lambda d: d.update(review={"reviewer": "x", "reviewed_at": "2026-10-07", "note": "ok"}), "only a maintainer"),
    (lambda d: d.update(composite=0.99), "unknown field"),
    (lambda d: d["model"].update(source_revision="main"), "40-character commit hash"),
    (lambda d: d["model"].update(weight_sha256="abc"), "weight_sha256"),
    (lambda d: d["evidence"].update(url="http://example.org/x"), "https://"),
    (lambda d: d["attestation"].update(protocol_unmodified=False), "must be true"),
    (lambda d: d["submitter"].update(github="someone-else"), "must match the directory"),
    (lambda d: d["hardware"].update(topology="/home/alice/rig"), "private path"),
    (lambda d: d.pop("evidence"), "evidence: missing"),
    (lambda d: d.update(submitted_at="2999-01-01"), "future"),
])
def test_violations_are_rejected(mutate, fragment):
    doc = copy.deepcopy(curated_document())
    mutate(doc)
    errors = check(doc)
    assert any(fragment in e for e in errors), errors


def test_path_and_size_are_enforced():
    doc = curated_document()
    assert check(doc, "submissions/octocat/../../reports/x.json")
    assert check(doc, "reports/well_known_suite_20260917.json")
    doc["model"]["name"] = "x" * (submission.CONFIG["max_submission_bytes"] + 1)
    assert "larger than" in check(doc)[0]


def test_maintainer_can_mark_reproduced_only_with_a_review():
    doc = curated_document()
    doc["status"] = "verified-reproduced"
    assert any("requires a review block" in e for e in check(doc, external=False))
    doc["review"] = {"reviewer": "maintainer", "reviewed_at": "2026-10-07", "note": "re-run within CI"}
    assert check(doc, external=False) == []


def added(*names: str) -> list[dict]:
    return [{"filename": n, "status": "added"} for n in names]


def test_guard_accepts_a_clean_external_submission():
    assert submission.guard_rules(added(PATH), "Octocat", "NONE") == []


@pytest.mark.parametrize("files, fragment", [
    (added("submissions/someone/x.json"), "only add files under submissions/octocat/"),
    (added(PATH, "scripts/benchmarks/eval_suite.py"), "must not be combined"),
    ([{"filename": PATH, "status": "modified"}], "append-only"),
    ([{"filename": PATH, "status": "removed"}], "append-only"),
    ([{"filename": PATH, "status": "renamed", "previous_filename": "reports/well_known_suite_20260917.json"}],
     "maintainer-only"),
    (added("reports/well_known_suite_20260917.json"), "maintainer-only"),
    (added(".github/workflows/ci.yml"), "maintainer-only"),
    (added("config/submission_protocol.json"), "maintainer-only"),
    (added("scripts/reporting/submission.py"), "maintainer-only"),
    (added("submissions/README.md"), "maintainer-only"),
    (added("submissions/octocat/run.py"), "path must be"),
    (added(*[f"submissions/octocat/m{i}.json" for i in range(6)]), "at most 5"),
])
def test_guard_rejects_protocol_violations(files, fragment):
    errors = submission.guard_rules(files, "octocat", "CONTRIBUTOR")
    assert any(fragment in e for e in errors), errors


def test_guard_lets_external_code_changes_outside_protected_paths_through():
    assert submission.guard_rules(added("scripts/benchmarks/new_probe.py", "tests/test_new_probe.py"),
                                  "octocat", "FIRST_TIME_CONTRIBUTOR") == []


def test_guard_does_not_restrict_maintainers():
    assert submission.guard_rules(added("reports/x.json", PATH), "febuz", "MEMBER") == []


def test_committed_submissions_are_valid():
    submission.load_valid()


def test_zcode_backend_runs_read_only_without_tools(monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts/benchmarks"))
    import external_cli_agent_proxy as proxy

    monkeypatch.setenv("VIRTUALV_ZCODE_CLI", "node /opt/zcode/zcode.cjs")
    command = proxy.zcode_command("2+2?", "/tmp/empty")
    assert command[:2] == ["node", "/opt/zcode/zcode.cjs"]
    assert command[command.index("--mode") + 1] == "plan"
    denied = command[command.index("--disallowed-tools") + 1].split()
    assert {"Bash", "Write", "Edit", "WebFetch", "Agent"} <= set(denied)
    monkeypatch.delenv("VIRTUALV_ZCODE_CLI")
    with pytest.raises(RuntimeError):
        proxy.zcode_command("2+2?", "/tmp/empty")
