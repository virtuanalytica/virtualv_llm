"""Controls for maker/data scoring and the paired agent proof gate."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/benchmarks"))
import software_data_suite as suite  # noqa: E402
import software_agent_comparison as agent_comparison  # noqa: E402


def test_pack_is_external_private_and_hash_locked(tmp_path, monkeypatch):
    pack = {"protocol": suite.PROTOCOL, "items": [
        {"id": f"{track}-{role}-{n}", "track": track, "role": role,
         "prompt": "private", "rubric": {"type": "exact", "expected": {}}}
        for track, roles in suite.ROLES.items() for role in roles for n in (1, 2)]}
    path = tmp_path / "sealed.json"
    path.write_text(json.dumps(pack))
    path.chmod(0o600)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    commitment = tmp_path / "commit.json"
    commitment.write_text(json.dumps({"sha256": digest, "counts": {
        track: {role: 2 for role in roles} for track, roles in suite.ROLES.items()}}))
    monkeypatch.setattr(suite, "COMMITMENT", commitment)
    assert suite.load_pack(path)[1] == digest
    path.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        suite.load_pack(path)
    path.chmod(0o600)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        suite.load_pack(path)


def test_sql_requires_read_only_query_and_passes_hidden_fixtures():
    item = {"rubric": {"type": "sql", "fixtures": [
        {"setup": "CREATE TABLE t(x INTEGER); INSERT INTO t VALUES (1),(2);",
         "expected": [[3]]}]}}
    assert suite.sql_score("SELECT SUM(x) FROM t", item)[0]
    assert not suite.sql_score("DELETE FROM t", item)[0]
    assert not suite.sql_score("SELECT 0", item)[0]


@pytest.mark.skipif(not shutil.which("docker") or subprocess.run(
    ["docker", "image", "inspect", suite.IMAGE], capture_output=True).returncode != 0,
    reason="isolated Python image unavailable")
def test_code_runs_in_networkless_container_with_hidden_cases():
    item = {"rubric": {"type": "code", "cases": [
        {"input": {"x": 2}, "expected": 4},
        {"input": {"x": 7}, "expected": 14}]}}
    assert suite.code_score("def solve(payload): return payload['x'] * 2", item)[0]
    assert not suite.code_score("def solve(payload): return 4", item)[0]


def _agent(stack, passed_per_role):
    rows = []
    for role in suite.ROLES["software"]:
        for n in range(73):
            rows.append({"id": f"{role}-{n}", "track": "software", "role": role,
                         "task_id": f"T-{role}-{n}", "passed": n < passed_per_role,
                         "verification_sha256": "a" * 64, "post_merge_errors": 0})
    return {"subject_type": "agent", "stack": stack, "independent_verification": True,
            "evaluator_isolation": "separate_identity",
            "protocol": suite.PROTOCOL, "pack_sha256": "b" * 64,
            "status": "complete", "results": rows}


def test_only_paired_verified_agent_stacks_can_prove_better():
    candidate = _agent("toddler_teacher_agent_claudeclaw", 60)
    baseline = _agent("claudeclaw_standard", 40)
    assert suite.compare_software(candidate, baseline)["better_proven"] is True
    model = dict(candidate, subject_type="model")
    assert suite.compare_software(model, baseline)["reason"] == "requires_paired_agent_stacks"
    candidate["results"] = candidate["results"][:-1]
    assert suite.compare_software(candidate, baseline)["reason"] == "different_software_items"


def test_pilot_never_proves_better_and_post_merge_regression_blocks():
    candidate = _agent("toddler_teacher_agent_claudeclaw", 60)
    baseline = _agent("claudeclaw_standard", 40)
    candidate["results"][0]["post_merge_errors"] = 1
    assert suite.compare_software(candidate, baseline)["better_proven"] is False
    candidate["results"][0]["post_merge_errors"] = 0
    candidate["results"] = [r for r in candidate["results"] if int(r["id"].split("-")[-1]) < 2]
    baseline["results"] = [r for r in baseline["results"] if int(r["id"].split("-")[-1]) < 2]
    assert suite.compare_software(candidate, baseline)["reason"] == "fewer_than_73_items_per_role"


def test_agent_publication_rejects_pilot_commitment(tmp_path, monkeypatch):
    candidate = _agent("toddler_teacher_agent_claudeclaw", 60)
    baseline = _agent("claudeclaw_standard", 40)
    for report in (candidate, baseline):
        report["started_at"] = "2026-10-09T03:00:00Z"
    paths = [tmp_path / name for name in ("candidate.json", "baseline.json", "commitment.json", "out.json")]
    paths[0].write_text(json.dumps(candidate))
    paths[1].write_text(json.dumps(baseline))
    paths[2].write_text(json.dumps({"status": "sealed-pilot", "sha256": "b" * 64,
                                   "sealed_at": "2026-10-09T02:00:00Z"}))
    monkeypatch.setattr(sys, "argv", ["comparison", "--candidate", str(paths[0]),
                                  "--baseline", str(paths[1]), "--commitment", str(paths[2]),
                                  "--out", str(paths[3])])
    assert agent_comparison.main() == 0
    result = json.loads(paths[3].read_text())
    assert result["answer"] == "nee, nog niet"
    assert result["proof"]["reason"] == "independent_preregistered_pack_missing"


def test_haiku_requires_cloud_permission_and_two_local_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "load_pack", lambda *_args: ({"items": []}, "c" * 64))
    pack, commitment, out = (tmp_path / name for name in ("pack.json", "commit.json", "results.json"))
    commitment.write_text(json.dumps({"status": "sealed-pilot", "cloud_allowed": False}))
    monkeypatch.setattr(sys, "argv", ["suite", "--pack", str(pack), "--commitment",
                                  str(commitment), "--provider", "haiku55",
                                  "--model-id", "haiku55", "--out", str(out)])
    with pytest.raises(ValueError, match="not approved for cloud"):
        suite.main()
    commitment.write_text(json.dumps({"status": "sealed-cloud-safe-pilot", "cloud_allowed": True}))
    with pytest.raises(ValueError, match="two distinct local model aliases"):
        suite.main()
    out.write_text(json.dumps({"results": [
        {"model": model, "pack_sha256": "c" * 64, "status": "complete",
         "runtime": {"provider": "local", "alias": "same-alias"}}
        for model in ("local-a", "local-b")]}))
    with pytest.raises(ValueError, match="two distinct local model aliases"):
        suite.main()
    data = json.loads(out.read_text())
    data["results"][1]["runtime"]["alias"] = "other-alias"
    out.write_text(json.dumps(data))
    monkeypatch.setattr(suite, "run_suite", lambda *_args: {"tracks": {}})
    monkeypatch.setattr(suite, "upsert_result", lambda *_args, **_kwargs: None)
    assert suite.main() == 0
    data["results"].append({"pack_sha256": "c" * 64, "status": "complete",
                            "runtime": {"provider": "haiku55"}})
    out.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="closed by its Haiku"):
        suite.main()
