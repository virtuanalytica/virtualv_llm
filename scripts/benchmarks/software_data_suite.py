#!/usr/bin/env python3
"""Private software-maker and data-role benchmark, separate from the ten-lane suite.

The pack lives outside the repository. Only prompts reach the model. Executable
code is run in a networkless, read-only Docker container with no host mounts
except a fresh directory containing the candidate and a fixed runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
import subprocess
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from result_store import upsert_result
from score_confidence import wilson

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/software_data_specialists.json"
COMMITMENT = ROOT / "reports/software_data_pack_commitment.json"
PROTOCOL = "software-data-private-v1-20261009"
ROLES = {
    "software": ("coder", "reviewer", "architect", "debugger"),
    "data": ("data_engineer", "data_analyst", "data_architect", "data_steward"),
}
IMAGE = "python@sha256:1b668429b3511ab407d8e00648891631b0b1a4d7e15e3ca70f38ab5b91ad4ab4"
MAX_RESPONSE = 24_000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pack(path: Path, commitment_path: Path | None = None) -> tuple[dict, str]:
    resolved = path.resolve(strict=True)
    if resolved.is_relative_to(ROOT.resolve()) or any((parent / ".git").exists()
                                                   for parent in resolved.parents):
        raise ValueError("sealed pack must live outside every Git checkout")
    if resolved.stat().st_mode & 0o077 or resolved.parent.stat().st_mode & 0o077:
        raise ValueError("sealed pack must have mode 0600 under a mode 0700 directory")
    commit = json.loads((commitment_path or COMMITMENT).read_text())
    digest = _sha(resolved)
    if digest != commit.get("sha256"):
        raise ValueError("sealed pack changed after commitment")
    pack = json.loads(resolved.read_text())
    if pack.get("protocol") != PROTOCOL:
        raise ValueError("wrong pack protocol")
    ids = [item["id"] for item in pack["items"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate item IDs")
    counts = {track: {role: sum(i["track"] == track and i["role"] == role
                               for i in pack["items"]) for role in roles}
              for track, roles in ROLES.items()}
    if counts != commit.get("counts") or any(n < 2 for roles in counts.values() for n in roles.values()):
        raise ValueError("pack role counts differ from commitment or are too small")
    if any(i["role"] not in ROLES.get(i["track"], ()) for i in pack["items"]):
        raise ValueError("unknown track/role")
    return pack, digest


def extract_json(response: str) -> dict:
    response = response.strip()
    if response.startswith("```"):
        response = re.sub(r"^```(?:json)?\s*|\s*```$", "", response, flags=re.I)
    value = json.loads(response)
    if not isinstance(value, dict):
        raise ValueError("answer must be a JSON object")
    return value


def extract_code(response: str) -> str:
    response = response.strip()
    if response.startswith("```"):
        response = re.sub(r"^```(?:python)?\s*|\s*```$", "", response, flags=re.I)
    if len(response) > MAX_RESPONSE or "def solve(" not in response:
        raise ValueError("expected a Python solve(payload) function")
    return response


def code_score(response: str, item: dict) -> tuple[bool, str]:
    source = extract_code(response)
    cases = item["rubric"]["cases"]
    runner = ("import json,sys\nfrom candidate import solve\n"
              "value=json.load(sys.stdin)\nprint(json.dumps(solve(value),sort_keys=True))\n")
    with tempfile.TemporaryDirectory(prefix="virtualv-code-") as tmp:
        work = Path(tmp)
        (work / "candidate.py").write_text(source)
        (work / "runner.py").write_text(runner)
        work.chmod(0o755)
        for file in work.iterdir():
            file.chmod(0o644)
        for case in cases:
            command = ["docker", "run", "--rm", "-i", "--network", "none", "--read-only",
                       "--cap-drop", "ALL",
                       "--pids-limit", "32", "--cpus", "1", "--memory", "256m",
                       "--user", "65534:65534", "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
                       "--mount", f"type=bind,src={work},dst=/work,readonly",
                       "--workdir", "/work", IMAGE, "python", "-S", "runner.py"]
            try:
                run = subprocess.run(command, input=json.dumps(case["input"]), text=True,
                                     capture_output=True, timeout=30, check=False)
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                return False, type(exc).__name__
            if run.returncode != 0:
                return False, "runtime_error"
            try:
                actual = json.loads(run.stdout.strip())
            except ValueError:
                return False, "invalid_json_output"
            if actual != case["expected"]:
                return False, "hidden_case_failed"
    return True, "all_hidden_cases_passed"


def sql_score(response: str, item: dict) -> tuple[bool, str]:
    query = response.strip().rstrip(";")
    if query.startswith("```"):
        query = re.sub(r"^```(?:sql)?\s*|\s*```$", "", query, flags=re.I).strip().rstrip(";")
    if len(query) > 4000 or ";" in query or not re.match(r"^(select|with)\b", query, re.I):
        return False, "not_one_read_only_query"
    for fixture in item["rubric"]["fixtures"]:
        db = sqlite3.connect(":memory:")
        try:
            db.executescript(fixture["setup"])
            def authorizer(action, _arg1, _arg2, _db, _trigger):
                return (sqlite3.SQLITE_OK if action in
                        {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
                        else sqlite3.SQLITE_DENY)
            db.set_authorizer(authorizer)
            started = time.monotonic()
            db.set_progress_handler(lambda: int(time.monotonic() - started > 1), 1000)
            rows = [list(row) for row in db.execute(query).fetchmany(101)]
        except sqlite3.Error:
            return False, "sql_error_or_disallowed_operation"
        finally:
            db.close()
        if rows != fixture["expected"]:
            return False, "hidden_fixture_failed"
    return True, "all_hidden_fixtures_passed"


def structured_score(response: str, item: dict) -> tuple[bool, str]:
    value = extract_json(response)
    rubric = item["rubric"]
    if rubric["type"] == "exact":
        passed = value == rubric["expected"]
    elif rubric["type"] == "numbers":
        passed = set(value) == set(rubric["expected"]) and all(
            isinstance(value[k], (int, float)) and not isinstance(value[k], bool)
            and abs(value[k] - target) <= rubric.get("tolerance", 1e-6)
            for k, target in rubric["expected"].items())
    else:
        raise ValueError("unknown structured rubric")
    return passed, "matched_rubric" if passed else "rubric_mismatch"


def score(response: str, item: dict) -> tuple[bool, str]:
    try:
        kind = item["rubric"]["type"]
        if kind == "code":
            return code_score(response, item)
        if kind == "sql":
            return sql_score(response, item)
        return structured_score(response, item)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, f"invalid_response:{type(exc).__name__}"


def summarize(items: list[dict], results: list[dict]) -> dict:
    by_role: dict[str, list[dict]] = defaultdict(list)
    lookup = {item["id"]: item for item in items}
    for result in results:
        item = lookup[result["id"]]
        by_role[f"{item['track']}/{item['role']}"].append(result)
    tracks = {}
    for track, roles in ROLES.items():
        track_roles = {}
        for role in roles:
            rows = by_role[f"{track}/{role}"]
            n, passed = len(rows), sum(bool(row["passed"]) for row in rows)
            interval = wilson(passed, n) if n else (0.0, 1.0)
            track_roles[role] = {"passed": passed, "n": n,
                                 "accuracy": round(passed / n, 4) if n else None,
                                 "wilson95": [round(x, 4) for x in interval],
                                 "status": "complete" if n == sum(i["track"] == track and i["role"] == role
                                                                 for i in items) else "partial"}
        scores = [row["accuracy"] for row in track_roles.values()]
        tracks[track] = {"roles": track_roles,
                         "passed": sum(r["passed"] for r in track_roles.values()),
                         "n": sum(r["n"] for r in track_roles.values()),
                         "composite": round(sum(scores) / len(scores), 4)
                         if all(score is not None for score in scores) else None}
    return tracks


def compare_software(candidate: dict, baseline: dict) -> dict:
    """Pre-registered paired, role-balanced proof gate for software makers."""
    if (candidate.get("subject_type"), candidate.get("stack")) != (
            "agent", "toddler_teacher_agent_claudeclaw") or (
            baseline.get("subject_type"), baseline.get("stack")) != (
            "agent", "claudeclaw_standard"):
        return {"better_proven": False, "reason": "requires_paired_agent_stacks"}
    if not candidate.get("independent_verification") or not baseline.get("independent_verification"):
        return {"better_proven": False, "reason": "independent_verification_missing"}
    if (candidate.get("evaluator_isolation"), baseline.get("evaluator_isolation")) != (
            "separate_identity", "separate_identity"):
        return {"better_proven": False, "reason": "independent_evaluator_isolation_missing"}
    if (candidate.get("protocol"), candidate.get("pack_sha256")) != (
            baseline.get("protocol"), baseline.get("pack_sha256")):
        return {"better_proven": False, "reason": "protocol_or_pack_mismatch"}
    if candidate.get("status") != "complete" or baseline.get("status") != "complete":
        return {"better_proven": False, "reason": "incomplete_result"}
    a = {r["id"]: r for r in candidate["results"] if r["track"] == "software"}
    b = {r["id"]: r for r in baseline["results"] if r["track"] == "software"}
    if (len(a) != sum(r["track"] == "software" for r in candidate["results"]) or
            len(b) != sum(r["track"] == "software" for r in baseline["results"]) or
            a.keys() != b.keys()):
        return {"better_proven": False, "reason": "different_software_items"}
    if any(a[key].get("task_id") != b[key].get("task_id") or
           a[key].get("role") != b[key].get("role") or
           not re.fullmatch(r"[0-9a-f]{64}", a[key].get("verification_sha256", "")) or
           not re.fullmatch(r"[0-9a-f]{64}", b[key].get("verification_sha256", "")) or
           not isinstance(a[key].get("post_merge_errors"), int) or
           not isinstance(b[key].get("post_merge_errors"), int) or
           a[key]["post_merge_errors"] < 0 or b[key]["post_merge_errors"] < 0
           for key in a):
        return {"better_proven": False, "reason": "unpaired_or_unverified_task_evidence"}
    diffs = {role: [int(a[key]["passed"]) - int(b[key]["passed"])
                    for key in sorted(a) if a[key]["role"] == role]
             for role in ROLES["software"]}
    n_by_role = {role: len(values) for role, values in diffs.items()}
    role_delta = {role: sum(values) / len(values) if values else None
                  for role, values in diffs.items()}
    if any(n < 73 for n in n_by_role.values()):
        return {"better_proven": False, "reason": "fewer_than_73_items_per_role",
                "n_by_role": n_by_role, "role_delta": role_delta}
    delta = sum(role_delta.values()) / len(role_delta)
    rng = random.Random(20261009)
    draws = []
    for _ in range(4000):
        draws.append(sum(sum(rng.choice(diffs[role]) for _ in diffs[role]) / len(diffs[role])
                         for role in ROLES["software"]) / len(ROLES["software"]))
    draws.sort()
    interval = [round(draws[100], 4), round(draws[3899], 4)]
    escaped_candidate = sum(row["post_merge_errors"] for row in a.values())
    escaped_baseline = sum(row["post_merge_errors"] for row in b.values())
    proven = (delta >= 0.03 and interval[0] > 0 and
              all(value >= -0.05 for value in role_delta.values()) and
              escaped_candidate <= escaped_baseline)
    return {"better_proven": proven, "reason": "gate_passed" if proven else "quality_gate_failed",
            "delta": round(delta, 4), "paired_bootstrap95": interval,
            "n_by_role": n_by_role, "role_delta": role_delta,
            "post_merge_errors": {"candidate": escaped_candidate, "baseline": escaped_baseline},
            "gate": "at least 73 items per role; >=3 percentage points composite gain; paired 95% lower bound >0; no role below -5 points; no extra post-merge errors"}


def run_suite(pack: dict, complete, model: str, digest: str) -> dict:
    results = []
    for item in pack["items"]:
        started = time.monotonic()
        try:
            answer = complete(item["prompt"], 2048)
            passed, reason = score(answer, item)
        except Exception as exc:  # one failed endpoint request stays an item failure
            passed, reason = False, f"request_error:{type(exc).__name__}"
            answer = ""
        results.append({"id": item["id"], "track": item["track"], "role": item["role"],
                        "passed": passed, "reason": reason,
                        "response_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                        "latency_seconds": round(time.monotonic() - started, 3)})
    complete_status = "partial" if any(r["reason"].startswith("request_error:") for r in results) else "complete"
    return {"model": model, "subject_type": "model", "protocol": PROTOCOL, "pack_sha256": digest,
            "status": complete_status, "at": datetime.now(timezone.utc).isoformat(),
            "access_profile": "sandbox", "results": results,
            "tracks": summarize(pack["items"], results),
            "promotion_eligible": False,
            "reason": "small pilot pack; no role has a sufficiently narrow confidence interval"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--commitment", type=Path, default=COMMITMENT)
    parser.add_argument("--provider", choices=("local", "haiku55"), default="local")
    parser.add_argument("--base")
    parser.add_argument("--alias")
    parser.add_argument("--effort", choices=("low", "medium"), default="medium")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    pack, digest = load_pack(args.pack, args.commitment)
    commitment = json.loads(args.commitment.read_text())
    prior = json.loads(args.out.read_text()) if args.out.exists() else {"results": []}
    baseline_finished = any(
        row.get("pack_sha256") == digest and row.get("status") == "complete"
        and (row.get("runtime") or {}).get("provider") == "haiku55"
        for row in prior.get("results", [])
    )
    if baseline_finished:
        raise ValueError("this pack is closed by its Haiku 5.5 baseline; use a new sealed pack")
    if args.provider == "local":
        if not args.base or not args.base.startswith("http://127.0.0.1:") or not args.alias:
            raise ValueError("local provider requires --base on 127.0.0.1 and --alias")
    else:
        if not commitment.get("cloud_allowed") or commitment.get("status") != "sealed-cloud-safe-pilot":
            raise ValueError("this sealed pack is not approved for cloud disclosure")
        locals_done = {(r.get("runtime") or {}).get("alias") for r in prior.get("results", [])
                       if r.get("pack_sha256") == digest and r.get("status") == "complete"
                       and (r.get("runtime") or {}).get("provider") == "local"
                       and (r.get("runtime") or {}).get("alias")}
        if len(locals_done) < 2:
            raise ValueError("Haiku baseline is last: first complete two distinct local model aliases on this pack")
    if any(item["rubric"]["type"] == "code" for item in pack["items"]):
        subprocess.run(["docker", "image", "inspect", IMAGE], check=True,
                       stdout=subprocess.DEVNULL)

    if args.provider == "local":
        def complete(prompt: str, limit: int) -> str:
            body = {"model": args.alias, "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0, "max_tokens": limit,
                    "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"}}
            request = Request(args.base.rstrip("/") + "/v1/chat/completions",
                              data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=900) as reply:
                data = json.load(reply)
            return data["choices"][0]["message"].get("content") or ""
    else:
        def complete(prompt: str, _limit: int) -> str:
            env = {key: value for key, value in os.environ.items()
                   if not (key.startswith("CLAUDE_CODE_") or key == "CLAUDECODE")}
            with tempfile.TemporaryDirectory(prefix="haiku55-software-") as cwd:
                proc = subprocess.run(
                    ["claude", "-p", "--model", "claude-haiku-5-5", "--setting-sources", "",
                     "--strict-mcp-config", "--no-session-persistence", "--tools", "",
                     "--output-format", "json", "--effort", args.effort,
                     "--system-prompt", "You are a helpful assistant. Answer directly.", prompt],
                    cwd=cwd, env=env, text=True, capture_output=True, timeout=180)
            data = json.loads(proc.stdout)
            if proc.returncode or data.get("is_error") or not str(data.get("result", "")).strip():
                raise RuntimeError("Haiku CLI returned no valid answer")
            return str(data["result"])

    report = run_suite(pack, complete, args.model_id, digest)
    report["runtime"] = {"provider": args.provider, "alias": args.alias,
                         "endpoint": args.base if args.provider == "local" else "Anthropic Claude CLI",
                         "decode_profile": ({"temperature": 0, "max_tokens": 2048,
                                             "enable_thinking": False, "reasoning_effort": "low"}
                                            if args.provider == "local"
                                            else {"temperature": "CLI default", "max_tokens": "uncapped",
                                                  "thinking": "adaptive", "effort": args.effort}),
                         "code_image": IMAGE}
    upsert_result(args.out, report, {"suite": "software_data_specialists", "protocol": PROTOCOL,
                                    "results": []}, key=lambda row: (row.get("model"), row.get("pack_sha256")))
    print(json.dumps({"model": args.model_id, "tracks": report["tracks"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
