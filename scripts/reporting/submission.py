#!/usr/bin/env python3
"""Community result submissions: build, validate and guard them.

    submission.py make      turn a local well_known_suite row into a submission
    submission.py validate  check every file under submissions/ (fails closed)
    submission.py guard     enforce the contribution protocol on a pull request

A submission is data, never code. The validator only parses JSON; it does not
import, execute or fetch anything a contributor controls. ``guard`` runs from
the base branch under ``pull_request_target`` and reads the pull request
through the GitHub API, so a pull request cannot weaken its own gate.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "config/submission_protocol.json").read_text())
SUBMISSIONS = ROOT / "submissions"
PATH_RE = re.compile(r"^submissions/([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/([a-z0-9][a-z0-9._-]{0,79})\.json$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
MAINTAINERS = {"OWNER", "MEMBER", "COLLABORATOR"}
PRIVATE_PATTERNS = (
    re.compile(r"/home/[a-z_][a-z0-9_-]*/"),
    re.compile(r"/media/[a-z_][a-z0-9_-]*/"),
    re.compile(r"(?:hf_[A-Za-z0-9]{24,}|sk-[A-Za-z0-9_-]{24,}|ghp_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{40,})"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
)
SHAPE = {
    "schema": str, "submission_id": str, "status": str, "submitted_at": str, "protocol": str,
    "submitter": {"github": str},
    "model": {"name": str, "source_repo": str, "source_revision": str, "quantization": str,
              "weight_sha256": str, "license": str},
    "runtime": {"engine": str, "engine_version": str, "context_tokens": int, "suite_commit": str},
    "hardware": {"gpus": list, "topology": str, "cuda_device_order": str},
    "scores": {name: {"correct": int, "n": int} for name in ("gsm8k", "bbh", "mmlu", "humaneval")},
    "evidence": {"url": str, "sha256": str, "bytes": int},
    "attestation": {"no_manual_score_edits": bool, "protocol_unmodified": bool,
                    "weights_license_respected": bool},
}
OPTIONAL = {
    "throughput": {"completion_tokens_per_second": (int, float), "batch": int, "max_tokens": int},
    "supersedes": str,
    "review": {"reviewer": str, "reviewed_at": str, "note": str},
}


def _check_shape(value: object, shape: object, where: str, errors: list[str]) -> None:
    if isinstance(shape, dict):
        if not isinstance(value, dict):
            errors.append(f"{where}: expected an object")
            return
        for key in sorted(set(shape) - set(value)):
            errors.append(f"{where}.{key}: missing")
        for key in sorted(set(value) - set(shape)):
            errors.append(f"{where}.{key}: unknown field")
        for key in set(shape) & set(value):
            _check_shape(value[key], shape[key], f"{where}.{key}", errors)
    # bool is an int subclass; a count or context size must not be true/false.
    elif not isinstance(value, shape) or (isinstance(value, bool) and shape is not bool):
        errors.append(f"{where}: wrong type")


def validate_document(text: str, repo_path: str, *, external: bool) -> list[str]:
    """Return every protocol violation in one submission; empty means valid."""
    errors: list[str] = []
    match = PATH_RE.match(repo_path)
    if not match:
        return [f"{repo_path}: path must be submissions/<github-handle>/<model-slug>.json"]
    handle, slug = match.groups()
    if len(text.encode()) > CONFIG["max_submission_bytes"]:
        return [f"{repo_path}: larger than {CONFIG['max_submission_bytes']} bytes"]
    for pattern in PRIVATE_PATTERNS:
        if pattern.search(text):
            errors.append(f"{repo_path}: private path or credential-like value ({pattern.pattern})")
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        return errors + [f"{repo_path}: invalid JSON ({exc.msg})"]
    if not isinstance(doc, dict):
        return errors + [f"{repo_path}: top level must be an object"]
    required = {k: v for k, v in doc.items() if k not in OPTIONAL}
    _check_shape(required, SHAPE, repo_path, errors)
    for key, shape in OPTIONAL.items():
        if key in doc:
            _check_shape(doc[key], shape, f"{repo_path}.{key}", errors)
    if errors:
        return errors

    def fail(message: str) -> None:
        errors.append(f"{repo_path}: {message}")

    if doc["schema"] != CONFIG["schema"]:
        fail(f"schema must be {CONFIG['schema']}")
    if doc["submission_id"] != f"{handle}/{slug}":
        fail("submission_id must equal <handle>/<model-slug> from the path")
    if doc["submitter"]["github"].lower() != handle.lower():
        fail("submitter.github must match the directory name")
    allowed = [CONFIG["external_status"]] if external else CONFIG["maintainer_statuses"]
    if doc["status"] not in allowed:
        fail(f"status must be one of {allowed}")
    if external and "review" in doc:
        fail("only a maintainer may add a review block")
    if doc["status"] == "verified-reproduced" and "review" not in doc:
        fail("verified-reproduced requires a review block naming the reproducing maintainer")
    try:
        if date.fromisoformat(doc["submitted_at"]) > date.today():
            fail("submitted_at is in the future")
    except ValueError:
        fail("submitted_at must be an ISO date (YYYY-MM-DD)")
    sizes = CONFIG["accepted_protocols"].get(doc["protocol"])
    if sizes is None:
        fail(f"protocol {doc['protocol']!r} is not accepted; current: {sorted(CONFIG['accepted_protocols'])}")
    else:
        for name, expected in sizes.items():
            score = doc["scores"][name]
            if score["n"] != expected:
                fail(f"scores.{name}.n must be {expected} under {doc['protocol']}")
            if not 0 <= score["correct"] <= score["n"]:
                fail(f"scores.{name}.correct must lie between 0 and n")
    if not SHA256_RE.match(doc["model"]["weight_sha256"]):
        fail("model.weight_sha256 must be 64 lowercase hex characters")
    if not REVISION_RE.match(doc["model"]["source_revision"]):
        fail("model.source_revision must be a full 40-character commit hash, not a branch or tag")
    if not REVISION_RE.match(doc["runtime"]["suite_commit"]):
        fail("runtime.suite_commit must be the full commit hash of the suite that produced the run")
    if not SHA256_RE.match(doc["evidence"]["sha256"]):
        fail("evidence.sha256 must be 64 lowercase hex characters")
    if not doc["evidence"]["url"].startswith("https://") or doc["evidence"]["bytes"] <= 0:
        fail("evidence must be a non-empty archive behind an https:// URL")
    if doc["hardware"]["cuda_device_order"] != "PCI_BUS_ID":
        fail("hardware.cuda_device_order must be PCI_BUS_ID")
    gpus = doc["hardware"]["gpus"]
    if not gpus or not all(isinstance(g, str) and g.strip() for g in gpus):
        fail("hardware.gpus must list every physical GPU by name")
    if doc["runtime"]["context_tokens"] <= 0:
        fail("runtime.context_tokens must be positive")
    for key, value in doc["attestation"].items():
        if value is not True:
            fail(f"attestation.{key} must be true")
    if "throughput" in doc and not (0 < doc["throughput"]["completion_tokens_per_second"] < 1e5
                                    and doc["throughput"]["batch"] >= 1):
        fail("throughput must be positive with batch >= 1")
    for section in ("model", "runtime"):
        for key, value in doc[section].items():
            if isinstance(value, str) and not value.strip():
                fail(f"{section}.{key} must not be empty")
    return errors


def composite(doc: dict) -> float:
    return sum(s["correct"] / s["n"] for s in doc["scores"].values()) / len(doc["scores"])


def load_valid(root: Path = SUBMISSIONS) -> list[dict]:
    """All committed submissions, for the dashboard. Raises on an invalid file."""
    documents = []
    for path in sorted(root.glob("*/*.json")):
        text = path.read_text()
        errors = validate_document(text, path.relative_to(root.parent).as_posix(), external=False)
        if errors:
            raise ValueError("; ".join(errors))
        documents.append(json.loads(text))
    return documents


def guard_rules(files: list[dict], author: str, association: str) -> list[str]:
    """Path-level protocol for a pull request. ``files`` holds the API's
    ``filename``/``status`` (and ``previous_filename`` on renames)."""
    if association in MAINTAINERS:
        return []
    errors = []
    touched = []
    for item in files:
        touched.append((item["filename"], item["status"]))
        if item.get("previous_filename"):
            touched.append((item["previous_filename"], "removed"))
    submissions = [(name, status) for name, status in touched if name.startswith("submissions/")
                   and name != "submissions/README.md"]
    others = [name for name, _ in touched if not name.startswith("submissions/")]
    for name, _ in touched:
        if name == "submissions/README.md" or any(
                name == p or (p.endswith("/") and name.startswith(p)) for p in CONFIG["protected_paths"]):
            errors.append(f"{name}: maintainer-only path; open an issue instead of editing it")
    if submissions and others:
        errors.append("a result submission must not be combined with other changes in one pull request")
    if len(submissions) > CONFIG["max_submissions_per_pull_request"]:
        errors.append(f"at most {CONFIG['max_submissions_per_pull_request']} submissions per pull request")
    for name, status in submissions:
        match = PATH_RE.match(name)
        if not match:
            errors.append(f"{name}: path must be submissions/<github-handle>/<model-slug>.json")
        elif match.group(1).lower() != author.lower():
            errors.append(f"{name}: you may only add files under submissions/{author}/")
        if status != "added":
            errors.append(f"{name}: submissions are append-only; correct one with a new file that sets 'supersedes'")
    return errors


def _api(url: str, raw: bool = False) -> bytes:
    request = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github.raw+json" if raw else "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read(CONFIG["max_submission_bytes"] * 64 if not raw else CONFIG["max_submission_bytes"] + 1)


def is_maintainer(base: str, author: str, association: str) -> bool:
    """Whether the author may change protected paths.

    The event's author_association says CONTRIBUTOR for an organisation member
    whose membership is private, so it cannot be the only test. The author's
    actual permission on the repository decides; an API failure counts as no.
    """
    if association in MAINTAINERS:
        return True
    try:
        level = json.loads(_api(f"{base}/collaborators/{urllib.parse.quote(author, safe='')}/permission"))
    except Exception:
        return False
    return level.get("permission") in ("admin", "maintain", "write")


def cmd_guard(args: argparse.Namespace) -> int:
    base = f"https://api.github.com/repos/{args.repo}"
    if is_maintainer(base, args.author, args.association):
        args.association = "COLLABORATOR"
    files: list[dict] = []
    for page in range(1, 31):
        batch = json.loads(_api(f"{base}/pulls/{args.pull_request}/files?per_page=100&page={page}"))
        files.extend(batch)
        if len(batch) < 100:
            break
    errors = guard_rules(files, args.author, args.association)
    external = args.association not in MAINTAINERS
    for item in files:
        name = item["filename"]
        if item["status"] == "removed" or not PATH_RE.match(name):
            continue
        text = _api(f"{base}/contents/{urllib.parse.quote(name)}?ref={args.head_sha}", raw=True).decode("utf-8", "replace")
        errors.extend(validate_document(text, name, external=external))
    for error in errors:
        print(f"::error::{error}")
    print(f"contribution guard: {len(files)} changed files, {len(errors)} violations "
          f"({'external' if external else 'maintainer'} author)")
    return 1 if errors else 0


def cmd_validate(args: argparse.Namespace) -> int:
    errors = []
    paths = sorted(SUBMISSIONS.glob("**/*.json"))
    for path in paths:
        errors.extend(validate_document(path.read_text(), path.relative_to(ROOT).as_posix(), external=False))
    stray = [p for p in SUBMISSIONS.rglob("*") if p.is_file() and p.suffix != ".json" and p.name != "README.md"]
    errors.extend(f"{p.relative_to(ROOT)}: only JSON submissions belong here" for p in stray)
    for error in errors:
        print(error, file=sys.stderr)
    print(f"validated {len(paths)} community submissions, {len(errors)} violations")
    return 1 if errors else 0


def cmd_make(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(ROOT / "scripts/benchmarks"))
    import score_confidence  # noqa: E402

    rows = {r.get("model"): r for r in json.loads(args.report.read_text()).get("results", [])}
    row = rows.get(args.model)
    if row is None or "error" in row:
        raise SystemExit(f"{args.model}: no error-free row in {args.report}")
    parts = score_confidence.components(row)
    if parts is None:
        raise SystemExit(f"{args.model}: row lacks one of GSM8K, BBH, MMLU, HumanEval")
    scores = {}
    for name, (accuracy, n) in parts.items():
        correct = round(accuracy * n)
        if abs(correct - accuracy * n) > 0.05:
            raise SystemExit(f"{name}: accuracy {accuracy} is not a whole number of {n} items")
        scores[name] = {"correct": correct, "n": n}
    doc = {
        "schema": CONFIG["schema"],
        "submission_id": f"{args.github}/{args.slug}",
        "status": CONFIG["external_status"],
        "submitted_at": date.today().isoformat(),
        "protocol": row.get("eval_protocol"),
        "submitter": {"github": args.github},
        "model": {"name": args.model, "source_repo": args.source_repo, "source_revision": args.source_revision,
                  "quantization": args.quantization, "weight_sha256": args.weight_sha256, "license": args.license},
        "runtime": {"engine": args.engine, "engine_version": args.engine_version,
                    "context_tokens": args.context_tokens, "suite_commit": args.suite_commit},
        "hardware": {"gpus": args.gpu, "topology": args.topology, "cuda_device_order": "PCI_BUS_ID"},
        "scores": scores,
        "evidence": {"url": args.evidence_url, "sha256": args.evidence_sha256, "bytes": args.evidence_bytes},
        "attestation": {"no_manual_score_edits": True, "protocol_unmodified": True,
                        "weights_license_respected": True},
    }
    if isinstance(row.get("completion_tokens_per_second"), (int, float)):
        doc["throughput"] = {"completion_tokens_per_second": round(row["completion_tokens_per_second"], 2),
                             "batch": 1, "max_tokens": 256}
    repo_path = f"submissions/{args.github}/{args.slug}.json"
    text = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
    errors = validate_document(text, repo_path, external=True)
    if errors:
        raise SystemExit("\n".join(errors))
    target = ROOT / repo_path
    if target.exists():
        raise SystemExit(f"{repo_path} exists; submissions are append-only, pick a new slug and set 'supersedes'")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    print(f"wrote {repo_path} (composite {composite(doc):.4f})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate").set_defaults(run=cmd_validate)
    guard = commands.add_parser("guard")
    for flag in ("--repo", "--pull-request", "--head-sha", "--author", "--association"):
        guard.add_argument(flag, required=True)
    guard.set_defaults(run=cmd_guard)
    make = commands.add_parser("make")
    make.add_argument("model", help="model identifier of the row in --report")
    make.add_argument("--report", type=Path, required=True, help="local well_known_suite.py --out file")
    for flag in ("--github", "--slug", "--source-repo", "--source-revision", "--quantization", "--weight-sha256",
                 "--license", "--engine", "--engine-version", "--suite-commit", "--topology", "--evidence-url",
                 "--evidence-sha256"):
        make.add_argument(flag, required=True)
    make.add_argument("--context-tokens", type=int, required=True)
    make.add_argument("--evidence-bytes", type=int, required=True)
    make.add_argument("--gpu", action="append", required=True, help="one physical GPU name; repeat per card")
    make.set_defaults(run=cmd_make)
    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
