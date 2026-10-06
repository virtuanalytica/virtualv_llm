"""Small, auditable knowledge bundle for the live MoM.

The runtime reads a frozen bundle. Building or publishing one is an explicit
offline action; a serving process never indexes private files on its own.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "virtualv-knowledge/v1"
WORDS = re.compile(r"[\w.-]{3,}", re.UNICODE)
PRIVATE_MARKERS = re.compile(
    r"(?:/home/|/media/|-----BEGIN .*PRIVATE KEY|\b(?:api[_-]?key|password|bearer)\s*[:=])",
    re.IGNORECASE,
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(records: list[dict]) -> str:
    return hashlib.sha256(canonical(records)).hexdigest()


def normalise(text: str) -> str:
    return " ".join(text.casefold().split())


def terms(text: str) -> set[str]:
    return set(WORDS.findall(normalise(text)))


def validate_record(record: dict) -> None:
    if record.get("kind") not in {"fact", "rule", "code"}:
        raise ValueError("record kind must be fact, rule or code")
    for key in ("id", "text", "source"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"record needs nonempty {key}")
    if not isinstance(record.get("publish_allowed"), bool):
        raise ValueError("record needs explicit publish_allowed boolean")
    if record["kind"] == "code":
        for key in ("repo", "commit", "path"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise ValueError(f"code record needs {key}")
        if not isinstance(record.get("line"), int) or record["line"] < 1:
            raise ValueError("code record needs positive line")
    if record.get("answer") and not record.get("query"):
        raise ValueError("a direct answer needs an exact query")
    if len(record["text"]) > 2000 or len(record.get("answer", "")) > 2000:
        raise ValueError("record text/answer exceeds 2000 characters")


def validate_public_record(record: dict) -> None:
    validate_record(record)
    if not record["publish_allowed"]:
        raise ValueError("public bundle contains a private record")
    if not record.get("license") or not record["source"].startswith("https://"):
        raise ValueError("public record needs source URL and license")
    for key in ("id", "text", "query", "answer", "source", "repo", "path", "license"):
        if PRIVATE_MARKERS.search(str(record.get(key, ""))):
            raise ValueError(f"public record {record['id']} contains private marker in {key}")
    if record["kind"] == "code" and (Path(record["path"]).is_absolute() or
                                      ".." in Path(record["path"]).parts or
                                      not str(record["repo"]).startswith("https://")):
        raise ValueError("public code record needs relative path and public repo URL")


def build(records: list[dict], *, bundle_id: str, scope: str) -> dict:
    if scope not in {"private", "public"}:
        raise ValueError("scope must be private or public")
    records = sorted(records, key=lambda row: row["id"])
    ids = set()
    for row in records:
        validate_record(row)
        if row["id"] in ids:
            raise ValueError(f"duplicate record id: {row['id']}")
        ids.add(row["id"])
        if scope == "public":
            validate_public_record(row)
    return {"schema": SCHEMA, "bundle_id": bundle_id, "scope": scope,
            "records_sha256": digest(records), "records": records}


def load(path: Path) -> dict:
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if bundle.get("schema") != SCHEMA or bundle.get("scope") not in {"private", "public"}:
        raise ValueError("unsupported knowledge bundle")
    expected = build(bundle["records"], bundle_id=bundle["bundle_id"], scope=bundle["scope"])
    if bundle.get("records_sha256") != expected["records_sha256"] or bundle["records"] != expected["records"]:
        raise ValueError("knowledge bundle digest/order mismatch")
    return bundle


@dataclass(frozen=True)
class Hit:
    record: dict
    score: int


def retrieve(bundle: dict | None, question: str, task: str = "", limit: int = 3) -> list[Hit]:
    if not bundle or not question:
        return []
    query_terms = terms(question)
    hits = []
    for record in bundle["records"]:
        tasks = record.get("tasks", [])
        if tasks and task not in tasks:
            continue
        if record["kind"] == "code" and task != "coding":
            continue
        exact = normalise(question) == normalise(record.get("query", ""))
        overlap = len(query_terms & terms(record["text"] + " " + record.get("query", "")))
        if exact or overlap >= 2:
            hits.append(Hit(record, (1000 if exact else 0) + overlap))
    return sorted(hits, key=lambda hit: (-hit.score, hit.record["id"]))[:limit]


def direct_answer(hits: list[Hit], question: str) -> dict | None:
    """Only exact, explicitly authored question/answer pairs bypass the model."""
    for hit in hits:
        row = hit.record
        if row["kind"] in {"fact", "rule"} and row.get("answer") and normalise(question) == normalise(row["query"]):
            return row
    return None


def context(hits: list[Hit]) -> str:
    lines = []
    for hit in hits:
        row = hit.record
        location = ""
        if row["kind"] == "code":
            location = f" ({row['repo']}@{row['commit']} {row['path']}:{row['line']})"
        lines.append(f"[{row['id']}] {row['text']}{location} Source: {row['source']}")
    return "\n".join(lines)
