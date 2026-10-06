#!/usr/bin/env python3
"""Convert existing LightRAG JSONL and GitNexus symbol exports to reviewable records.

All imported rows are private by default. A reviewer may create a separate
public JSONL with explicit per-record publication permission and licence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from knowledge_layer import canonical


def stable_id(prefix: str, *parts: str) -> str:
    return prefix + ":" + hashlib.sha256("\n".join(parts).encode()).hexdigest()[:20]


def import_lightrag(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        content = str(row.get("content") or row.get("text") or "").strip()
        if not content:
            continue
        source = str(row.get("source_path") or row.get("source") or path)
        records.append({"id": stable_id("lr", source, content), "kind": "fact",
                        "text": content[:2000], "source": source,
                        "source_type": str(row.get("source_type") or row.get("kind") or "lightrag"),
                        "publish_allowed": False})
    return records


def import_gitnexus_symbols(path: Path, repo: str, commit: str,
                            include_prefixes: tuple[str, ...] = ()) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        source_path = str(row.get("source_path") or "")
        if not source_path or Path(source_path).is_absolute() or ".." in Path(source_path).parts:
            continue
        if include_prefixes and not source_path.startswith(include_prefixes):
            continue
        line_no = row.get("line") or 1
        if not isinstance(line_no, int) or line_no < 1:
            continue
        symbol = str(row.get("qualname") or row.get("name") or "").strip()
        if not symbol:
            continue
        description = str(row.get("doc") or "").strip().splitlines()[0][:500]
        text = f"{row.get('kind', 'symbol')} {symbol} in {source_path}:{line_no}. {description}".strip()
        source = (f"{repo.rstrip('/')}/blob/{commit}/{source_path}#L{line_no}"
                  if repo.startswith("https://") else f"{repo}@{commit}:{source_path}:{line_no}")
        records.append({"id": stable_id("gn", repo, commit, source_path, symbol, str(line_no)),
                        "kind": "code", "text": text, "source": source,
                        "repo": repo, "commit": commit, "path": source_path, "line": line_no,
                        "tasks": ["coding"], "publish_allowed": False})
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lightrag-facts", type=Path, action="append", default=[])
    ap.add_argument("--gitnexus-symbols", type=Path, action="append", default=[])
    ap.add_argument("--repo", default="", help="repository URL/name for GitNexus records")
    ap.add_argument("--commit", default="", help="indexed commit for GitNexus records")
    ap.add_argument("--include-prefix", action="append", default=[],
                    help="only include GitNexus paths under these repo-relative prefixes")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.gitnexus_symbols and (not args.repo or not args.commit):
        ap.error("--gitnexus-symbols requires --repo and --commit")
    records = []
    for path in args.lightrag_facts:
        records.extend(import_lightrag(path))
    for path in args.gitnexus_symbols:
        records.extend(import_gitnexus_symbols(path, args.repo, args.commit,
                                               tuple(args.include_prefix)))
    by_id = {row["id"]: row for row in records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as handle:
        for row in sorted(by_id.values(), key=lambda item: item["id"]):
            handle.write(canonical(row) + b"\n")
    print(json.dumps({"path": str(args.out), "records": len(by_id), "publish_allowed": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
