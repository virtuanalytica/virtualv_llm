#!/usr/bin/env python3
"""Freeze reviewed JSONL records into a private or filtered public MoM bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from knowledge_layer import build, canonical


def read_records(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"{path}:{line_no}: expected object")
                    records.append(row)
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", type=Path, help="reviewed JSONL records")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bundle-id", required=True)
    ap.add_argument("--scope", choices=("private", "public"), default="private")
    args = ap.parse_args()
    records = read_records(args.inputs)
    if args.scope == "public":
        records = [row for row in records if row.get("publish_allowed") is True]
    bundle = build(records, bundle_id=args.bundle_id, scope=args.scope)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(canonical(bundle) + b"\n")
    print(json.dumps({"path": str(args.out), "records": len(records),
                      "sha256": bundle["records_sha256"], "scope": args.scope}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
