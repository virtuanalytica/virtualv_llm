"""Read-only Lens ranking over a frozen, locally reviewed knowledge bundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from knowledge_layer import Hit, allowed_for_task, load, terms


def rows_for_bundle(bundle: dict) -> dict:
    return {"schema": "virtualv-lens-rows/v1", "bundle_sha256": bundle["records_sha256"],
            "rows": [{"id": row["id"], "title": row.get("query") or row["id"],
                      "text": row["text"], "source_uri": row["source"],
                      "kind": row["kind"], "weight": 1} for row in bundle["records"]]}


class LensBridge:
    def __init__(self, bundle: dict, rows_path: Path, lens_root: Path) -> None:
        document = json.loads(rows_path.read_text(encoding="utf-8"))
        if (document.get("schema") != "virtualv-lens-rows/v1" or
                document.get("bundle_sha256") != bundle["records_sha256"]):
            raise ValueError("Lens rows do not match the frozen knowledge bundle")
        self.records = {row["id"]: row for row in bundle["records"]}
        rows = document["rows"]
        if (len(rows) != len(self.records) or len({row["id"] for row in rows}) != len(rows) or
                any(row["id"] not in self.records or row["text"] != self.records[row["id"]]["text"] or
                    row["source_uri"] != self.records[row["id"]]["source"] for row in rows)):
            raise ValueError("Lens rows differ from the frozen knowledge bundle")
        sys.path.insert(0, str(lens_root / "src"))
        from knitweb_lens import MappingRowsAdapter, Retriever
        self.chunks = tuple(MappingRowsAdapter(rows).iter_chunks())
        self.retriever = Retriever()

    def retrieve(self, question: str, task: str, limit: int = 3) -> list[Hit]:
        if not question:
            return []
        out = []
        for ranked in self.retriever.rank(question, self.chunks):
            record = self.records[ranked.chunk.ref.node_id]
            overlap = len(terms(question) & terms(record["text"] + " " + record.get("query", "")))
            if ranked.lexical_score <= 0 or overlap < 2 or not allowed_for_task(record, task):
                continue
            out.append(Hit(record, ranked.lexical_score))
            if len(out) >= limit:
                break
        return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    document = rows_for_bundle(load(args.bundle))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"rows": len(document["rows"]), "bundle_sha256": document["bundle_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
