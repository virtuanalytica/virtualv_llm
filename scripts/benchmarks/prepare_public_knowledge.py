#!/usr/bin/env python3
"""Stage reviewed public knowledge for Lens, Knitweb/Pulse and FieldIntelligence.

This command writes local exports only. Signing and sending them to a public
Knitweb/Pulse node are separate operator actions after reviewing these bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from knowledge_layer import build, canonical, load


def exports(private_bundle: dict) -> dict[str, dict]:
    public = build([row for row in private_bundle["records"] if row["publish_allowed"]],
                   bundle_id=private_bundle["bundle_id"] + "-public", scope="public")
    rows = public["records"]
    lens = {"rows": [{"id": row["id"], "title": row.get("query") or row["id"],
                      "content": row["text"], "source": row["source"],
                      "source_uri": row["source"],
                      "relationships": [], "weight": 1} for row in rows]}
    field = {"records": [{"id": row["id"], "title": row.get("query") or row["id"],
                          "body": row["text"], "tags": [row["kind"]],
                          "source": row["source"], "license": row["license"]} for row in rows]}
    relations = []
    for row in rows:
        subject = "virtualv:" + row["id"]
        relations.extend([
            {"subject": subject, "predicate": "hasText", "object": row["text"],
             "type": "Webpage"},
            {"subject": subject, "predicate": "hasSource", "object": row["source"],
             "type": "Webpage"},
        ])
        if row["kind"] == "code":
            relations.append({"subject": subject, "predicate": "definedIn",
                              "object": f"{row['repo']}@{row['commit']}:{row['path']}:{row['line']}",
                              "type": "Webpage"})
    knitweb = {"@id": "virtualv:" + public["records_sha256"],
               "originator": "virtuanalytica/virtualv_llm", "@graph": relations}
    return {"public_bundle.json": public, "lens_rows.json": lens,
            "field_snapshot.json": field, "knitweb_asset.json": knitweb}


def stage(bundle_path: Path, out_dir: Path, knitweb_root: Path | None = None) -> dict:
    documents = exports(load(bundle_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for name, document in documents.items():
        payload = canonical(document) + b"\n"
        (out_dir / name).write_bytes(payload)
        checksums[name] = hashlib.sha256(payload).hexdigest()
    manifest = {"schema": "virtualv-public-stage/v1",
                "source_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
                "records": len(documents["public_bundle.json"]["records"]),
                "files_sha256": checksums, "published": False, "signed": False}
    if knitweb_root:
        sys.path.insert(0, str(knitweb_root / "src"))
        from knitweb.synaptic.bytecode import Relation, bundle_digest, compile_bundle, decode_bundle
        asset = documents["knitweb_asset.json"]
        relations = [Relation(row["subject"], row["predicate"], row["object"], row["type"])
                     for row in asset["@graph"]]
        bytecode = compile_bundle(asset["@id"], asset["originator"], relations)
        decode_bundle(bytecode)
        name = "knitweb.synaptic.unsigned"
        (out_dir / name).write_bytes(bytecode)
        manifest["files_sha256"][name] = hashlib.sha256(bytecode).hexdigest()
        manifest["knitweb_digest"] = bundle_digest(bytecode)
    (out_dir / "manifest.json").write_bytes(canonical(manifest) + b"\n")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--knitweb-root", type=Path, help="optional local Knitweb checkout for unsigned bytecode")
    args = ap.parse_args()
    print(json.dumps(stage(args.bundle, args.out_dir, args.knitweb_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
