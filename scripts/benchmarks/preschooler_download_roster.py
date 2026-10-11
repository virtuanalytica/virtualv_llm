#!/usr/bin/env python3
"""Fetch pinned small-model candidates outside Git with disk and hash checks.

This script downloads data only. It never imports or executes model repository code.
The default action is a read-only size and license preflight.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
ROSTER = ROOT / "config/preschooler_teacher_roster.json"
DEFAULT_STORE = Path("/media/knight2/claude-data/knight1/models/preschooler-teachers")
SMALL_FILES = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "chat_template.jinja", "vocab.json", "merges.txt",
               "README.md", "LICENSE", "LICENSE.md"}
MIN_HEADROOM = 200_000_000_000


def selected_files(info, pattern: str) -> list:
    files = []
    for entry in info.siblings:
        name = entry.rfilename
        if fnmatch.fnmatch(name, pattern):
            files.append(entry)
        elif Path(name).name in SMALL_FILES or name.endswith((".model", ".tiktoken", ".index.json")):
            files.append(entry)
        elif name.endswith(".py") and (entry.size or 0) < 1_000_000:
            # Store remote-code source for review; it is never executed here.
            files.append(entry)
    if not any(fnmatch.fnmatch(f.rfilename, pattern) for f in files):
        raise ValueError(f"no weights match {pattern} in {info.id}")
    return files


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def plan(model: dict, api: HfApi) -> tuple[object, list, int]:
    info = api.model_info(model["repo"], revision=model["revision"], files_metadata=True)
    license_id = (info.cardData or {}).get("license")
    if info.sha != model["revision"] or info.gated or license_id != model["license_id"]:
        raise ValueError(f"{model['alias']}: revision, access or license differs from pinned roster")
    if license_id not in {"apache-2.0", "mit"}:
        raise ValueError(f"{model['alias']}: license needs review")
    files = selected_files(info, model["weight_pattern"])
    size = sum(f.size or 0 for f in files)
    return info, files, size


def fetch(model: dict, info: object, files: list, store: Path) -> dict:
    alias = model["alias"]
    destination = store / alias
    missing = sum((f.size or 0) for f in files if not (destination / f.rfilename).exists())
    if shutil.disk_usage(store).free < missing + MIN_HEADROOM:
        raise RuntimeError(f"{alias}: less than 200 GB will remain on model SSD")
    snapshot_download(repo_id=model["repo"], revision=model["revision"],
                      local_dir=destination, allow_patterns=[f.rfilename for f in files])
    verified = {}
    for entry in files:
        path = destination / entry.rfilename
        if not path.is_file() or (entry.size is not None and path.stat().st_size != entry.size):
            raise RuntimeError(f"{alias}: missing or truncated {entry.rfilename}")
        digest = sha256(path)
        upstream = (entry.lfs or {}).get("sha256") if entry.lfs else None
        if upstream and digest != upstream:
            raise RuntimeError(f"{alias}: SHA-256 mismatch for {entry.rfilename}")
        verified[entry.rfilename] = {"bytes": path.stat().st_size, "sha256": digest,
                                     "upstream_sha256": upstream}
    result = {"schema": "preschooler-local-snapshot/v1", "alias": alias,
              "repo": model["repo"], "revision": info.sha,
              "license_id": model["license_id"], "status": "download_verified_loader_untested",
              "captured_at": datetime.now(timezone.utc).isoformat(), "files": verified,
              "output_training_rights": "pending_per_model_review"}
    (destination / "snapshot-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, default=ROSTER)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--model", action="append", help="alias; repeat for a subset")
    parser.add_argument("--download", action="store_true", help="fetch selected pinned snapshots")
    args = parser.parse_args()
    roster = json.loads(args.roster.read_text())
    if roster.get("schema") != "preschooler-teacher-roster/v1":
        parser.error("unexpected roster schema")
    models = [m for m in roster["models"] if not args.model or m["alias"] in args.model]
    if args.model and len(models) != len(set(args.model)):
        parser.error("unknown or duplicated model alias")
    args.store.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    planned = []
    for model in models:
        info, files, size = plan(model, api)
        planned.append((model, info, files, size))
        print(json.dumps({"alias": model["alias"], "revision": info.sha,
                          "license": model["license_id"], "files": len(files),
                          "download_gb": round(size / 1e9, 3)}), flush=True)
    needed = sum(size for model, _, _, size in planned
                 if not (args.store / model["alias"] / "snapshot-manifest.json").is_file())
    if shutil.disk_usage(args.store).free < needed + MIN_HEADROOM:
        raise RuntimeError("roster exceeds available SSD headroom")
    print(json.dumps({"models": len(planned), "estimated_missing_gb": round(needed / 1e9, 3),
                      "free_gb": round(shutil.disk_usage(args.store).free / 1e9, 3),
                      "download": args.download}), flush=True)
    if args.download:
        for model, info, files, _ in planned:
            result = fetch(model, info, files, args.store)
            print(json.dumps({"alias": result["alias"], "status": result["status"]}), flush=True)


if __name__ == "__main__":
    main()
