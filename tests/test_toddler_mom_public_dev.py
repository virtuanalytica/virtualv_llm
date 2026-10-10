"""Public development export must keep paired items and fixed decoding."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "benchmarks"))
import toddler_mom_public_dev as mom  # noqa: E402


def test_procedural_pack_is_answer_keyed_and_has_unique_postweight_ids():
    first, second = mom.make_pack(123), mom.make_pack(123)
    assert [(row["id"], row["prompt"], row["answer"]) for row in first["items"]] == [
        (row["id"], row["prompt"], row["answer"]) for row in second["items"]]
    assert len(first["items"]) == 90
    assert len({row["id"] for row in first["items"]}) == 90
    assert all(row["answer"].isdigit() for row in first["items"])


def test_json_answer_parser_rejects_truncated_or_explanatory_text():
    assert mom.integer_answer('{"answer": 42}') == "42"
    assert mom.integer_answer('{"answer": true}') is None
    assert mom.integer_answer("42") is None
    assert mom.integer_answer('{"answer":') is None


def test_assemble_rejects_misaligned_model_rows(tmp_path):
    pack = mom.make_pack(123)
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(json.dumps(pack))
    pack_hash = hashlib.sha256(pack_path.read_bytes()).hexdigest()

    def write_model(name):
        tasks = {}
        for task in mom.TASKS:
            items = [item for item in pack["items"] if item["task"] == task]
            ids = [item["id"] for item in items]
            tasks[task] = {"n": len(ids), "item_ids": ids,
                           "item_ids_sha256": mom.ordered_id_hash(ids),
                           "item_scores": [0.0] * len(ids), "quality": 0.0,
                           "evidence": [{"id": item["id"], "response": '{"answer": -1}',
                                         "parsed": "-1", "finish_reason": "stop", "score": 0.0}
                                        for item in items],
                           "latency_s": 1.0, "decode_tps": 10.0,
                           "gpu_board_wh_per_answer": 0.01}
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"model": name, "weights_sha256": "a" * 64,
                                    "access": "local", "energy_scope": "gpu_board",
                                    "resident_vram_gb": 10, "tasks": tasks,
                                    "pack_sha256": pack_hash,
                                    "decode_profile_sha256": hashlib.sha256(mom.canonical(mom.DECODE)).hexdigest()}))
        return path

    a, b = write_model("a"), write_model("b")
    result = mom.assemble(pack_path, [a, b])
    assert result["schema"] == "toddler-mom-public-dev/v1"
    assert result["promotion_eligible"] is False
    modified = json.loads(b.read_text())
    modified["tasks"]["arithmetic"]["item_ids"].reverse()
    b.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="paired public item scores"):
        mom.assemble(pack_path, [a, b])
