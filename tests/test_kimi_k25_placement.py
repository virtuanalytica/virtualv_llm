"""Per-layer placement must preserve activation skew and capacity limits."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("k25_placement", ROOT / "infra/kimi_k25/placement.py")
placement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(placement)


def test_real_calibration_has_all_23040_instances():
    path = ROOT / "evidence/kimi_k25/unsloth_imatrix_expert_counts.jsonl"
    rows, source = placement.read_counts(path)
    result = placement.plan(rows, source, expert_bytes=1, gpu_bytes=0,
                            dram_bytes=60 * 100, pmem0_bytes=60 * 142,
                            pmem1_bytes=60 * 142)
    assert len(result["assignments"]) == 60 * 384
    assert result["used_bytes"]["disk"] == 0
    assert result["estimated_routed_traffic_share"]["dram"] > 6000 / 23040
    assert abs(sum(result["estimated_routed_traffic_share"].values()) - 1) < .00001


def test_hot_expert_id_is_chosen_independently_per_layer():
    rows = [(layer, [100 if expert == (layer % 384) else 1 for expert in range(384)])
            for layer in range(1, 61)]
    result = placement.plan(rows, "synthetic", expert_bytes=1, gpu_bytes=0,
                            dram_bytes=2, pmem0_bytes=0, pmem1_bytes=0)
    hot = {(row["layer"], row["expert"]) for row in result["assignments"]
           if row["tier"] == "dram"}
    assert hot == {(1, 1), (2, 2)}


def test_rejects_missing_layer(tmp_path):
    path = tmp_path / "counts.jsonl"
    path.write_text('{"layer":1,"counts":[' + ','.join(['1'] * 384) + ']}\n')
    with pytest.raises(ValueError, match="cover"):
        placement.read_counts(path)
