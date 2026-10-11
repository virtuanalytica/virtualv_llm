"""A router ceiling must use identical sealed items and remain labeled offline."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmarks/preschooler_mixture_frontier.py"
spec = importlib.util.spec_from_file_location("preschooler_mixture_frontier", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(model, passed, pack="a" * 64):
    return {"model": model, "protocol": "private-v1", "results": {
        "software": {"status": "complete", "pack_sha256": pack,
                     "samples": [{"id": str(i), "correct": value} for i, value in enumerate(passed)]}}}


def test_all_ten_model_subsets_are_offline_bounds():
    rows = [row(f"m{i}", [item % 5 == i % 5 for item in range(20)]) for i in range(10)]
    result = module.enumerate_subsets(rows, {f"m{i}": f"family{i // 2}" for i in range(10)})
    assert result["subset_count"] == 2 ** 10 - 10 - 1
    assert result["status"] == "offline_oracle_not_live_mixture"
    assert result["subsets"][0]["oracle_upper_bound"] == 1.0
    assert result["best_fixed"]["accuracy"] == 0.2


def test_mismatched_pack_is_rejected_instead_of_reducing_item_set():
    with pytest.raises(ValueError, match="identical items"):
        module.enumerate_subsets([row("a", [True] * 20), row("b", [True] * 20, "b" * 64)],
                                 {"a": "A", "b": "B"})
