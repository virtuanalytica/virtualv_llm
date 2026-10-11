"""Subsecond answers still need a measured GPU board-energy interval."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmarks/preschooler_private_bridge.py"
spec = importlib.util.spec_from_file_location("preschooler_private_bridge", SCRIPT)
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def test_subsecond_board_energy_is_interpolated():
    samples = [{"epoch": t, "gpus": [{"index": 0, "watts": w}]} for t, w in
               ((0.0, 20.0), (2.0, 40.0), (4.0, 60.0))]
    result = bridge.board_energy(samples, 1.0, 1.5, {0})
    assert result["status"] == "complete"
    assert result["gpu_board_joules"] == 16.25
    assert result["gpu_board_wh"] > 0


def test_unbounded_window_is_rejected():
    samples = [{"epoch": 0.0, "gpus": [{"index": 0, "watts": 20.0}]},
               {"epoch": 2.0, "gpus": [{"index": 0, "watts": 20.0}]}]
    with pytest.raises(ValueError):
        bridge.board_energy(samples, 0.0, 5.0, {0})
