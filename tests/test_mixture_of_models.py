"""Focused contracts for the post-hoc model-mixture benchmark."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmarks/mixture_of_models.py"
SPEC = spec_from_file_location("mixture_of_models", SCRIPT)
assert SPEC and SPEC.loader
mixture = module_from_spec(SPEC)
SPEC.loader.exec_module(mixture)


def test_majority_vote_uses_declared_order_for_even_tie():
    assert mixture.majority_vote(["strong-first", "other", "other", "strong-first"]) == "strong-first"


def test_truthfulqa_accuracy_uses_only_accuracy_indicators():
    metrics = {
        "bleu_acc,none": 0.5,
        "rouge1_acc,none": 0.7,
        "rouge2_acc,none": 0.3,
        "rougeL_acc,none": 0.9,
        "rougeL_max,none": 48.0,
        "rougeL_acc_stderr,none": 0.1,
        "policy": "task routed",
    }
    assert mixture.truthfulqa_accuracy(metrics) == 0.6


def test_truthfulqa_accuracy_requires_measured_metrics():
    assert mixture.truthfulqa_accuracy({"note": "not majority-voted"}) is None
