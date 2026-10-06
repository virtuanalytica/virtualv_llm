"""Regression checks for the repository-owned deterministic fixtures."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmarks/eval_suite.py"
SPEC = spec_from_file_location("eval_suite", SCRIPT)
assert SPEC and SPEC.loader
suite = module_from_spec(SPEC)
SPEC.loader.exec_module(suite)


def test_fixture_prompts_keep_declared_gold_facts():
    assert "2011-04-08" in suite.NEEDLE_HAYSTACK_PROMPT
    assert "13,350" in suite.SUMMARY_FAITHFULNESS_PROMPT
    assert "21,535" in suite.SUMMARY_FAITHFULNESS_PROMPT
    assert "62%" in suite.SUMMARY_FAITHFULNESS_PROMPT
    assert "Tiingo" in suite.SUMMARY_FAITHFULNESS_PROMPT


def test_suite_has_stable_unique_task_ids():
    task_ids = [task.id for task in suite.TASKS]
    assert len(task_ids) == 8
    assert len(task_ids) == len(set(task_ids))


def test_task_aware_completion_preserves_shared_scoring(monkeypatch):
    one = suite.Task("probe", "answer 4", 5, lambda raw: (1.0 if raw.strip() == "4" else 0.0, {}))
    monkeypatch.setattr(suite, "TASKS", [one])
    seen = []

    def task_completion(task):
        seen.append(task.id)
        return {"choices": [{"message": {"content": "4"}}]}

    result = suite.run_tasks(lambda *_: {}, complete_task_fn=task_completion)
    assert seen == ["probe"]
    assert result["_mean_score"] == 1.0
