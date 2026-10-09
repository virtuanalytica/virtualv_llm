"""Publishing aggregate scores must never copy private items or answers."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/benchmarks"))
from redact_private_report import redact  # noqa: E402


def test_private_prompts_answers_and_samples_are_removed():
    secret = "PRIVATE ANSWER TOKEN"
    source = {"protocol": "private", "results": [{
        "model": "candidate", "status": "complete", "eight_task_mean": 0.5,
        "specialist": {"value": 0.6, "samples": [{"answer": secret}]},
        "anti_contamination": {"post_cutoff_holdout": {
            "status": "complete", "accuracy": 0.7, "pack_sha256": "a" * 64,
            "samples": [{"prompt": secret, "target": secret, "prediction": secret}]}},
        "throughput_probe": {"completion_tokens": 256, "completion_tokens_per_second": 50,
                             "prompt": secret},
        "canary": {"status": "failed", "observations": [
            {"reason": "repetition_loop", "passed": False, "answer": secret}]},
        "requests": 140, "log_dir": secret,
    }]}
    result = redact(source)
    encoded = json.dumps(result)
    assert secret not in encoded
    assert "samples" not in result["results"][0]["anti_contamination"]["post_cutoff_holdout"]
    assert result["results"][0]["canary"] == {
        "status": "failed", "passed": 0, "total": 1, "repetition_loops": 1}
    assert result["results"][0]["throughput_probe"]["completion_tokens"] == 256
