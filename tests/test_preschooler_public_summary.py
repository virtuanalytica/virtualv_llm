import json

import pytest

from scripts.benchmarks.preschooler_public_summary import build


def _profile(root, alias, revision, pack="pack-a", status="complete_private"):
    folder = root / f"{alias}-private"
    private = folder / "private"
    private.mkdir(parents=True)
    (folder / "status.json").write_text(json.dumps({
        "status": status, "alias": alias, "revision": revision,
        "canary": {"status": "passed", "prompt": "PRIVATE_SENTINEL"},
        "engine": "transformers", "gpu_index": 0, "finished_at": "2026-10-11T00:00:00Z",
    }))
    row = {"model": alias, "protocol": "sealed-v1", "status": "complete",
           "promotion_eligible": False, "eight_task_mean": .75,
           "specialist": {"value": .4, "ci95": [.2, .6], "items": 44,
                          "lanes": 10, "complete": True},
           "throughput_probe": {"completion_tokens": 256,
                                "completion_tokens_per_second": 25.5,
                                "timing_source": "server decode"},
           "gpu_board_wh_per_answer": .05, "gpu_board_wh": 7.0,
           "requests": 140, "private_prompt": "PRIVATE_SENTINEL",
           "provenance": {"specialist_pack_sha256": {"chemistry": pack},
                          "holdout_pack_sha256": "holdout-a"},
           "anti_contamination": {
               "canary_recall": {"recall_rate": 0, "samples": ["PRIVATE_SENTINEL"]},
               "paraphrase_invariance": {"original_accuracy": .8,
                                         "paraphrase_accuracy": .7},
               "post_cutoff_holdout": {"accuracy": .9, "answer": "PRIVATE_SENTINEL"},
           }}
    (private / "private-result.json").write_text(json.dumps({
        "suite": "private", "results": [row], "private_key": "PRIVATE_SENTINEL"}))


def test_public_summary_exports_only_scalars_and_checks_pack_match(tmp_path):
    roster = {"models": [{"alias": "qwen", "revision": "sha-q", "family": "qwen"},
                         {"alias": "phi", "revision": "sha-p", "family": "phi"},
                         {"alias": "invalid", "revision": "sha-i", "family": "smol"}]}
    _profile(tmp_path, "qwen", "sha-q")
    _profile(tmp_path, "phi", "sha-p")
    _profile(tmp_path, "invalid", "sha-i", status="invalid_protocol")
    report = build(tmp_path, roster)
    assert report["complete_profiles"] == 2
    assert report["rows"][0]["throughput_scope"] == "request wall time including prefill"
    assert "PRIVATE_SENTINEL" not in json.dumps(report)
    _profile_dir = tmp_path / "phi-private" / "private" / "private-result.json"
    raw = json.loads(_profile_dir.read_text())
    raw["results"][0]["provenance"]["specialist_pack_sha256"]["chemistry"] = "changed"
    _profile_dir.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="different specialist"):
        build(tmp_path, roster)


def test_explicit_runtime_folder_exports_only_safe_compatibility_fields(tmp_path):
    _profile(tmp_path, "ministral", "sha-m")
    source = tmp_path / "ministral-private"
    target = tmp_path / "ministral-bf16-private"
    source.rename(target)
    path = target / "status.json"
    status = json.loads(path.read_text())
    status["runtime_compat"] = {"transformers": "5.16.1", "accelerate": "1.12.0",
                                "mode": "BF16 dequantized text-only",
                                "overlay": "/private/PRIVATE_SENTINEL"}
    path.write_text(json.dumps(status))
    model = {"alias": "ministral", "revision": "sha-m", "family": "mistral",
             "private_run_folder": "ministral-bf16-private"}
    report = build(tmp_path, {"models": [model]})
    assert report["complete_profiles"] == 1
    assert report["rows"][0]["runtime_compat"]["transformers"] == "5.16.1"
    assert "PRIVATE_SENTINEL" not in json.dumps(report)
    model["private_run_folder"] = "../elsewhere"
    with pytest.raises(ValueError, match="basename"):
        build(tmp_path, {"models": [model]})
