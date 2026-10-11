from scripts.benchmarks.preschooler_transformers_server import wall_timings


def test_transformers_wall_time_cannot_masquerade_as_decode_time():
    timing = wall_timings(256, 10.0)
    assert timing["wall_tokens_per_second_including_prefill"] == 25.6
    assert timing["wall_seconds_including_prefill"] == 10.0
    assert "predicted_per_second" not in timing
