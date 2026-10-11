import pytest

from scripts.benchmarks.preschooler_transformers_server import (encode_chat,
                                                               validate_no_think_template,
                                                               wall_timings)


def test_transformers_wall_time_cannot_masquerade_as_decode_time():
    timing = wall_timings(256, 10.0)
    assert timing["wall_tokens_per_second_including_prefill"] == 25.6
    assert timing["wall_seconds_including_prefill"] == 10.0
    assert "predicted_per_second" not in timing


def test_private_no_thinking_profile_reaches_chat_template():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "test"}]
            return kwargs

    messages = [{"role": "user", "content": "test"}]
    options = encode_chat(Tokenizer(), messages,
                          {"chat_template_kwargs": {"enable_thinking": False,
                                                    "reasoning_effort": "low"}})
    assert options["enable_thinking"] is False
    assert encode_chat(Tokenizer(), messages, {})["enable_thinking"] is False
    with pytest.raises(ValueError, match="boolean"):
        encode_chat(Tokenizer(), messages,
                    {"chat_template_kwargs": {"enable_thinking": "false"}})


def test_thinking_only_template_rejected_before_weights_load():
    class Tokenizer:
        def __init__(self, suffix):
            self.suffix = suffix

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return "<|im_start|>assistant\n" + self.suffix

    with pytest.raises(ValueError, match="thinking-on protocol"):
        validate_no_think_template(Tokenizer("<think>\n"))
    validate_no_think_template(Tokenizer("/no_think\n"))
