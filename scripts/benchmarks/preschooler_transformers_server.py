#!/usr/bin/env python3
"""Local BF16 reference endpoint for small-model compatibility measurements.

Provides the OpenAI request subset used by VirtualV and basic Prometheus
counters for llmwatch. It is not vLLM; throughput rows must identify this
engine and cannot be compared as same-runtime vLLM results.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def wall_timings(tokens: int, elapsed: float) -> dict:
    """Expose total request time without claiming an isolated decode rate.

    ``predicted_per_second`` is reserved by the private runner for genuine
    server decode timing. Model.generate here includes prefill in ``elapsed``.
    """
    return {"predicted_n": tokens,
            "wall_seconds_including_prefill": elapsed,
            "wall_tokens_per_second_including_prefill": tokens / elapsed,
            "method": "wall time including prefill"}


def encode_chat(tokenizer, messages: list[dict], body: dict):
    """Apply the request's thinking profile instead of a model-card default."""
    options = body.get("chat_template_kwargs") or {}
    if not isinstance(options, dict):
        raise ValueError("chat_template_kwargs must be an object")
    enable_thinking = options.get("enable_thinking", False)
    if not isinstance(enable_thinking, bool):
        raise ValueError("enable_thinking must be a boolean")
    return tokenizer.apply_chat_template(messages, tokenize=True,
                                         add_generation_prompt=True,
                                         return_tensors="pt",
                                         enable_thinking=enable_thinking)


def validate_no_think_template(tokenizer) -> None:
    """Fail before loading weights when the fixed no-thinking profile is impossible."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Protocol check."}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    if rendered.rstrip().endswith("<think>"):
        raise ValueError("model chat template still opens <think> with enable_thinking=false; "
                         "use a separate thinking-on protocol")


def serve(model_path: Path, alias: str, host: str, port: int) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.device_count() != 1:
        raise RuntimeError("one explicitly visible GPU is required")
    configuration = json.loads((model_path / "config.json").read_text())
    is_mistral3 = configuration.get("model_type") == "mistral3"
    if is_mistral3:
        from transformers import FineGrainedFP8Config, Mistral3ForConditionalGeneration, MistralCommonBackend

        # The pinned checkpoint is FP8. Use the model card's BF16 dequant path
        # and its Tekken tokenizer so text-only requests are comparable to the
        # other BF16 reference endpoints. No image input is accepted here.
        tokenizer = MistralCommonBackend.from_pretrained(model_path, local_files_only=True)
        validate_no_think_template(tokenizer)
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_path, device_map="cuda:0", local_files_only=True,
            quantization_config=FineGrainedFP8Config(dequantize=True),
            attn_implementation="sdpa")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        validate_no_think_template(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                                                    attn_implementation="sdpa",
                                                    trust_remote_code=False, device_map="cuda:0")
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    lock = threading.Lock()
    counters = {"prompt_tokens": 0, "generation_tokens": 0, "requests": 0,
                "running": 0, "errors": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            print(format % args, flush=True)

        def reply(self, status: int, payload: dict | str, content_type: str = "application/json") -> None:
            data = (json.dumps(payload) if isinstance(payload, dict) else payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self.reply(200, {"status": "ready", "engine": "transformers-bf16-reference"})
            elif self.path == "/v1/models":
                self.reply(200, {"data": [{"id": alias}]})
            elif self.path == "/metrics":
                metric = (f"vllm:prompt_tokens_total {counters['prompt_tokens']}\n"
                          f"vllm:generation_tokens_total {counters['generation_tokens']}\n"
                          f"vllm:request_success_total {counters['requests']}\n"
                          f"vllm:num_requests_running {counters['running']}\n"
                          f"preschooler:errors_total {counters['errors']}\n")
                self.reply(200, metric, "text/plain; version=0.0.4")
            else:
                self.reply(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path not in {"/v1/chat/completions", "/v1/completions"}:
                self.reply(404, {"error": "not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 1_000_000:
                    raise ValueError("invalid request size")
                body = json.loads(self.rfile.read(size))
                if body.get("model") != alias or body.get("temperature", 0) != 0:
                    raise ValueError("model alias or deterministic decode mismatch")
                limit = int(body.get("max_tokens", 0))
                if not 1 <= limit <= 2048:
                    raise ValueError("max_tokens outside 1..2048")
                chat = self.path.endswith("chat/completions")
                if chat:
                    messages = body.get("messages")
                    if not isinstance(messages, list) or not messages:
                        raise ValueError("messages required")
                    encoding = encode_chat(tokenizer, messages, body)
                    input_ids = encoding["input_ids"] if isinstance(encoding, Mapping) else encoding
                else:
                    prompt = body.get("prompt")
                    if not isinstance(prompt, str):
                        raise ValueError("prompt required")
                    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
                input_ids = input_ids.to("cuda:0")
                prompt_tokens = input_ids.shape[-1]
                if prompt_tokens + limit > 4096:
                    raise ValueError("context exceeds 4096 tokens")
                with lock:
                    counters["running"] += 1
                    started = time.monotonic()
                    try:
                        with torch.inference_mode():
                            output = model.generate(input_ids, attention_mask=torch.ones_like(input_ids),
                                                    max_new_tokens=limit,
                                                    min_new_tokens=limit if body.get("ignore_eos") else 0,
                                                    do_sample=False,
                                                    pad_token_id=tokenizer.pad_token_id)
                        new_ids = output[0, prompt_tokens:]
                        text = tokenizer.decode(new_ids, skip_special_tokens=True)
                        elapsed = max(time.monotonic() - started, 1e-6)
                        tokens = int(new_ids.numel())
                        counters["prompt_tokens"] += int(prompt_tokens)
                        counters["generation_tokens"] += tokens
                        counters["requests"] += 1
                    finally:
                        counters["running"] -= 1
                result = {"model": alias, "usage": {"prompt_tokens": int(prompt_tokens),
                                                    "completion_tokens": tokens},
                          "timings": wall_timings(tokens, elapsed),
                          "choices": [{"message": {"role": "assistant", "content": text},
                                       "text": text, "finish_reason": "length" if tokens == limit else "stop"}]}
                self.reply(200, result)
            except Exception as exc:
                counters["errors"] += 1
                self.reply(400 if isinstance(exc, ValueError) else 500,
                           {"error": f"{type(exc).__name__}: {exc}"})
                traceback.print_exc()

    print(f"ready engine=transformers-bf16-reference model={alias} port={port}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    serve(args.model_path, args.alias, args.host, args.port)


if __name__ == "__main__":
    main()
