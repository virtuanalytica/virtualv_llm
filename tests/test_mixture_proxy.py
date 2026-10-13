import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "benchmarks"))
import mixture_proxy as mp  # noqa: E402


def _serve(handler_cls):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"


def _upstream(answer, seen, fail=False, stream_chunks=None):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            if fail:
                self.send_response(500)
                self.end_headers()
                return
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for c in stream_chunks or []:
                    self.wfile.write(c)
                return
            out = json.dumps({"choices": [{"message": {"role": "assistant", "content": answer}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
    return H


@pytest.fixture
def mom():
    seen = {"agg": [], "p1": [], "p2": []}
    agg, agg_url = _serve(_upstream("final", seen["agg"], stream_chunks=[b"data: {\"x\":1}\n\n", b"data: [DONE]\n\n"]))
    p1, p1_url = _serve(_upstream("draft one", seen["p1"]))
    p2, p2_url = _serve(_upstream("", seen["p2"], fail=True))
    handler = type("H", (mp.Handler,), {"name": "mom-test", "aggregator": mp.Member("agg", agg_url),
                                        "proposers": (mp.Member("p1", p1_url), mp.Member("p2", p2_url)),
                                        "proposer_timeout": 10.0, "pool": mp.ThreadPoolExecutor(4)})
    proxy, url = _serve(handler)
    yield url, seen
    for s in (agg, p1, p2, proxy):
        s.shutdown()


def _post(url, body):
    req = Request(f"{url}/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=20) as r:
        return r.headers, r.read()


def test_user_turn_gathers_drafts_and_reports_failing_proposers(mom):
    url, seen = mom
    tools = [{"type": "function", "function": {"name": "play_episode", "parameters": {}}}]
    headers, body = _post(url, {"messages": [{"role": "user", "content": "2+2?"}], "tools": tools})
    out = json.loads(body)
    assert out["model"] == "mom-test" and out["mom"]["proposers"] == ["p1:ok", "p2:error:HTTPError"]
    sent = seen["agg"][0]
    assert sent["tools"] == tools                                   # tools reach the aggregator only
    assert "draft one" in sent["messages"][0]["content"] and sent["messages"][-1]["content"] == "2+2?"
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert "tools" not in seen["p1"][0]


def test_tool_result_turn_skips_proposers(mom):
    url, seen = mom
    msgs = [{"role": "user", "content": "play"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "play_episode", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "{\"return\": 0.9}"}]
    _post(url, {"messages": msgs})
    assert seen["p1"] == [] and seen["agg"][0]["messages"] == msgs


def test_streaming_is_passed_through(mom):
    url, _ = mom
    headers, body = _post(url, {"messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert body == b"data: {\"x\":1}\n\ndata: [DONE]\n\n" and headers["X-MoM-Proposers"].startswith("p1:ok")


def test_proposers_see_tool_steps_as_text():
    conv = mp.as_text_conversation([
        {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "f", "arguments": "{\"a\":1}"}}]},
        {"role": "tool", "content": "42"}])
    assert conv == [{"role": "assistant", "content": "[called tools: f({\"a\":1})]"},
                    {"role": "user", "content": "[tool result]\n42"}]


def test_member_spec_parsing_and_round_robin_replicas():
    m = mp.parse_member("q=http://h:1/v1/#m")
    assert (m.name, m.base_url, m.model, m.replicas) == ("q", "http://h:1/v1", "m", ("http://h:1/v1",))
    r = mp.parse_member("q=http://a:1/v1|http://b:2/v1")
    assert [r.next_url() for _ in range(4)] == ["http://a:1/v1", "http://b:2/v1"] * 2
    with pytest.raises(Exception):
        mp.parse_member("nourl")


def test_drafts_join_the_leading_system_message_never_a_second_one():
    p = mp.with_drafts({"messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "q"}]},
                       {"a": "x"})
    assert [m["role"] for m in p["messages"]] == ["system", "user"]
    assert p["messages"][0]["content"].startswith("be brief") and "Draft from a" in p["messages"][0]["content"]
    parts = mp.with_drafts({"messages": [{"role": "system", "content": [{"type": "text", "text": "s"}]},
                                         {"role": "user", "content": "q"}]}, {"a": "x"})
    assert len(parts["messages"]) == 2 and parts["messages"][0]["content"][-1]["type"] == "text"


def _router(answers, seen, **attrs):
    """Proxy with the thinking router on; the upstream replies with answers[] in call order."""
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            reply = answers[min(len(seen), len(answers)) - 1]
            if reply is None:
                self.send_response(500)
                self.end_headers()
                return
            out = json.dumps({"choices": [{"message": {"role": "assistant", "content": reply}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
    agg, agg_url = _serve(H)
    handler = type("H", (mp.Handler,), {"name": "mom-test", "aggregator": mp.Member("agg", agg_url),
                                        "pool": mp.ThreadPoolExecutor(1), "thinking_router": True, **attrs})
    proxy, url = _serve(handler)
    return url, (agg, proxy)


def test_thinking_router_switches_thinking_on_for_calculation_and_raises_the_budget():
    seen = []
    url, servers = _router(["Calculation", "42"], seen)
    headers, raw = _post(url, {"messages": [{"role": "user", "content": "17 * 23?"}], "max_tokens": 80})
    assert headers["X-MoM-Thinking"] == "calculation:on" and json.loads(raw)["mom"]["thinking"] == "calculation:on"
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False} and seen[0]["max_tokens"] == 6
    assert seen[1]["chat_template_kwargs"] == {"enable_thinking": True} and seen[1]["max_tokens"] == 6000
    for s in servers:
        s.shutdown()


def test_thinking_router_keeps_thinking_off_for_lookup_and_leaves_the_budget():
    seen = []
    url, servers = _router(["lookup", "x"], seen)
    headers, _ = _post(url, {"messages": [{"role": "user", "content": "amount in this record?"}], "max_tokens": 80})
    assert headers["X-MoM-Thinking"] == "lookup:off"
    assert seen[1]["chat_template_kwargs"] == {"enable_thinking": False} and seen[1]["max_tokens"] == 80
    for s in servers:
        s.shutdown()


def test_thinking_router_respects_the_client_and_survives_a_failing_classifier():
    seen = []
    url, servers = _router(["never used"], seen)
    headers, _ = _post(url, {"messages": [{"role": "user", "content": "q"}],
                             "chat_template_kwargs": {"enable_thinking": True}})
    assert headers["X-MoM-Thinking"] == "client:on" and len(seen) == 1
    for s in servers:
        s.shutdown()
    seen = []
    url, servers = _router([None, "answer"], seen)
    headers, raw = _post(url, {"messages": [{"role": "user", "content": "q"}]})
    assert headers["X-MoM-Thinking"].startswith("error:") and headers["X-MoM-Thinking"].endswith(":off")
    assert json.loads(raw)["choices"][0]["message"]["content"] == "answer"
    assert seen[1]["chat_template_kwargs"] == {"enable_thinking": False}
    for s in servers:
        s.shutdown()


def test_thinking_router_is_off_by_default_and_skips_tool_result_turns(mom):
    url, seen = mom
    headers, _ = _post(url, {"messages": [{"role": "user", "content": "q"}]})
    assert headers.get("X-MoM-Thinking") is None and "chat_template_kwargs" not in seen["agg"][-1]
    seen2 = []
    url2, servers = _router(["x"], seen2)
    headers, _ = _post(url2, {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": None,
                              "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                              {"role": "tool", "tool_call_id": "1", "content": "r"}]})
    assert headers.get("X-MoM-Thinking") is None and len(seen2) == 1
    for s in servers:
        s.shutdown()


def _overflow_router(error_body, prompt_tokens):
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                out = json.dumps({"count": prompt_tokens}).encode()
                self.send_response(200 if prompt_tokens else 404)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            seen.append(body)
            if len(seen) == 1:
                reply = "code"
            elif body.get("max_tokens", 0) > 5000:  # context overflow, as vLLM reports it
                out = error_body.encode()
                self.send_response(400)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            else:
                reply = "ok"
            out = json.dumps({"choices": [{"message": {"role": "assistant", "content": reply}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
    agg, agg_url = _serve(H)
    handler = type("H", (mp.Handler,), {"name": "mom-test", "aggregator": mp.Member("agg", agg_url),
                                        "pool": mp.ThreadPoolExecutor(1), "thinking_router": True})
    proxy, url = _serve(handler)
    return url, seen, (agg, proxy)


VLLM_OVERFLOW = ('{"error":{"message":"This model\'s maximum context length is 8192 tokens. However, you requested '
                 '6000 output tokens and your prompt contains at least %d input tokens, for a total of at least 8193 tokens."}}')


def test_thinking_router_shrinks_the_budget_to_the_room_left_in_the_context():
    url, seen, servers = _overflow_router(VLLM_OVERFLOW % 2193, 4098)
    headers, raw = _post(url, {"messages": [{"role": "user", "content": "trace this"}], "max_tokens": 300})
    assert json.loads(raw)["choices"][0]["message"]["content"] == "ok"
    assert headers["X-MoM-Thinking"] == "code:on:budget-4078"
    assert [b.get("max_tokens") for b in seen] == [6, 6000, 4078]
    assert seen[2]["chat_template_kwargs"] == {"enable_thinking": True}
    for s in servers:
        s.shutdown()


def test_thinking_router_turns_thinking_off_when_no_room_is_left():
    url, seen, servers = _overflow_router(VLLM_OVERFLOW % 2193, 7800)
    headers, raw = _post(url, {"messages": [{"role": "user", "content": "trace this"}], "max_tokens": 300})
    assert headers["X-MoM-Thinking"] == "code:on:no-room-off"
    assert seen[2]["max_tokens"] == 300 and seen[2]["chat_template_kwargs"] == {"enable_thinking": False}
    for s in servers:
        s.shutdown()


def test_thinking_router_turns_thinking_off_when_the_prompt_cannot_be_counted():
    url, seen, servers = _overflow_router(VLLM_OVERFLOW % 2193, None)
    headers, _ = _post(url, {"messages": [{"role": "user", "content": "trace this"}], "max_tokens": 300})
    assert headers["X-MoM-Thinking"] == "code:on:no-room-off" and seen[2]["chat_template_kwargs"] == {"enable_thinking": False}
    for s in servers:
        s.shutdown()
