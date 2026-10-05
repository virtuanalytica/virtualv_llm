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


def test_member_spec_parsing():
    assert mp.parse_member("q=http://h:1/v1/#m") == mp.Member("q", "http://h:1/v1", "m")
    with pytest.raises(Exception):
        mp.parse_member("nourl")
