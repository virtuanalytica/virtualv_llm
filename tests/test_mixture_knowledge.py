"""Contract tests for exact answers, evidence context and labelled specialists."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "benchmarks"))
import knowledge_layer as kl  # noqa: E402
import mixture_proxy as mp  # noqa: E402
import prepare_public_knowledge as public_stage  # noqa: E402
import summarize_mom_events as mom_events  # noqa: E402
import import_knowledge_sources as knowledge_import  # noqa: E402
import lens_bridge  # noqa: E402
import eight_task_external as eight_task  # noqa: E402


def _server(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _model(seen, answer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            data = json.dumps({"choices": [{"message": {"role": "assistant", "content": answer}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    return Handler


def _post(url, question, task="", stream=False):
    body = {"model": "mom-test", "messages": [{"role": "user", "content": question}], "stream": stream}
    req = Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json", "X-MoM-Task": task})
    with urlopen(req, timeout=10) as response:
        return response.read()


@pytest.fixture
def stack():
    agg_seen, kimi_seen, jev_seen = [], [], []
    agg, agg_url = _server(_model(agg_seen, "aggregate"))
    kimi, kimi_url = _server(_model(kimi_seen, "kimi draft"))

    class Jev(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            jev_seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            data = json.dumps({"model": "local-jev", "answers": {
                "supported": {"type": "noul", "noul": 0.7}},
                "usage": {"input_tokens": 4, "output_tokens": 1}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    jev, jev_url = _server(Jev)
    records = [
        {"id": "f1", "kind": "fact", "query": "What is the port?", "answer": "8030",
         "text": "The MoM proxy listens on port 8030.", "source": "https://example.org/ports",
         "publish_allowed": True, "license": "CC0"},
        {"id": "c1", "kind": "code", "text": "The parse_member function parses replica URLs.",
         "source": "git://virtualv_llm", "publish_allowed": False,
         "repo": "virtualv_llm", "commit": "abc123", "path": "scripts/benchmarks/mixture_proxy.py", "line": 70},
    ]
    bundle = kl.build(records, bundle_id="test", scope="private")
    handler = type("Mom", (mp.Handler,), {
        "name": "mom-test", "aggregator": mp.Member("agg", agg_url + "/v1"),
        "proposers": (), "specialist": mp.Member("kimi", kimi_url + "/v1"),
        "specialist_tasks": frozenset({"coding"}), "knowledge_bundle": bundle,
        "jev_url": jev_url + "/v1/systemone", "jev_key": "test",
        "pool": mp.ThreadPoolExecutor(2)})
    mom, mom_url = _server(handler)
    yield mom_url, agg_seen, kimi_seen, jev_seen, bundle
    for srv in (mom, agg, kimi, jev):
        srv.shutdown()


def test_exact_fact_bypasses_models_and_reports_zero_model_tokens(stack):
    url, agg, kimi, jev, _ = stack
    result = json.loads(_post(url, "What is the port?", "fact"))
    assert result["choices"][0]["message"]["content"] == "8030"
    assert result["mom"]["path"] == "knowledge_direct"
    assert result["usage"]["completion_tokens"] == 0
    assert not agg and not kimi and not jev


def test_coding_graph_context_and_specialist_are_labelled(stack):
    url, agg, kimi, jev, _ = stack
    result = json.loads(_post(url, "How does parse_member parse replica URLs?", "coding"))
    assert result["mom"]["path"] == "model"
    assert result["mom"]["knowledge_ids"] == ["c1"]
    assert result["mom"]["jev"]["model"] == "local-jev"
    assert "kimi draft" in str(agg[0]["messages"])
    assert "scripts/benchmarks/mixture_proxy.py:70" in str(agg[0]["messages"])
    assert len(kimi) == len(jev) == 1


def test_unlabelled_request_does_not_invoke_kimi(stack):
    url, agg, kimi, jev, _ = stack
    result = json.loads(_post(url, "Please explain parse_member"))
    assert result["mom"]["task"] == ""
    assert len(agg) == 1 and not kimi
    assert result["mom"]["jev"]["mode"] == "fact_rule_classifier"
    assert "deterministic fact or rule" in jev[0]["questions"]["supported"]["instructions"]


def test_public_export_filters_private_code_and_verifies_digest(tmp_path, stack):
    _, _, _, _, bundle = stack
    public = kl.build([row for row in bundle["records"] if row["publish_allowed"]],
                      bundle_id="test-public", scope="public")
    path = tmp_path / "public.json"
    path.write_bytes(kl.canonical(public))
    assert [row["id"] for row in kl.load(path)["records"]] == ["f1"]
    public["records"][0]["text"] = "tampered"
    path.write_bytes(kl.canonical(public))
    with pytest.raises(ValueError, match="digest"):
        kl.load(path)


def test_public_stage_has_matching_lens_field_and_knitweb_records(tmp_path, stack):
    _, _, _, _, bundle = stack
    source = tmp_path / "private.json"
    source.write_bytes(kl.canonical(bundle))
    manifest = public_stage.stage(source, tmp_path / "stage")
    assert manifest["records"] == 1 and manifest["published"] is False
    lens = json.loads((tmp_path / "stage/lens_rows.json").read_text())
    field = json.loads((tmp_path / "stage/field_snapshot.json").read_text())
    knitweb = json.loads((tmp_path / "stage/knitweb_asset.json").read_text())
    assert lens["rows"][0]["id"] == field["records"][0]["id"] == "f1"
    assert {row["predicate"] for row in knitweb["@graph"]} == {"hasText", "hasSource"}
    assert "c1" not in str(lens) + str(field) + str(knitweb)


def test_public_record_rejects_private_path_marker():
    row = {"id": "bad", "kind": "fact", "text": "Read /home/alice/.ssh/id_rsa",
           "source": "https://example.org", "license": "CC0", "publish_allowed": True}
    with pytest.raises(ValueError, match="private marker"):
        kl.build([row], bundle_id="bad", scope="public")


def test_query_task_label_routes_coding_specialist(stack):
    url, _, kimi, _, _ = stack
    body = {"model": "mom-test", "messages": [{"role": "user", "content":
            "How does parse_member parse replica URLs?"}]}
    req = Request(url + "/v1/chat/completions?mom_task=coding",
                  data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=10) as response:
        result = json.load(response)
    assert result["mom"]["task"] == "coding"
    assert len(kimi) == 1


def test_exact_answer_streams_without_model(stack):
    url, agg, kimi, _, _ = stack
    chunks = _post(url, "What is the port?", "fact", stream=True).decode()
    assert "8030" in chunks and "data: [DONE]" in chunks
    assert not agg and not kimi


def test_mom_system_metrics_keep_model_rate_unset():
    report = mom_events.summarize([
        {"task": "fact", "path": "knowledge_direct", "wall_seconds": .02,
         "model_output_tokens": 0},
        {"task": "coding", "path": "model", "wall_seconds": 2.1,
         "model_output_tokens": 20},
    ])
    assert report["knowledge_direct_share"] == .5
    assert report["model_tokens_per_second"] is None
    assert report["request_latency_seconds_p95"] == 2.1


def test_gitnexus_symbol_without_doc_imports_privately(tmp_path):
    source = tmp_path / "symbols.jsonl"
    source.write_text(json.dumps({"source_path": "scripts/benchmarks/example.py", "name": "fn",
                                  "kind": "function", "line": 4, "doc": ""}) + "\n")
    rows = knowledge_import.import_gitnexus_symbols(source, "https://example.org/repo", "abc")
    assert len(rows) == 1 and rows[0]["publish_allowed"] is False
    assert rows[0]["line"] == 4


def test_lens_bridge_ranks_frozen_rows_and_preserves_task_boundary(tmp_path, stack):
    _, _, _, _, bundle = stack
    rows = tmp_path / "lens-rows.json"
    rows.write_bytes(kl.canonical(lens_bridge.rows_for_bundle(bundle)))
    root = Path("/media/knight2/EDS2/projects/radicle-knitweb/lens")
    if not (root / "src/knitweb_lens").exists():
        pytest.skip("local Lens checkout absent")
    bridge = lens_bridge.LensBridge(bundle, rows, root)
    assert [hit.record["id"] for hit in bridge.retrieve("parse_member replica URLs", "coding")] == ["c1"]
    assert bridge.retrieve("parse_member replica URLs", "math") == []
    assert [hit.record["id"] for hit in bridge.retrieve("parse_member replica URLs", "humaneval")] == ["c1"]
    tampered = lens_bridge.rows_for_bundle(bundle)
    tampered["rows"][0]["text"] = "changed"
    rows.write_bytes(kl.canonical(tampered))
    with pytest.raises(ValueError, match="differ"):
        lens_bridge.LensBridge(bundle, rows, root)


def test_eight_task_labels_select_only_relevant_kimi_requests():
    assert eight_task.mom_task_label("code_exec_drawdown") == "coding"
    assert eight_task.mom_task_label("needle_in_haystack") == "long_context"
    assert eight_task.mom_task_label("gics_format_following") == "benchmark"
