#!/usr/bin/env python3
"""Live mixture-of-models (MoM) as one OpenAI-compatible chat endpoint.

mixture_of_models.py scores a mixture offline, by voting over answers each model already gave in a
benchmark run. A chat (the OMP harness, virtualpc) needs the mixture live, with tool calls and
streaming. This proxy implements Mixture-of-Agents (Wang et al. 2024, arXiv:2406.04692):

  1. proposers  every member except the aggregator answers the conversation in parallel
                (text only: tool calls and tool results are shown to them as text, no tools offered)
  2. aggregator receives the ORIGINAL request (tools, streaming, sampling) plus one system message
                with the proposers' drafts, and writes the final answer / tool calls, streamed back

Proposals are only gathered when the last message is from the user (a new question). A turn that
continues after a tool result goes straight to the aggregator: the drafts of that question are
already in its context, and re-proposing on every tool step would multiply latency for no gain.
A proposer that fails or times out is dropped from that turn and reported in the response header
`X-MoM-Proposers` (name:ok|error); the aggregator always answers, so the chat never stalls on one
member. The same path serves benchmark requests, so a well_known_suite.py score
(`--external-url http://127.0.0.1:<port>/v1 --external-model <name>`) measures exactly the mixture
the chat uses.

    python3 scripts/benchmarks/mixture_proxy.py --port 8030 --name mom-live-4 \
        --aggregator qwen38=http://127.0.0.1:8021/v1 \
        --proposer devstral=http://127.0.0.1:8022/v1 --proposer qwen35=http://127.0.0.1:8023/v1 \
        --proposer gemma4=http://127.0.0.1:8024/v1
"""
from __future__ import annotations

import argparse
import itertools
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DRAFT_HEADER = (
    "You are the aggregator of a mixture of models. Below are draft answers from other models to the "
    "user's latest message. They may be wrong or incomplete. Use them as evidence: keep what is correct, "
    "fix what is wrong, and give one final answer in your own words. Use your tools when the task needs "
    "them; the drafts had no tool access. Do not mention the drafts unless asked."
)


@dataclass(frozen=True)
class Member:
    """One mixture member, served by one or more identical replicas (round-robin per request), so a
    slow member can be spread over several GPUs without changing the mixture itself."""
    name: str
    base_url: str          # first replica, ends in /v1
    model: str = ""        # model id sent upstream; "" = the server's default
    replicas: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "_rr", itertools.cycle(self.replicas or (self.base_url,)))
        object.__setattr__(self, "_lock", threading.Lock())

    def next_url(self) -> str:
        with self._lock:
            return next(self._rr)


def parse_member(spec: str) -> Member:
    """name=http://host:port/v1[|http://host2:port/v1...][#model-id]"""
    name, _, rest = spec.partition("=")
    urls, _, model = rest.partition("#")
    replicas = tuple(u.rstrip("/") for u in urls.split("|") if u)
    if not name or not replicas or not all(u.startswith("http") for u in replicas):
        raise argparse.ArgumentTypeError(f"member must look like name=http://host:port/v1[|...][#model], got {spec!r}")
    return Member(name, replicas[0], model, replicas)


def as_text_conversation(messages: list[dict]) -> list[dict]:
    """Conversation for proposers: no tool roles or tool_calls, which they cannot act on."""
    out = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if isinstance(content, list):         # multi-part content: keep the text parts
            content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
        content = content or ""
        if role == "tool":
            out.append({"role": "user", "content": f"[tool result]\n{content}"})
        elif role == "assistant" and m.get("tool_calls"):
            calls = "; ".join(f"{c.get('function', {}).get('name')}({c.get('function', {}).get('arguments', '')})"
                              for c in m["tool_calls"])
            out.append({"role": "assistant", "content": f"{content}\n[called tools: {calls}]".strip()})
        elif role in ("system", "developer"):
            out.append({"role": "system", "content": content})
        else:
            out.append({"role": role or "user", "content": content})
    return out


def needs_proposals(messages: list[dict]) -> bool:
    return bool(messages) and messages[-1].get("role") == "user"


def with_drafts(payload: dict, drafts: dict[str, str]) -> dict:
    if not drafts:
        return payload
    body = "\n\n".join(f"### Draft from {name}\n{text}" for name, text in drafts.items())
    note = f"{DRAFT_HEADER}\n\n{body}"
    msgs = list(payload.get("messages", []))
    # Chat templates (Qwen, Gemma) reject a system message that is not the first message, so the
    # drafts are appended to the leading system message, or become the leading one.
    if msgs and msgs[0].get("role") in ("system", "developer"):
        first = dict(msgs[0])
        c = first.get("content")
        if isinstance(c, list):
            first["content"] = [*c, {"type": "text", "text": note}]
        else:
            first["content"] = f"{c or ''}\n\n{note}".strip()
        msgs[0] = first
    else:
        msgs.insert(0, {"role": "system", "content": note})
    return {**payload, "messages": msgs}


def _post(url: str, payload: dict, timeout: float) -> bytes:
    req = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def propose(member: Member, messages: list[dict], max_tokens: int, timeout: float) -> str:
    payload = {"messages": as_text_conversation(messages), "max_tokens": max_tokens, "temperature": 0.3,
               "stream": False}
    if member.model:
        payload["model"] = member.model
    data = json.loads(_post(f"{member.next_url()}/chat/completions", payload, timeout))
    return (data["choices"][0]["message"].get("content") or "").strip()


class Handler(BaseHTTPRequestHandler):
    name: str = "mom-live"
    aggregator: Member
    proposers: tuple[Member, ...] = ()
    draft_tokens: int = 768
    proposer_timeout: float = 180.0
    pool: ThreadPoolExecutor

    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/").endswith("/models"):
            members = [self.aggregator.name, *(p.name for p in self.proposers)]
            self._send_json(200, {"object": "list", "data": [{"id": self.name, "object": "model",
                                                              "owned_by": "virtualv_llm", "members": members}]})
        elif self.path.rstrip("/").endswith("/health"):
            self._send_json(200, {"ok": True})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send_json(404, {"error": "only /v1/chat/completions is served"})
            return
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"invalid JSON body: {exc}"})
            return
        messages = payload.get("messages") or []
        drafts, status = {}, []
        if self.proposers and needs_proposals(messages):
            futures = {p.name: self.pool.submit(propose, p, messages, self.draft_tokens, self.proposer_timeout)
                       for p in self.proposers}
            for name, fut in futures.items():
                try:
                    text = fut.result(timeout=self.proposer_timeout + 5)
                    if text:
                        drafts[name] = text
                    status.append(f"{name}:ok")
                except Exception as exc:  # a failing member is dropped for this turn, reported in the header
                    status.append(f"{name}:error:{type(exc).__name__}")
        upstream = with_drafts(payload, drafts)
        upstream["model"] = self.aggregator.model or upstream.get("model", "")
        req = Request(f"{self.aggregator.next_url()}/chat/completions", data=json.dumps(upstream).encode(),
                      headers={"Content-Type": "application/json"})
        try:
            resp = urlopen(req, timeout=900)
        except HTTPError as exc:
            self._send_raw(exc.code, exc.read(), "application/json", status)
            return
        except URLError as exc:
            self._send_json(502, {"error": f"aggregator {self.aggregator.name} unreachable: {exc.reason}"}, status)
            return
        with resp:
            ctype = resp.headers.get("Content-Type", "application/json")
            if payload.get("stream"):
                self.send_response(resp.status)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-MoM-Proposers", ",".join(status) or "none")
                self.end_headers()
                while chunk := resp.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            else:
                body = json.loads(resp.read())
                body["model"] = self.name
                body.setdefault("mom", {})["proposers"] = status
                self._send_raw(resp.status, json.dumps(body).encode(), "application/json", status)

    def _send_json(self, code: int, payload: dict, status: list[str] | None = None) -> None:
        self._send_raw(code, json.dumps(payload).encode(), "application/json", status or [])

    def _send_raw(self, code: int, body: bytes, ctype: str, status: list[str]) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if status:
            self.send_header("X-MoM-Proposers", ",".join(status))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--name", default="mom-live")
    ap.add_argument("--aggregator", type=parse_member, required=True)
    ap.add_argument("--proposer", type=parse_member, action="append", default=[])
    ap.add_argument("--draft-tokens", type=int, default=768)
    ap.add_argument("--proposer-timeout", type=float, default=180.0)
    a = ap.parse_args()
    handler = type("BoundHandler", (Handler,), {
        "name": a.name, "aggregator": a.aggregator, "proposers": tuple(a.proposer), "draft_tokens": a.draft_tokens,
        "proposer_timeout": a.proposer_timeout, "pool": ThreadPoolExecutor(max_workers=max(1, 4 * len(a.proposer)))})
    server = ThreadingHTTPServer((a.host, a.port), handler)
    print(f"mixture_proxy {a.name}: aggregator={a.aggregator.name} proposers={[p.name for p in a.proposer]} "
          f"on {a.host}:{a.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
