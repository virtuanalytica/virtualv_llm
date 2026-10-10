#!/usr/bin/env python3
"""Generate and measure a fresh public development pack for Toddler's MoM search.

These are procedural, answer-keyed research items created after local weights
already exist. They never enter a private promotion score or the retired suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import secrets
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

TASKS = ("arithmetic", "code_trace", "structured_extraction")
DECODE = {"temperature": 0, "max_tokens": 80, "parallel": 1,
          "response_format": {"type": "json_object"}}
SCHEMA = "toddler-mom-public-dev/v1"


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode()


def ordered_id_hash(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_pack(seed: int, n: int = 30) -> dict:
    if n < 30:
        raise ValueError("at least 30 items per task are required")
    rng = random.Random(seed)
    rows = []
    for task in TASKS:
        for index in range(n):
            nonce = f"{rng.getrandbits(96):024x}"
            if task == "arithmetic":
                if index % 3 == 0:
                    a, b = rng.randrange(100, 1000), rng.randrange(100, 1000)
                    answer, body = str(a + b), f"Compute {a} + {b}."
                elif index % 3 == 1:
                    a, b = rng.randrange(12, 90), rng.randrange(12, 90)
                    answer, body = str(a * b), f"Compute {a} * {b}."
                else:
                    a, b, c = rng.randrange(100, 1000), rng.randrange(20, 100), rng.randrange(3, 20)
                    answer = str((a * c + b) % 997)
                    body = f"Compute (({a} * {c}) + {b}) mod 997."
            elif task == "code_trace":
                if index % 3 == 0:
                    start, count = rng.randrange(2, 15), rng.randrange(4, 10)
                    answer = str(sum(range(start, start + count)))
                    body = f"Evaluate this Python expression: sum(range({start}, {start + count}))."
                elif index % 3 == 1:
                    start, step, count = rng.randrange(2, 15), rng.randrange(2, 8), rng.randrange(3, 7)
                    answer = str(sum(start + step * k for k in range(count)))
                    body = ("Evaluate this Python expression: "
                            f"sum({start} + {step}*k for k in range({count})).")
                else:
                    start, step, count = rng.randrange(10, 100), rng.randrange(2, 12), rng.randrange(5, 13)
                    shift = rng.randrange(3, 30)
                    answer = str(sum((start + step * k + shift) % 17 for k in range(count)))
                    body = ("Evaluate this Python expression: "
                            f"sum(({start} + {step}*k + {shift}) % 17 for k in range({count})).")
            else:
                account = f"AC{rng.randrange(100000, 999999)}"
                invoice = f"IV{rng.randrange(100000, 999999)}"
                amount = rng.randrange(1000, 99999)
                answer = str(amount)
                if index % 3 == 0:
                    body = (f"Record: invoice={invoice}; account={account}; amount_eur_cents={amount}; "
                            f"status=approved. Give only amount_eur_cents.")
                elif index % 3 == 1:
                    distractor = rng.randrange(1000, 99999)
                    body = (f"Draft amount_eur_cents={distractor} was superseded. Approved invoice={invoice}, "
                            f"account={account}, final amount_eur_cents={amount}. Give the final cents.")
                else:
                    body = (f"Invoice {invoice} has account {account}. The approved cents amount is {amount}; "
                            "the account and invoice numbers are identifiers, not money. Give approved cents.")
            item_id = f"{task}-{index:03d}-{nonce}"
            rows.append({"id": item_id, "task": task,
                         "prompt": f"Research item {nonce}. {body} Return only JSON with one integer field named answer. No explanation. /no_think",
                         "answer": answer})
    return {"schema": "toddler-mom-public-pack/v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "generation": "Python random.Random with a newly sampled 128-bit seed; no model generated the pack",
            "seed": seed, "tasks": list(TASKS), "items": rows}


def _gpu_row(index: int) -> tuple[float, float]:
    proc = subprocess.run(["nvidia-smi", "--id", str(index),
                           "--query-gpu=power.draw,memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True, timeout=10)
    watts, mib = (float(part.strip()) for part in proc.stdout.strip().split(","))
    return watts, mib / 1024


class BoardSampler:
    def __init__(self, gpu_index: int):
        self.gpu_index = gpu_index
        self.rows: list[tuple[float, float]] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.error: Exception | None = None

    def _sample(self) -> None:
        while not self.stop.is_set():
            try:
                watts, _ = _gpu_row(self.gpu_index)
                self.rows.append((time.monotonic(), watts))
            except Exception as exc:  # fail the measurement, not just the sampler thread
                self.error = exc
                break
            self.stop.wait(0.4)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=15)

    def wh(self) -> float:
        if self.error:
            raise RuntimeError(f"GPU board sampling failed: {self.error}")
        if len(self.rows) < 2:
            raise RuntimeError("too few GPU board power samples")
        joules = sum((b[0] - a[0]) * (a[1] + b[1]) / 2
                      for a, b in zip(self.rows, self.rows[1:]))
        return joules / 3600


def request(base: str, path: str, payload: dict, timeout: int = 180) -> dict:
    req = Request(base.rstrip("/") + path, data=canonical(payload),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def decode_probe(base: str) -> float:
    response = request(base, "/completion", {
        "prompt": "Write a continuous explanation of reproducible software benchmarks in complete sentences. "
                  "Continue until the token budget ends.",
        "n_predict": 256, "temperature": 0, "ignore_eos": True})
    timings = response.get("timings", {})
    tps = timings.get("predicted_per_second")
    if not isinstance(tps, (float, int)) or tps <= 0 or timings.get("predicted_n", 0) < 240:
        raise RuntimeError("256-token decode probe did not return usable llama.cpp timings")
    return float(tps)


def integer_answer(content: str) -> str | None:
    try:
        value = json.loads(content)["answer"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return str(value) if type(value) is int else None


def measure(pack_path: Path, model: str, weights: Path, base: str, gpu: int) -> dict:
    pack_raw = pack_path.read_bytes()
    pack = json.loads(pack_raw)
    if pack.get("schema") != "toddler-mom-public-pack/v1" or set(pack["tasks"]) != set(TASKS):
        raise ValueError("wrong public development pack")
    created = datetime.fromisoformat(pack["created_utc"])
    if weights.stat().st_mtime > created.timestamp():
        raise ValueError("model weight file was modified after public prompts were generated")
    memory = _gpu_row(gpu)[1]
    probe_tps = decode_probe(base)
    results = {}
    for task in TASKS:
        items = [row for row in pack["items"] if row["task"] == task]
        if len(items) < 30 or len({row["id"] for row in items}) != len(items):
            raise ValueError(f"{task} pack is incomplete or duplicated")
        scores = []
        evidence = []
        started = time.monotonic()
        with BoardSampler(gpu) as sampler:
            for item in items:
                response = request(base, "/v1/chat/completions", {
                    "model": "x", "messages": [{"role": "user", "content": item["prompt"]}], **DECODE})
                content = response["choices"][0]["message"]["content"]
                parsed = integer_answer(content)
                score = float(parsed == item["answer"])
                scores.append(score)
                evidence.append({"id": item["id"], "response": content, "parsed": parsed,
                                 "finish_reason": response["choices"][0].get("finish_reason"),
                                 "score": score})
        seconds = time.monotonic() - started
        ids = [row["id"] for row in items]
        results[task] = {"n": len(items), "item_ids": ids,
                         "item_ids_sha256": ordered_id_hash(ids), "item_scores": scores,
                         "evidence": evidence,
                         "quality": sum(scores) / len(scores),
                         "latency_s": seconds / len(scores),
                         "decode_tps": probe_tps,
                         "gpu_board_wh_per_answer": sampler.wh() / len(scores)}
    return {"model": model, "weights_sha256": sha256_file(weights),
            "weight_path": str(weights.resolve()), "access": "local",
            "energy_scope": "gpu_board", "physical_gpu": gpu,
            "resident_vram_gb": memory, "tasks": results,
            "pack_sha256": hashlib.sha256(pack_raw).hexdigest(),
            "decode_profile_sha256": hashlib.sha256(canonical(DECODE)).hexdigest(),
            "measured_utc": datetime.now(timezone.utc).isoformat()}


def assemble(pack_path: Path, measurements: list[Path]) -> dict:
    pack_raw = pack_path.read_bytes()
    pack = json.loads(pack_raw)
    pack_hash = hashlib.sha256(pack_raw).hexdigest()
    models = []
    for path in measurements:
        raw = path.read_bytes()
        row = json.loads(raw)
        row["source_measurement_sha256"] = hashlib.sha256(raw).hexdigest()
        models.append(row)
    if len(models) < 2 or len({row["model"] for row in models}) != len(models):
        raise ValueError("need at least two distinct local model measurements")
    for row in models:
        if row["pack_sha256"] != pack_hash or row["decode_profile_sha256"] != hashlib.sha256(canonical(DECODE)).hexdigest():
            raise ValueError("different public pack or decode profile")
        if set(row["tasks"]) != set(TASKS):
            raise ValueError("a model is missing a measured task")
        for task in TASKS:
            expected = [item["id"] for item in pack["items"] if item["task"] == task]
            measured = row["tasks"][task]
            if (measured["item_ids"] != expected or measured["item_ids_sha256"] != ordered_id_hash(expected)
                    or measured["n"] != len(expected) or len(measured["item_scores"]) != len(expected)):
                raise ValueError(f"{row['model']}/{task} lacks paired public item scores")
            public_items = [item for item in pack["items"] if item["task"] == task]
            evidence = measured.get("evidence")
            if not isinstance(evidence, list) or len(evidence) != len(expected):
                raise ValueError(f"{row['model']}/{task} lacks response evidence")
            for item, response, score in zip(public_items, evidence, measured["item_scores"]):
                parsed = integer_answer(response["response"])
                if (response["id"] != item["id"] or response["parsed"] != parsed
                        or response["score"] != score or score != float(parsed == item["answer"])):
                    raise ValueError(f"{row['model']}/{task} score differs from its response evidence")
            measured.pop("evidence")
        row.pop("weight_path", None)
        row.pop("physical_gpu", None)
        row.pop("pack_sha256", None)
        row.pop("decode_profile_sha256", None)
        row.pop("measured_utc", None)
    prompts = [row["prompt"] for row in pack["items"]]
    return {"schema": SCHEMA, "split": "public_development",
            "benchmark_policy": "anti_contamination_public_development",
            "promotion_eligible": False, "training_overlap_check": "passed",
            "training_overlap_method": "fresh random 96-bit item nonce and numeric inputs, generated after local weight files; exact prompts only",
            "item_bank_sha256": pack_hash,
            "prompt_pack_sha256": hashlib.sha256(canonical(prompts)).hexdigest(),
            "decode_profile_sha256": hashlib.sha256(canonical(DECODE)).hexdigest(),
            "tasks": list(TASKS), "models": models,
            "warning": "public development only; exact-prompt novelty does not prove generalization or absence of template contamination"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--items-per-task", type=int, default=30)
    run = sub.add_parser("measure")
    run.add_argument("--pack", type=Path, required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--weights", type=Path, required=True)
    run.add_argument("--base-url", required=True)
    run.add_argument("--gpu", type=int, required=True)
    run.add_argument("--out", type=Path, required=True)
    combine = sub.add_parser("assemble")
    combine.add_argument("--pack", type=Path, required=True)
    combine.add_argument("--measurement", type=Path, action="append", required=True)
    combine.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        result = make_pack(secrets.randbits(128), args.items_per_task)
    elif args.command == "measure":
        result = measure(args.pack, args.model, args.weights, args.base_url, args.gpu)
    else:
        result = assemble(args.pack, args.measurement)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
