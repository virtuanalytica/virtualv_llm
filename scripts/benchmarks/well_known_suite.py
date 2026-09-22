#!/usr/bin/env python3
"""Run the well-known public benchmark battery (GSM8K, BBH, TruthfulQA-gen, an
MMLU subject sample, and HumanEval pass@1) against local GGUF models on the
two V100s only -- same GPU-isolation contract as benchmark_local_gguf_tp2.py.

Why these tasks specifically: GSM8K and HumanEval are close to universal in
model cards/leaderboards; BBH, TruthfulQA and MMLU are the other most-cited
lm-evaluation-harness benchmarks. Multiple-choice tasks that need per-model
tokenizer-exact loglikelihood scoring (mmlu, hellaswag, arc_challenge,
winogrande, truthfulqa_mc2) are deliberately left out: lm-eval requires a
real HF tokenizer id per model to compute those, and getting that wrong
silently produces bad scores across many differently-sourced GGUF community
quants -- worse than not running them. mmlu_generative sidesteps this by
scoring the generated letter answer instead of token loglikelihoods.
HumanEval/MBPP's built-in lm-eval tasks are broken in this environment (they
import HF `evaluate`'s code_eval metric, which calls a huggingface_hub API
removed by the newer huggingface_hub installed here) -- see
humaneval_harness.py for the reimplementation used instead.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_local_gguf_tp2 import (  # noqa: E402
    MODELS, PROFILES, SERVER, gpu_sample, monitor_gpu, telemetry_summary, wait_ready,
)
from humaneval_harness import run_humaneval  # noqa: E402
from specialist_suite import SPECIALISTS, run_specialists  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
PORT = 18011
BASE_URL = f"http://127.0.0.1:{PORT}"
MODEL_ALIAS = "x"
LM_EVAL_BIN = str(Path.home() / ".local/bin/lm-eval")
EVAL_PROTOCOL = "v4-mmlu-fewshot-20260918"

MMLU_SUBJECT_SAMPLE = [
    "abstract_algebra", "anatomy", "astronomy", "college_computer_science",
    "high_school_psychology", "formal_logic", "professional_law", "marketing",
]
GSM8K_LIMIT = 50
# The full "bbh" group task is all 23 subtasks and, with long-CoT subtasks like
# multistep_arithmetic_two/dyck_languages/tracking_shuffled_objects, took >10
# minutes for a single model even at --limit 5 -- untenable across the full
# model sweep. This curated 6-subtask sample keeps a representative spread
# (logic, causal reasoning, date/temporal, navigation) while bounding runtime.
BBH_SUBTASKS = [
    "bbh_cot_fewshot_boolean_expressions", "bbh_cot_fewshot_causal_judgement",
    "bbh_cot_fewshot_date_understanding", "bbh_cot_fewshot_logical_deduction_five_objects",
    "bbh_cot_fewshot_navigate", "bbh_cot_fewshot_temporal_sequences",
]
BBH_LIMIT = 8  # per selected subtask
TRUTHFULQA_LIMIT = 30
MMLU_LIMIT_PER_SUBJECT = 20
HUMANEVAL_LIMIT = 40
PERF_PROMPT = ("Write a continuous technical explanation of point-in-time validation for financial "
               "machine learning. Use complete sentences and keep writing until the token budget ends.")


def request_json(path: str, payload: dict[str, Any] | None = None, timeout: int = 240) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode()
    req = Request(f"{BASE_URL}{path}", data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def complete_text(prompt: str, max_tokens: int, stop: list[str] | None = None) -> str:
    payload = {"model": MODEL_ALIAS, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0}
    if stop:
        payload["stop"] = stop
    resp = request_json("/v1/chat/completions", payload)
    return resp["choices"][0]["message"]["content"]


def complete_vision(prompt: str, image_path: Path, max_tokens: int) -> str:
    """OpenAI-compatible image request used only for declared VLM endpoints."""
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode()
    payload = {"model": MODEL_ALIAS, "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
    ]}], "max_tokens": max_tokens, "temperature": 0}
    resp = request_json("/v1/chat/completions", payload)
    return resp["choices"][0]["message"]["content"]


# 2026-09-19: was a hardcoded 1800s (30min) inline literal, calibrated for the
# ~90-130 tok/s fast candidates this suite was originally built around. The
# Ada+A4000 profile's heavy --n-cpu-moe offload candidates (Mistral Small 4,
# DeepSeek-V4-Flash-REAP-150b) run at 5-7 tok/s -- gsm8k alone (50 samples in
# one subprocess call, the single largest batch of any task here) blew past
# 1800s and TimeoutExpired killed the whole run after ~30 minutes of real
# progress. Module-level so a slow-hardware throwaway script can raise it
# (e.g. `wks.LM_EVAL_TIMEOUT = 7200`) without touching this function; fast
# candidates keep the same effective 1800s ceiling by default.
LM_EVAL_TIMEOUT = 1800


def run_lm_eval_task(
    task: str, limit: int, out_root: Path, log_samples: bool = False, gen_kwargs: str | None = None,
    num_fewshot: int | None = None,
) -> dict[str, Any]:
    out_dir = out_root / task
    args = [
        LM_EVAL_BIN, "--model", "local-chat-completions", "--apply_chat_template",
        "--model_args", f"model={MODEL_ALIAS},base_url={BASE_URL}/v1/chat/completions,"
                        f"num_concurrent=1,tokenized_requests=False,tokenizer_backend=None",
        "--tasks", task, "--limit", str(limit), "--output_path", str(out_dir),
    ]
    if log_samples:
        args.append("--log_samples")
    if gen_kwargs:
        args += ["--gen_kwargs", gen_kwargs]
    if num_fewshot is not None:
        args += ["--num_fewshot", str(num_fewshot)]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=LM_EVAL_TIMEOUT)
    result_files = sorted(out_dir.glob("*/results_*.json"))
    if not result_files:
        return {"task": task, "error": (proc.stderr or proc.stdout)[-800:]}
    data = json.loads(result_files[-1].read_text())
    return {"task": task, "metrics": data.get("results", {}).get(task, {}), "out_dir": out_dir}


_LETTER_RE = re.compile(r"\b([ABCD])\b")


def score_mmlu_letter_match(out_dir: Path, task: str) -> dict[str, Any] | None:
    """lm-eval's built-in exact_match filter for mmlu_generative requires the
    ENTIRE first line to equal the bare target letter -- but instruction-tuned
    chat models routinely answer "B. 4" (letter + the choice text) rather than
    a bare "B", which lm-eval's exact_match then silently scores as wrong even
    when the model picked the right choice (confirmed via --log_samples: e.g.
    doc target "B", model response "B. 4", exact_match=0). This re-scores from
    the raw logged samples by extracting the first standalone A/B/C/D token
    instead of requiring an exact whole-line match.
    """
    sample_files = sorted(out_dir.glob("*/samples_*.jsonl"))
    if not sample_files:
        return None
    correct = 0
    total = 0
    for line in sample_files[-1].read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        target = row.get("target")
        raw = (row.get("resps") or [[""]])[0][0]
        match = _LETTER_RE.search(raw.strip())
        predicted = match.group(1) if match else None
        total += 1
        correct += int(predicted == target)
    return {"accuracy": round(correct / total, 4) if total else None, "n": total}


# 2026-09-17 bug found via --log_samples inspection: lm-eval-harness's own
# mmlu_*_generative task YAMLs default generation_kwargs.until to
# ["</s>", "Q:", "<|im_end|>", "\n"] -- the bare "\n" stops generation at the
# model's very first newline. Chat/instruct models routinely emit a newline
# before their actual answer (e.g. "Answer:\nB" or a blank line before
# reasoning), so this was silently truncating 40-90% of responses to an EMPTY
# string per subject (confirmed: 8/20 empty in marketing, 18/20 in
# professional_law, 20/20 in astronomy) -- the resulting ~25% MMLU mean was
# almost entirely an artifact of this premature stop, not real model
# capability. Override removes the bare "\n" and keeps everything else,
# including a "\n\n" fallback so generation still can't run away forever.
#
# 2026-09-18: the "\n\n" fallback above turned out to have the exact same
# failure mode for a DIFFERENT reason -- gpt-oss-120b's harmony/markdown
# output style ("**Header**\n\n1. ...") hits a bare "\n\n" almost
# immediately, producing an empty message.content on effectively every
# sample (confirmed: [''] in raw per_task responses for every subject tested,
# mean_accuracy 0.275 was ~random-chance MCQ guessing from empty-response
# fallback scoring, not real capability). max_gen_toks (set per-task, e.g.
# 1024 for bbh) is already the real backstop against runaway generation, so
# "\n\n" was only ever a nice-to-have early-stop, not the only safety net.
# Dropped it; </s>/Q:/<|im_end|> remain as genuine end-of-turn markers.
MMLU_SAFE_GEN_KWARGS = 'until=["</s>","Q:","<|im_end|>"]'
# Same fix applied to BBH, which previously had no override at all and used
# lm-eval's raw default (until=["</s>","Q","\n\n"]) -- same bare "\n\n"
# empty-content bug, confirmed on gpt-oss-120b's boolean_expressions samples.
BBH_SAFE_GEN_KWARGS = 'until=["</s>","Q"]'

# 2026-09-18: with --reasoning off (content no longer swallowed into
# reasoning_content, see benchmark_model()), mmlu_*_generative still scored
# 0.0 across every subject. Root cause is separate from the thinking-mode
# bug: this task ships 0-shot (fewshot_config exists but nothing sets
# num_fewshot, so the harness defaults to 0), and an instruct model given a
# bare "...Answer:" prompt with no example of the expected reply format
# writes a full explanation of the underlying concept instead of a letter --
# confirmed on 20/20 real (non-empty, non-truncated) responses, none
# containing a standalone A/B/C/D. gsm8k (num_fewshot: 5) and
# bbh_cot_fewshot (num_fewshot: 3) already ship fewshot in their own task
# YAMLs and don't need this; truthfulqa_gen's num_fewshot: 0 is intentional
# (its few-shot Q&A pairs are baked into doc_to_text itself). MMLU is the one
# family invoked with no fewshot at all.
MMLU_NUM_FEWSHOT = 5


_BBH_CHOICE_RE = re.compile(r"\(([A-Z])\)")
_BBH_BOOLEAN_RE = re.compile(r"\b(Yes|No|True|False)\b", re.IGNORECASE)


def score_bbh_logged_samples(out_dir: Path, task: str) -> dict[str, Any] | None:
    """Score BBH from logged answers without leaking special tokens into EM.

    lm-eval's BBH regex correctly extracts the semantic answer but may leave
    punctuation and a model-specific end token (for example
    ``False.<|im_end|>`` or ``(B).<|im_end|>``). Its final exact-match then
    compares that whole string to ``False``/``(B)`` and reports a false zero.
    The raw logs let us compare the first typed answer token to the target.
    """
    sample_files = sorted(out_dir.glob("*/samples_*.jsonl"))
    if not sample_files:
        return None
    correct = 0
    total = 0
    for line in sample_files[-1].read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        target = str(row.get("target", "")).strip()
        filtered = row.get("filtered_resps") or []
        answer = str(filtered[0] if filtered else "")
        if answer == "[invalid]":
            raw = (row.get("resps") or [[""]])[0][0]
            answer = str(raw)
        target_choice = _BBH_CHOICE_RE.fullmatch(target)
        if target_choice:
            match = _BBH_CHOICE_RE.search(answer)
            predicted = f"({match.group(1)})" if match else None
        elif target.casefold() in {"yes", "no", "true", "false"}:
            match = _BBH_BOOLEAN_RE.search(answer)
            predicted = match.group(1).title() if match else None
        else:
            cleaned = re.sub(r"<\|[^>]+\|>", "", answer).strip()
            predicted = cleaned.strip(" \t\r\n.,;:!?\"'")
        total += 1
        correct += int(predicted is not None and predicted.casefold() == target.casefold())
    return {"accuracy": round(correct / total, 4) if total else None, "n": total}


def run_mmlu_sample(out_root: Path) -> dict[str, Any]:
    per_subject = {}
    correct_total = 0
    n_total = 0
    for subject in MMLU_SUBJECT_SAMPLE:
        task = f"mmlu_{subject}_generative"
        result = run_lm_eval_task(
            task, MMLU_LIMIT_PER_SUBJECT, out_root, log_samples=True, gen_kwargs=MMLU_SAFE_GEN_KWARGS,
            num_fewshot=MMLU_NUM_FEWSHOT,
        )
        rescored = score_mmlu_letter_match(result.get("out_dir", out_root), task) if "out_dir" in result else None
        if rescored is not None:
            per_subject[subject] = rescored["accuracy"]
            if rescored["accuracy"] is not None:
                correct_total += rescored["accuracy"] * rescored["n"]
                n_total += rescored["n"]
            continue
        metrics = result.get("metrics", {})
        acc_key = next((k for k in metrics if k.startswith("exact_match")), None)
        per_subject[subject] = metrics.get(acc_key) if acc_key else None
        if acc_key and metrics.get(acc_key) is not None:
            correct_total += metrics[acc_key] * MMLU_LIMIT_PER_SUBJECT
            n_total += MMLU_LIMIT_PER_SUBJECT
    return {
        "per_subject": per_subject,
        "mean_accuracy": round(correct_total / n_total, 4) if n_total else None,
        "n_samples": n_total,
    }


def run_bbh_sample(out_root: Path) -> dict[str, Any]:
    per_subtask: dict[str, Any] = {}
    correct_total = 0.0
    n_total = 0
    for subtask in BBH_SUBTASKS:
        result = run_lm_eval_task(subtask, BBH_LIMIT, out_root, log_samples=True, gen_kwargs=BBH_SAFE_GEN_KWARGS)
        rescored = score_bbh_logged_samples(result.get("out_dir", out_root), subtask)
        metrics = result.get("metrics", {})
        exact_key = next((k for k in metrics if "exact_match" in k), None)
        acc = rescored["accuracy"] if rescored is not None else (metrics.get(exact_key) if exact_key else None)
        per_subtask[subtask] = acc if acc is not None else result.get("error", "n/a")
        if isinstance(acc, (int, float)):
            count = rescored["n"] if rescored is not None else BBH_LIMIT
            correct_total += acc * count
            n_total += count
    return {
        "per_subtask": per_subtask,
        "mean_accuracy": round(correct_total / n_total, 4) if n_total else None,
        "n_samples": n_total,
        "scorer": "normalized_logged_answer_v1",
    }


def run_suite_against_running_server(model_name: str,
                                     physical_gpus: tuple[int, ...] | list[int] = (1, 2),
                                     specialists: tuple[str, ...] = (),
                                     vision_capable: bool = False,
                                     access_profile: str = "sandbox") -> dict[str, Any]:
    out_root = REPORTS / "lm_eval_runs" / model_name
    started = time.time()

    # log_samples on every task (not just MMLU): scripts/benchmarks/mixture_of_models.py
    # builds a post-hoc per-question ensemble across already-tested models by reading
    # these logged samples back, so no extra GPU time is needed for the mixture step.
    gsm8k = run_lm_eval_task("gsm8k", GSM8K_LIMIT, out_root, log_samples=True)
    truthfulqa = run_lm_eval_task("truthfulqa_gen", TRUTHFULQA_LIMIT, out_root, log_samples=True)
    bbh = run_bbh_sample(out_root)
    mmlu = run_mmlu_sample(out_root)
    humaneval = run_humaneval(complete_text, limit=HUMANEVAL_LIMIT)
    # Specialist scores have their own artifact/table. They are intentionally
    # excluded from the general composite: all historical models stay blank
    # rather than acquiring an incomparable zero or a retroactive score.
    specialist = run_specialists(model_name, complete_text, specialists,
                                 complete_vision if vision_capable else None,
                                 access_profile=access_profile) if specialists else None

    samples: list[dict[int, dict[str, float]]] = []
    stop = threading.Event()
    thread = threading.Thread(target=monitor_gpu, args=(stop, samples, physical_gpus), daemon=True)
    thread.start()
    try:
        perf = request_json("/v1/completions", {
            "model": MODEL_ALIAS, "prompt": PERF_PROMPT, "max_tokens": 256, "temperature": 0, "ignore_eos": True,
        })
    finally:
        stop.set()
        thread.join(timeout=2)
    timings = perf.get("timings", {})

    return {
        "model": model_name,
        "access_profile": access_profile,
        "eval_protocol": EVAL_PROTOCOL,
        "gsm8k": gsm8k.get("metrics", gsm8k),
        "truthfulqa_gen": truthfulqa.get("metrics", truthfulqa),
        "bbh": bbh,
        "mmlu_sample": mmlu,
        "humaneval": humaneval,
        "specialist_protocol": specialist.get("protocol") if specialist else None,
        "completion_tokens_per_second": timings.get("predicted_per_second"),
        "prompt_tokens_per_second": timings.get("prompt_per_second"),
        "gpu_decode_telemetry": telemetry_summary(samples, physical_gpus),
        "runtime_sec": round(time.time() - started, 1),
    }


def benchmark_model(name: str, model_path: Path, profile_name: str,
                    specialists: tuple[str, ...] = (), vision_capable: bool = False,
                    access_profile: str = "sandbox") -> dict[str, Any]:
    profile = PROFILES[profile_name]
    log_path = REPORTS / "lm_eval_runs" / f"{name}-wellknown-{profile_name}-server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(profile["visible"])
    env["GGML_CUDA_P2P"] = "1"
    from benchmark_local_gguf_tp2 import BENCH  # noqa: E402
    visible = subprocess.check_output([str(BENCH), "--list-devices"], env=env, text=True, stderr=subprocess.STDOUT)
    visible_rows = [line.strip() for line in visible.splitlines() if line.strip().startswith("CUDA")]
    # 2026-09-19: generalized from a hardcoded "must be V100, must not be RTX"
    # check to a per-profile expected_gpu match, so single-a4000 (and any future
    # non-V100 profile) can pass this guard deliberately instead of being
    # permanently rejected. Still exists for the original reason (CUDA's
    # FASTEST_FIRST default device order silently swaps ordinals on this box --
    # this confirms CUDA_VISIBLE_DEVICES actually selected the intended cards).
    # expected_gpu=None (mixed-hardware profiles like dual-a4000ada) checks each
    # row is one of the profile's allowed names instead of one single string.
    expected_gpu = profile["expected_gpu"]
    allowed_gpus = profile.get("allowed_gpus")
    if expected_gpu is not None:
        ok = len(visible_rows) == len(profile["physical"]) and all(expected_gpu in line for line in visible_rows)
    elif allowed_gpus is not None:
        ok = len(visible_rows) == len(profile["physical"]) and all(
            any(name in line for name in allowed_gpus) for line in visible_rows)
    else:
        raise RuntimeError(f"profile has neither expected_gpu nor allowed_gpus set: {profile}")
    if not ok:
        wanted = expected_gpu or allowed_gpus
        raise RuntimeError(f"device guard failed (expected {len(profile['physical'])}x {wanted!r}):\n{visible}")
    # 2026-09-18 bug found via manual curl A/B against a live server: llama-server's
    # --jinja chat template runs Qwen3-style models in "thinking" mode by default and
    # splits <think>...</think> into message.reasoning_content, leaving message.content
    # -- the only field lm-eval's local-chat-completions backend reads -- EMPTY whenever
    # a stop string or the token budget is hit before the model closes its think block.
    # Confirmed empirically: identical request with chat_template_kwargs.enable_thinking
    # false (equivalently --reasoning off) returns real content instead of "". This was
    # silently producing exact 0.0 mean_accuracy on mmlu_sample/bbh/truthfulqa_gen for
    # every model benchmarked so far (gsm8k/humaneval survived only because their
    # generation happened to finish within budget). --reasoning off makes the model
    # answer directly in message.content, matching what every scorer here expects.
    command = [str(SERVER), "--model", str(model_path), "--alias", "x",
               "--host", "127.0.0.1", "--port", str(PORT), "--ctx-size", "8192",
               "--parallel", "1", "--split-mode", str(profile["split_mode"]),
               "--main-gpu", "0", "--flash-attn", "on", "--reasoning", "off",
               "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--jinja"]
    # 2026-09-22: --fit needs to cover any model too large to gpu-layers=99
    # onto the available VRAM without automatic CPU-offload distribution --
    # not just GLM. deepseek-v4-flash-0731-iq3xxs is ~104GiB, larger than
    # even all 4 GPUs combined (~100GiB), so it structurally needs --fit too.
    NEEDS_FIT_PREFIXES = ("glm53-", "deepseek-v4-flash-0731-")
    if name.startswith(NEEDS_FIT_PREFIXES):
        margins = ",".join("1024" for _ in profile["physical"])
        # 2026-09-22: a retry of glm53-reap50-iq3m-v100 (dual-layer) after the
        # tensor_split fix below still failed. First diagnosis wrongly blamed
        # a tensor_split regression -- that was reading a stale server log
        # from the *pre-fix* attempt (log mtime 01:13, retry's own
        # benchmark_started event fired at 13:22, well_known_suite_
        # 20260917.json's mtime 12:57 predates the retry entirely -- neither
        # file was ever touched by the post-fix attempt, so its real failure
        # was never captured). An isolated llama-fit-params build (same
        # runtime, same args minus server-only flags) confirmed --fit +
        # --split-mode layer with no --tensor-split succeeds cleanly here,
        # ruling the tensor_split path back out. Real cause of the retry's
        # failure is still open. --verbose forces GGML_LOG_LEVEL_DEBUG so the
        # next attempt's server log has real diagnostics instead of guessing
        # further from a stale file.
        command.extend(["--fit", "on", "--fit-ctx", "4096", "--fit-target", margins, "--verbose"])
        if name.startswith("glm53-"):
            command.extend([
                "--override-kv", (
                    "tokenizer.ggml.eot_token_id=int:154827,"
                    "tokenizer.ggml.eom_token_id=int:154829"
                ),
            ])
            # 2026-09-22: found by reading the model's actual chat_template.jinja
            # (zai-org/GLM-5.3-Flash) after noticing 212 empty message.content
            # responses in the glm53-reap50-iq3m-allfour server log, each paired
            # with a long message.reasoning_content -- the well-known --reasoning
            # off / enable_thinking=false fix (see the comment below on --jinja)
            # does nothing here: GLM-5.3's template only reads a *different*
            # variable, "reasoning_effort", and its own fallback is
            # `reasoning_effort if ... in ['low','high'] else 'max'` -- there is
            # no "off" value, so any unset/unsupported value (including what
            # --reasoning off actually sets) silently becomes 'max'. Server logs
            # confirmed every response carried "Reasoning Effort: Max" in the
            # rendered prompt regardless of --reasoning. "low" is the lowest
            # value the template accepts; HumanEval's 512-token budget was
            # likely being consumed by the forced think block before any code,
            # which would explain GLM's anomalously low humaneval scores (0.05
            # v100 / 0.125 allfour) against every other model's 0.975-1.0.
            # Not yet re-benchmarked (weights were pruned after that run) --
            # flagged for a future GLM retry, not applied retroactively to the
            # already-recorded rows.
            command.extend(["--chat-template-kwargs", '{"reasoning_effort": "low"}'])
    else:
        command.extend(["--gpu-layers", "99"])
    # 2026-09-22 fix: this check used to read "glm53-flash-" while the --fit
    # activation above was widened to "glm53-" to also cover the REAP50
    # cascade -- leaving the two checks out of sync meant REAP50 got BOTH
    # --fit on and --tensor-split 1,1, which llama.cpp's fit logic refuses to
    # reconcile (aborts the fit, loads unfitted, OOMs). Both checks must stay
    # aligned with NEEDS_FIT_PREFIXES so --tensor-split is never combined
    # with --fit for any model that uses it.
    if len(profile["physical"]) > 1 and not name.startswith(NEEDS_FIT_PREFIXES):
        command.extend(["--tensor-split", "1,1"])
    import signal
    with log_path.open("w") as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            wait_ready(proc)
            result = run_suite_against_running_server(name, profile["physical"], specialists, vision_capable, access_profile)
            result["visible_device_probe"] = visible
            result["topology"] = f"physical GPU {profile['physical']}, split={profile['split_mode']}"
            result["cuda_device_order"] = "PCI_BUS_ID"
            result["cuda_visible_devices"] = str(profile["visible"])
            result["excluded_physical_gpus"] = [
                idx for idx in (0, 1, 2, 3) if idx not in profile["physical"]
            ]
            return result
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> int:
    global BASE_URL, MODEL_ALIAS
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*")
    # Tensor split has twice reproduced a driver-level V100 hang during long
    # decode suites. Layer split is slower but has remained stable, so keep it
    # as the unattended cascade default; tensor remains opt-in for short tests.
    parser.add_argument("--profile", default="dual-layer", choices=PROFILES)
    parser.add_argument("--out", type=Path, default=REPORTS / "well_known_suite_20260917.json")
    parser.add_argument("--external-url", help="Base URL of an already-running OpenAI-compatible server")
    parser.add_argument("--external-model", default="x", help="Model alias accepted by --external-url")
    parser.add_argument("--physical-gpus", help="Comma-separated physical GPU indexes for telemetry")
    parser.add_argument("--topology", help="Human-readable topology for an external server")
    parser.add_argument("--engine", default="vLLM", help="Runtime label for an external server")
    parser.add_argument("--specialists", default="all", metavar="LIST",
                        help="all (default), none, or comma-separated " + ",".join(SPECIALISTS))
    parser.add_argument("--vision-capable", action="store_true",
                        help="declare an image-capable endpoint (requires a configured VLM adapter)")
    parser.add_argument("--model-release-date", help="ISO publication/release date; stored with this result")
    parser.add_argument("--model-release-source", help="Primary release-note or model-card URL for that date")
    parser.add_argument("--access-profile", default="sandbox", choices=("sandbox", "disk", "internet_disk"),
                        help="tool-access condition; non-sandbox needs the dedicated agent runner")
    args = parser.parse_args()
    if bool(args.model_release_date) != bool(args.model_release_source):
        raise SystemExit("supply --model-release-date and --model-release-source together")
    if args.access_profile != "sandbox":
        raise SystemExit("disk/internet_disk require the dedicated tool-agent runner; this raw chat endpoint is sandbox-only")
    if args.specialists == "all":
        selected_specialists = SPECIALISTS
    elif args.specialists in ("none", "off", ""):
        selected_specialists = ()
    else:
        selected_specialists = tuple(item.strip() for item in args.specialists.split(",") if item.strip())
        unknown = set(selected_specialists) - set(SPECIALISTS)
        if unknown:
            raise SystemExit(f"unknown --specialists: {sorted(unknown)}; use all, none, or {','.join(SPECIALISTS)}")
    if not args.external_url and not SERVER.exists():
        raise SystemExit(f"missing V100 llama-server: {SERVER}")
    if args.external_url and not args.models:
        raise SystemExit("--external-url requires exactly one model name")
    selected = args.models or list(MODELS)
    missing = [name for name in selected if name not in MODELS]
    if missing and not args.external_url:
        raise SystemExit(f"unknown model name(s): {missing}; known: {list(MODELS)}")
    if args.external_url and len(selected) != 1:
        raise SystemExit("--external-url accepts exactly one model name per invocation")
    if args.external_url:
        BASE_URL = args.external_url.rstrip("/")
        MODEL_ALIAS = args.external_model
    # Merge into any existing report rather than overwrite: this script is invoked
    # once per model across a long multi-model sweep, and each invocation must not
    # discard results already recorded for other models.
    if args.out.exists():
        payload: dict[str, Any] = json.loads(args.out.read_text())
        payload.setdefault("results", [])
    else:
        payload = {
            "suite": "gsm8k+bbh+truthfulqa_gen+mmlu_sample(8 subjects)+humaneval(40)",
            "physical_gpus": [1, 2], "results": [],
        }
    payload["profile"] = args.profile
    failures = 0
    for name in selected:
        print(f"START {name}", flush=True)
        try:
            if args.external_url:
                physical = tuple(int(value) for value in (args.physical_gpus or "").split(",") if value.strip())
                if not physical:
                    raise ValueError("--physical-gpus is required with --external-url")
                result = run_suite_against_running_server(name, physical, selected_specialists, args.vision_capable, args.access_profile)
                result.update({
                    "engine": args.engine,
                    "topology": args.topology or f"physical GPU {list(physical)} · external server",
                    "cuda_device_order": "PCI_BUS_ID",
                    "external_base_url": BASE_URL,
                    "external_model_alias": MODEL_ALIAS,
                    "excluded_physical_gpus": [idx for idx in (0, 1, 2, 3) if idx not in physical],
                })
            else:
                result = benchmark_model(name, MODELS[name], args.profile, selected_specialists, args.vision_capable, args.access_profile)
            if args.model_release_date:
                result["model_release_date"] = args.model_release_date
                result["model_release_source"] = args.model_release_source
            else:
                result["model_release_date_status"] = "missing; add primary source before temporal contamination comparison"
            payload["results"] = [r for r in payload["results"] if r.get("model") != name] + [result]
            print(f"DONE {name}: gsm8k={result['gsm8k']} humaneval={result['humaneval']['pass_at_1']} "
                  f"mmlu={result['mmlu_sample']['mean_accuracy']} t/s={result['completion_tokens_per_second']}",
                  flush=True)
        except Exception as exc:
            failures += 1
            payload["results"] = [r for r in payload["results"] if r.get("model") != name] + [
                {"model": name, "error": f"{type(exc).__name__}: {exc}"}]
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
        write_result(args.out, payload)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
