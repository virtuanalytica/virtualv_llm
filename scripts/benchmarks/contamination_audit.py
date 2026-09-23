#!/usr/bin/env python3
"""Data-contamination mitigations for the well-known/specialist suites.

A high public-benchmark score alone is not independent evidence of general
ability: public GSM8K/MMLU/BBH/HumanEval question-answer pairs can end up in
pretraining, post-training, or benchmark-specific optimization for any model
whose training cutoff postdates their publication. This module does not try
to prove contamination happened (that requires access to training data this
project doesn't have) -- it runs three standard, training-data-independent
audit methods and reports the gap as a research signal for a human to weigh,
never as a verdict.

Comparisons here are only meaningful for the SAME model configuration that
also has a standard well_known_suite.py result -- run that first.

1. canary_recall (verbatim-completion canary test): give the model only a
   short, identifying fragment of a real, published GSM8K test question (not
   the full question) and ask it to complete the fragment verbatim if it
   recognizes it, or say UNKNOWN otherwise. A model that reproduces the exact
   published continuation from a two-word cue did not derive that text by
   reasoning about the prompt -- it is reciting memorized training data.
   Standard technique in contamination literature (the "canary GUID" family).

2. paraphrase_invariance: ask the same underlying question two ways -- the
   original wording, and a deterministically permuted variant (multiple-
   choice options reordered) -- and compare accuracy. A large drop under a
   superficial rewrite that preserves the reasoning task indicates the
   original answer came from matching the exact surface form of a memorized
   question, not from solving the underlying problem.

3. post_cutoff_holdout: score the model on config/contamination_audit/
   post_cutoff_holdout.csv, a set of original word problems written for this
   project and never published anywhere before 2026-09-23 -- by construction,
   no model's training data can contain them. Compare this accuracy against
   the same model's public-suite accuracy on an equivalent-difficulty task
   (gsm8k). A public-suite score well above the holdout score on comparable
   problems is the contamination research signal described above; a
   comparable holdout score is reassuring, not proof of a clean model.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Callable

from result_store import upsert_result

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config" / "contamination_audit"
OUT = REPORTS / "contamination_audit_20260923.json"
PROTOCOL = "v1-canary-paraphrase-postcutoff-20260923"

_ANSWER_LETTER_RE = re.compile(r"\b([ABCD])\b")
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _load_json(name: str) -> dict[str, Any]:
    return json.loads((CONFIG / name).read_text())


def _run_canary_recall(complete: Callable[[str, int], str]) -> dict[str, Any]:
    seed = _load_json("gsm8k_canary_fingerprints.json")
    details = []
    recalled = 0
    for item in seed["items"]:
        question = item["question"]
        # Roughly the first 8-10 words: enough to uniquely identify a specific
        # published question to a model that has actually memorized it, too
        # short to let the model just solve the arithmetic from context (an
        # 8-word fragment of a multi-step GSM8K problem is rarely solvable on
        # its own -- a correct verbatim continuation therefore signals recall).
        words = question.split()
        fragment = " ".join(words[: min(9, max(3, len(words) // 3))])
        rest = question[len(fragment):].strip()
        prompt = (
            "The following is the start of a sentence. If you recognize this exact sentence "
            "from a dataset you have seen, continue it VERBATIM, word for word, with nothing "
            "else. If you do not recognize it, reply with exactly: UNKNOWN\n\n"
            f"Sentence start: {fragment}"
        )
        reply = complete(prompt, 256).strip()
        if reply.upper().startswith("UNKNOWN"):
            overlap = 0.0
        else:
            rest_tokens = set(re.findall(r"\w+", rest.lower()))
            reply_tokens = set(re.findall(r"\w+", reply.lower()))
            overlap = (len(rest_tokens & reply_tokens) / len(rest_tokens)) if rest_tokens else 0.0
        is_recall = overlap >= 0.6
        recalled += int(is_recall)
        details.append({"doc_id": item["doc_id"], "fragment": fragment, "token_overlap": round(overlap, 3),
                        "verbatim_recall": is_recall})
    return {
        "status": "complete", "method": "canary_recall",
        "recall_rate": round(recalled / len(seed["items"]), 4), "n_samples": len(seed["items"]),
        "interpretation": ("Fraction of held-out GSM8K test questions the model could continue "
                           "verbatim from a short fragment alone. High values indicate the exact "
                           "published text is present in training data, not just the underlying "
                           "math skill."),
        "source": seed["source"], "samples": details,
    }


def _permute_choices(choices: list[str], answer_index: int) -> tuple[list[str], int]:
    # Deterministic reversal: same permutation every run (reproducible), and
    # for 4-option MC it never leaves any choice in its original slot, unlike
    # some fixed cyclic shifts -- maximizes the surface-form change being
    # tested while the underlying question and correct choice stay identical.
    permuted = list(reversed(choices))
    new_index = len(choices) - 1 - answer_index
    return permuted, new_index


def _run_paraphrase_invariance(complete: Callable[[str, int], str]) -> dict[str, Any]:
    seed = _load_json("mmlu_paraphrase_seed.json")
    letters = "ABCD"
    original_correct = 0
    paraphrase_correct = 0
    details = []
    for item in seed["items"]:
        question, choices, answer_index = item["question"], item["choices"], item["answer_index"]
        target_letter = letters[answer_index]

        def ask(opts: list[str]) -> str | None:
            opts_text = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(opts))
            prompt = f"{question}\n\n{opts_text}\n\nReason privately, then output only the single answer letter."
            reply = complete(prompt, 512)
            match = _ANSWER_LETTER_RE.search(reply.strip())
            return match.group(1) if match else None

        original_pred = ask(choices)
        permuted_choices, permuted_index = _permute_choices(choices, answer_index)
        permuted_target_letter = letters[permuted_index]
        permuted_pred = ask(permuted_choices)

        original_ok = original_pred == target_letter
        permuted_ok = permuted_pred == permuted_target_letter
        original_correct += int(original_ok)
        paraphrase_correct += int(permuted_ok)
        details.append({"doc_id": item["doc_id"], "subject": item.get("subject"),
                        "original_correct": original_ok, "paraphrase_correct": permuted_ok})

    n = len(seed["items"])
    original_acc = round(original_correct / n, 4) if n else None
    paraphrase_acc = round(paraphrase_correct / n, 4) if n else None
    return {
        "status": "complete", "method": "paraphrase_invariance",
        "original_accuracy": original_acc, "paraphrase_accuracy": paraphrase_acc,
        "accuracy_gap": round(original_acc - paraphrase_acc, 4) if n else None,
        "n_samples": n,
        "interpretation": ("Accuracy on original MMLU questions vs. the same questions with "
                           "multiple-choice options deterministically reordered (correct answer "
                           "moved to a different letter, question text unchanged). A large drop "
                           "under reordering indicates the original score depended on matching a "
                           "memorized answer-letter pattern rather than the underlying reasoning."),
        "samples": details,
    }


def _extract_number(text: str) -> str | None:
    matches = _NUMBER_RE.findall(text.replace(",", ""))
    return matches[-1] if matches else None


def _run_post_cutoff_holdout(complete: Callable[[str, int], str]) -> dict[str, Any]:
    pack = CONFIG / "post_cutoff_holdout.csv"
    with pack.open(newline="") as fh:
        rows = [row for row in csv.DictReader(fh) if row.get("enabled", "true").strip().lower() == "true"]
    correct = 0
    details = []
    for row in rows:
        prompt = (
            row["prompt"] + "\n\n" +
            "\n".join(f"{letter}. {row[f'option_{letter.lower()}']}" for letter in "ABCD") +
            "\n\nReason privately, then output only the single answer letter."
        )
        reply = complete(prompt, 512)
        match = _ANSWER_LETTER_RE.search(reply.strip())
        predicted = match.group(1) if match else None
        target = row["answer"].strip().upper()
        is_correct = predicted == target
        correct += int(is_correct)
        details.append({"id": row["id"], "correct": is_correct, "target": target, "prediction": predicted})
    import hashlib
    return {
        "status": "complete", "method": "post_cutoff_holdout",
        "accuracy": round(correct / len(rows), 4) if rows else None, "n_samples": len(rows),
        "question_pack": str(pack.relative_to(ROOT)), "pack_sha256": hashlib.sha256(pack.read_bytes()).hexdigest(),
        "interpretation": ("Accuracy on original word/logic problems written for this project on "
                           "2026-09-23 and never published anywhere before -- no model's training "
                           "data can contain them by construction. Compare against this same "
                           "model's gsm8k/mmlu_sample accuracy in well_known_suite_20260917.json: "
                           "a public-suite score well above this holdout on comparable difficulty "
                           "is the contamination research signal; a comparable score is "
                           "reassuring, not proof of an uncontaminated model."),
        "samples": details,
    }


METHODS = ("canary_recall", "paraphrase_invariance", "post_cutoff_holdout")


def run_contamination_audit(model: str, complete: Callable[[str, int], str],
                            selected: tuple[str, ...] = METHODS, out: Path = OUT,
                            access_profile: str = "sandbox") -> dict[str, Any]:
    unknown = set(selected) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown contamination-audit methods: {sorted(unknown)}")
    result: dict[str, Any] = {"model": model, "protocol": PROTOCOL, "access_profile": access_profile,
                              "results": {}}
    runners = {"canary_recall": _run_canary_recall, "paraphrase_invariance": _run_paraphrase_invariance,
              "post_cutoff_holdout": _run_post_cutoff_holdout}
    for method in selected:
        try:
            result["results"][method] = runners[method](complete)
        except Exception as exc:
            result["results"][method] = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    upsert_result(out, result, {"suite": "contamination_audit", "protocol": PROTOCOL, "results": []},
                  key=lambda row: (row.get("model"), row.get("access_profile", "sandbox")))
    return result
