# Contributing to VirtualV LLM

Outside contributions are welcome under the protocol below. It is strict on
purpose: a benchmark is only worth reading if nobody, the maintainers
included, can change a number without leaving evidence.

Two kinds of pull request are accepted. Do not mix them.

## 1. Result submission

You measured a model on your own hardware and want it listed.

1. Run the unmodified suite at a commit of `main`:
   ```bash
   python3 scripts/benchmarks/well_known_suite.py MODEL_ID \
     --external-url http://127.0.0.1:PORT --external-model SERVED_MODEL \
     --physical-gpus 0,1 --topology '2x <your GPU>' --engine '<runtime>' \
     --out /tmp/my_run.json
   ```
2. Publish the raw evidence (the `lm_eval_runs` logs of that run) as one
   archive at a public `https://` URL and note its SHA-256 and size.
3. Generate the submission file. Every flag is required; nothing is guessed:
   ```bash
   python3 scripts/reporting/submission.py make MODEL_ID --report /tmp/my_run.json \
     --github YOUR_HANDLE --slug model-quant-hardware \
     --source-repo org/repo --source-revision <40-char commit> \
     --quantization Q4_K_M --weight-sha256 <sha256 of the weights> --license apache-2.0 \
     --engine llama.cpp --engine-version <build or commit> --context-tokens 8192 \
     --suite-commit <40-char commit of this repository> \
     --gpu 'NVIDIA RTX 4090' --gpu 'NVIDIA RTX 4090' --topology 'layer split, PCIe' \
     --evidence-url https://… --evidence-sha256 <sha256> --evidence-bytes <size>
   ```
4. Open a pull request that adds only that file.

What the gate enforces (`contribution-guard`, run from the base branch):

| Rule | Reason |
|---|---|
| Only new files under `submissions/<your handle>/` | You speak for your own runs; existing evidence is immutable |
| No other file in the same pull request | Results and code are reviewed separately |
| Accepted protocol ID and its exact sample sizes | Scores under another protocol are not comparable |
| Scores as integer `correct`/`n`; no composite field | The composite is derived, never typed |
| Full commit hashes for weights source and suite; SHA-256 for weights and evidence | The run must be reproducible from the file alone |
| `status: community-unverified`, no `review` block | Only a maintainer who reproduced a run may raise its status |
| No private paths or credential-shaped strings, at most 64 KiB, at most 5 files | Hygiene and review load |

A correction is a new file with `"supersedes": "<handle>/<old-slug>"`; the old
file stays.

Statuses: `community-unverified` (passed the gate, not reproduced),
`verified-reproduced` (a maintainer re-ran it and the composite fell inside
the 95% interval reported by `scripts/benchmarks/score_confidence.py`),
`rejected`, `withdrawn`. Community rows are displayed apart from the reference
ranking in every status.

## 2. Code or documentation change

Welcome anywhere except the protected paths in
`config/submission_protocol.json`: workflows, `config/`, `reports/`, `infra/`,
the scorers and runners that define the protocol, the validators and the
standard. To change one of those, open an issue describing the change and the
evidence; a maintainer makes it under a new protocol ID where the standard
requires one (section 10).

Requirements: tests for new behaviour, `python -m pytest -q` passing with the
real output pasted in the pull request, comments in English, no model weights,
no private evaluation items, no secrets.

## Review and merge

- Every pull request needs the `validate` and `guard` checks green and an
  approving maintainer review (CODEOWNERS).
- Workflows from first-time contributors run only after a maintainer approves
  the run.
- The maintainer merges the exact commit that was reviewed; a push after
  review resets it.
- Disputes about a published row go in an issue with evidence. A row that
  cannot be defended is marked `rejected` or `withdrawn`, not deleted.

## Security

Never put tokens, internal hostnames or home-directory paths in a submission
or log archive. Report a vulnerability privately through GitHub security
advisories on this repository, not in a public issue.
