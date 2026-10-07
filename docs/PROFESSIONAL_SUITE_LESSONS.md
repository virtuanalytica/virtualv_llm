# Lessons from professional LLM benchmarking suites

Status date: 2026-10-07. What established suites do that this suite did not
yet do, what was adopted in this change, and what remains open. "Adopted"
means code and a CI check exist in this repository, not an intention.

## What was compared

| Suite | Core idea taken from it |
|---|---|
| [MLPerf Inference](https://github.com/mlcommons/inference_policies) | Results are submissions: fixed rules, a machine checker, a review period, and a closed division (identical workload) kept apart from an open one |
| [HELM](https://crfm.stanford.edu/helm/) | Scenario, metric and adaptation are recorded separately; every prediction is inspectable, not only the aggregate |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Tasks carry versions; every metric is reported with a standard error |
| [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640) | Report confidence intervals; compare models with paired differences on the same items; plan sample size from the effect you need to detect |
| [LiveBench](https://arxiv.org/abs/2406.19314) | Contamination is handled by rotating fresh questions and objective scoring, not by trusting a static set |
| [SWE-bench](https://www.swebench.com/) submissions | A leaderboard entry ships its logs and trajectories; "verified" is a status a maintainer grants |

## Findings on our own data

Computed by `scripts/benchmarks/score_confidence.py` from
`reports/well_known_suite_20260917.json` (committed as
`reports/score_confidence.json`):

- The composite rests on 298 items. Its 95% margin is about ±3.3 to ±5
  percentage points per row.
- Of 46 complete sandbox rows, 10 cannot be told apart from the leader. All
  seven Qwen3.8 Flash-Next rows (0.9113 to 0.9272) are inside each other's
  margin, so the suite has not shown that one quant or topology is more
  accurate than another. Their throughput differences are real; their
  composite order is not.
- Specialist packs hold 2 to 6 items per lane. A perfect 3 of 3 only shows the
  true rate is above 44%. Claiming "at least 95% correct" needs 73 consecutive
  correct items in a lane.

- HumanEval (40 items) is the one task with per-item outcomes, so it can be
  compared pairwise (`humaneval_paired` in the same report). 18 of 45 rows
  pass exactly the same items as the leader (39 of 40, the same item failed):
  zero discordant pairs. At the top of the table this component is saturated
  and adds nothing to the order; it only separates weaker models (13 of 45
  rows differ from the leader at the 5% level).

## Adopted in this change

| Lesson | Implementation |
|---|---|
| Error bars on every ranked score | `score_confidence.py`; dashboard section "Statistische betrouwbaarheid" lists margin, interval and whether a row is tied with the leader |
| Results are submissions with a checker | `scripts/reporting/submission.py` (`make`, `validate`, `guard`), `config/submission_protocol.json` |
| Closed division kept apart | Community rows render in their own table and never enter the reference ranking |
| "Verified" is granted, not claimed | `community-unverified` is the only status an outside author may set; `verified-reproduced` needs a maintainer review block |
| A gate the submitter cannot edit | `contribution-guard` runs the base branch's guard under `pull_request_target` and reads the pull request as data |
| Evidence travels with the number | A submission must carry weight and evidence SHA-256, full source and suite commits, and a public evidence URL |
| Scores are counts, aggregates are derived | Submissions hold integer `correct`/`n`; a `composite` field is rejected |

## Open, in priority order

1. **Per-item outcomes for every task.** Only HumanEval keeps them in the
   curated JSON. With them, a paired test (McNemar or paired differences)
   replaces the conservative unpaired one and separates close models.
2. **Confirmation runs for finalists.** Rows tied at the top need a larger
   sample before one is called better. This is a new protocol ID (standard,
   section 10), for example full GSM8K and HumanEval, run only for the tied set.
3. **Throughput as a distribution.** One 256-token completion gives a point
   value. Record time to first token and the median and 95th percentile over
   repeated runs at each batch size, as MLPerf's scenarios do.
4. **Larger rotated specialist packs.** At least 73 items per lane, rotated,
   with a frozen holdout that nothing is ever tuned on.
5. **Repeatability.** Re-run one reference row weekly; a drift beyond the
   interval means the runtime changed, not the model.
