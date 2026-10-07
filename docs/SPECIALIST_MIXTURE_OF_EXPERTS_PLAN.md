# Specialist mixture of experts: plan and evidence

Status date: 2026-10-07. Design for an own mixture of models and expert models
that answers every specialist lane except vision, within a throughput budget,
and improves itself through the Toddler/Teacher loop. Nothing here is measured
yet unless a number is given with its source.

## Where we stand (measured)

From `reports/specialist_suite_20260922.json`, sandbox profile, protocol v2:

| Lane | Items | Best local model | Best local | Best overall |
|---|---:|---|---:|---:|
| chemistry | 6 | Qwen3.8 Flash-Next AP-IQ4_XS, all four | 17% | 100% (Gemini 3.8 Flash, cloud) |
| physics | 6 | Qwen3.8 Flash-Next AP-IQ4_XS | 67% | 100% (cloud) |
| IQ | 3 | GLM-5.3 REAP50 IQ3_M, V100 | 67% | 100% (cloud) |
| EQ | 3 | several | 100% | 100% |
| FQ | 2 | GLM-5.3 REAP50 IQ3_M, all four | 50% | 100% (Gemini 3.6 Flash, cloud) |
| QQ | 4 | Qwen3.8 Flash-Next AP-Q4_K_XL V100; Granite 4.2 | 100% | 100% |
| video | 3 | none | 0% | 0% |
| finance | 24 | not yet run on these rows | – | – |

Two consequences:

- Locally the gap is large in chemistry, FQ and video. Routing between the
  models we have cannot close it: no local model is good at those lanes.
- The packs are too small to prove 100%. With 2 to 6 items a perfect score has
  a 95% lower bound between 34% and 61%. The target "100% on every lane" needs
  at least 73 items per lane before it can be claimed at the 95% level.

## Architecture

1. **Router.** Classifies the question into a lane and picks the expert. Rule
   based on lane keywords first; a small fast model only for the remainder.
   The router is scored on its own (lane accuracy), apart from answer quality.
2. **One expert per lane, chosen by measurement.** An expert is a model plus
   its tools and prompt, not a model alone:
   - *FQ* is a deterministic differential-drive simulation. The expert writes
     the parameters and a simulator computes the answer; the model does not do
     the arithmetic.
   - *Video* asks for an FFmpeg filtergraph. The expert runs its own graph in
     the fixed sandbox, reads the error, and repairs it up to a retry limit
     before answering.
   - *Chemistry, physics, QQ* use the strongest reasoner that fits the budget,
     with a calculator and unit checker, and self-consistency over several
     samples.
   - *Finance* answers from the identifier and crypto fact files, with the
     cited row returned alongside the answer.
   - *IQ, EQ* use the fast general model; EQ is already at ceiling.
3. **Verifier.** A second model or a deterministic check accepts the answer or
   sends it back once. Disagreement escalates to the slow quality lane.
4. **Abstention.** "Not sure" is a valid output and is scored as wrong, never
   hidden. Vision stays `unsupported`.

## Throughput budget

Measured single-stream rates decide who may sit where: Qwen3.6-35B-A3B NVFP4
on 1Cat TP2 at 111 t/s (B1), Qwen3.8 Flash-Next AP-IQ2_S at 41 to 43 t/s,
DeepSeek-V4-Flash at 13.9 t/s. A mixture's usable rate is
`1 / sum(1/member_rate)` over the members actually called, so:

- Day lane, target 50 t/s interactive: router plus one fast expert, verifier
  only on low confidence.
- Background lane, target 25 t/s: expert plus verifier on every answer.
- Evening lane, 5 t/s allowed: self-consistency and the slow quality judge.

Every mixture row reports the rate of the path that was really taken, per
lane, next to its score.

## Self-healing, self-learning, self-actualising

Three loops, each with a gate so it cannot fool itself:

| Loop | Trigger | Action | Gate |
|---|---|---|---|
| Self-healing | An expert errors, times out or fails its deterministic check | Retry with the repair prompt, then fall back to the next expert; record the failure class | Fallback rate and failure taxonomy are reported; a rising rate blocks promotion |
| Self-learning | A wrong answer on the *training* pool | The Teacher writes new own items for that weakness; the Toddler-side learner updates router rules, prompts and retrieval notes | Promotion only if the frozen holdout improves and no lane regresses beyond its interval |
| Self-actualising | Weekly | The system proposes its own next experiment from the weakest lane and the cost per point gained | A person or the reviewer agent approves before weights, tools or protocol change |

Hard rules: the frozen holdout is never used to tune anything; Teacher items
pass the originality and leakage gates already used for syllabi; a change that
alters prompts, scorers or packs gets a new protocol ID; every promotion is a
pull request with before and after numbers and their intervals.

## Order of work

1. Grow each lane to at least 73 items with a frozen holdout split
   (`config/benchmark_specialist_holdout_v1.csv` is the start).
2. Run the finance lane and the missing chemistry and physics cells on the
   current local rows, so the table above has no gaps.
3. Build the two tool experts first (FQ simulator, video repair loop): they
   address the lanes where every model scores at or near zero.
4. Measure router accuracy, then the mixture per lane with its real rate.
5. Only then wire the self-learning loop to the Teacher.
