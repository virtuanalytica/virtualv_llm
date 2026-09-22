# VirtualV LLM Testsuite

> Measure what a local model actually does on the hardware you actually have.

VirtualV LLM is a hardware-aware evaluation suite for GGUF/llama.cpp and
1Cat-vLLM workloads on heterogeneous GPU pools. It keeps quality, throughput,
hardware topology and provenance together, while failures remain explicit
instead of becoming invented scores.

The reference host combines an RTX A4000, two Tesla V100-SXM2 32 GB cards over
NVLink and an RTX 4000 Ada. Its measurements illustrate the method; they are
not universal performance claims.

## Current results

- [Interactive benchmark dashboard](reports/dual_v100_nvlink_benchmark.html)
- [Curated machine-readable evidence](reports/well_known_suite_20260917.json)
- [Business standard and operator manual](docs/VIRTUALV_LLM_TESTSUITE_STANDARD.md)
- [Current model and runtime roadmap](docs/MODEL_TEST_ROADMAP.md)
- [Business PDF and weight-free v1 backup](output/pdf/)

The completed migration from `numerai-signals` is documented in Git history.
The active runners, scheduled dashboard build and systemd unit now use this
repository. Model weights, secrets, private evaluation packs and raw logs are
excluded from Git; raw evidence is published separately as a checksummed
release artifact.

## Quick start

```bash
git clone https://github.com/virtuanalytica/virtualv_llm.git
cd virtualv_llm
cp .env.example .env
python3 scripts/benchmarks/bootstrap_public_data.py
python3 -m py_compile scripts/benchmarks/*.py scripts/reporting/*.py
python3 scripts/benchmarks/build_dual_v100_html.py
```

Benchmark an OpenAI-compatible local endpoint:

```bash
python3 scripts/benchmarks/well_known_suite.py MODEL_ID \
  --external-url http://127.0.0.1:PORT \
  --external-model SERVED_MODEL \
  --physical-gpus 1,2 \
  --topology '2x Tesla V100-SXM2-32GB NVLink' \
  --engine 'llama.cpp' \
  --out reports/well_known_suite_local.json
```

The full run protocol, failure handling and evidence requirements are in the
[operator standard](docs/VIRTUALV_LLM_TESTSUITE_STANDARD.md). Python 3.12+ is
recommended. A compatible local inference runtime and `lm-eval` are required
for actual GPU runs.

## Repository policy

- Curated JSON, HTML, documentation and PDFs are tracked.
- Raw logs are immutable release artifacts with SHA-256 manifests.
- No result is comparable without matching protocol, engine, context and
  hardware metadata.
- Long GPU jobs are resumable, exclusive and independently managed by systemd.

Licensed under Apache-2.0. Third-party models, datasets and runtimes retain
their own licenses; consult `NOTICE` before redistributing them.
