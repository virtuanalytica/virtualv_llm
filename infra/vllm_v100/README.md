# vLLM on the two V100s

Everything is placed on EDS2: the isolated CUDA-12.4 environment, Hugging Face
cache, vLLM cache, temporary files, logs and model weights.  The launcher fixes
`CUDA_VISIBLE_DEVICES=1,2`, uses tensor parallelism two, and interleaves CPU RAM
over both NUMA nodes because the V100s are local to NUMA node 1.

Run the hardware/model check first:

```bash
/media/knight2/EDS2/envs/vllm-v100-cu124/bin/python preflight.py \
  --model deepseek-ai/DeepSeek-V4-Flash-0731
```

It intentionally rejects the requested DeepSeek-V4 Flash / DSpark releases on
this host.  The official release has roughly 304B FP8-oriented parameters,
while the two V100s have 64GiB combined VRAM and no native FP8/FP4 execution.
CPU offload can consume DDR4 but cannot make that weight format execute on SM70;
reserving all 320GB would only harm the forecasting workload.

The installed V100-compatible environment is vLLM 0.6.5 with PyTorch 2.5.1
CUDA 12.4 (`sm_70` present). It is suitable for ordinary SM70-compatible
models, but not DSpark: vLLM DSpark support requires a much newer vLLM release,
whose supplied CUDA wheel on this host lacks V100 (`sm_70`) kernels. Therefore
there is no honest vLLM+DSpark configuration for the requested Flash release on
this hardware.

For a separately approved SM70-compatible model, first run `preflight.py`, then:

```bash
./serve_two_v100s.sh MODEL_ID
python benchmark_tokens.py --model MODEL_ID
```

The latter stores real output-token throughput in
`/media/knight2/EDS2/logs/vllm/tokens_per_second.json`; it never invents a
tokens/s result when no compatible model is served.

## 1Cat-vLLM 1.5 TP=2 test lane

The pinned 1Cat-vLLM 1.5 environment and Qwen3.8 checkpoints are now separate
from the obsolete official-vLLM environments:

- environment: `/media/knight2/EDS2/envs/1cat-vllm-1.5.0`
- target: `/media/knight2/EDS2/models/1cat-vllm/Qwen3.8-27B-QUASAR-NVFP4`
- draft: `/media/knight2/EDS2/models/1cat-vllm/Qwen3.8-27B-DFlash2`

Run the matched target-only/DFlash2 matrix with:

```bash
./run_1cat_tp2_matrix.sh
```

It exposes only physical GPU 1 and 2, holds `/tmp/v100_exclusive.lock`, stops
the independent V100 chat services for the test, keeps the A4000 model services
off, and restores the V100 services on exit. The initial local contract uses
8K context and one resident sequence. It records a forced 256-token decode,
the six-field RIV-AU extraction, per-card utilization/memory/power/temperature,
and DFlash acceptance metrics. Do not compare its wall t/s with the llama.cpp
server-side timing without labeling the measurement method.
