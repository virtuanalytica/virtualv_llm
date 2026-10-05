#!/bin/bash
# Wait for the pinned AesSedai BPW2.5 shards and the sm_70 llama.cpp build,
# then run the current well-known suite on physical V100s 1 and 2.
set -euo pipefail

ROOT=/media/knight2/EDS2/projects/virtualv_llm
DEST=/media/knight2/EDS2/models/llm/mimo-v26-pro-bpw2.5/BPW2.5
BIN=/media/knight2/EDS2/tools/llama.cpp-b11149/build-v100/bin/llama-server
BENCH=/media/knight2/EDS2/tools/llama.cpp-b11149/build-v100/bin/llama-bench
REPORT="$ROOT/reports/well_known_suite_20260917.json"
DASH_NUMERAI=/media/knight2/EDS2/projects/numerai-signals/reports/dual_v100_nvlink_benchmark.html
REV=a0af536b79b0b312eaddb7fea6a3f56e50e739c0

declare -A EXPECT=(
  [MiMo-V2.6-Pro-RL-BPW2.5-00001-of-00008.gguf]=5949120
  [MiMo-V2.6-Pro-RL-BPW2.5-00002-of-00008.gguf]=49591158560
  [MiMo-V2.6-Pro-RL-BPW2.5-00003-of-00008.gguf]=48301453152
  [MiMo-V2.6-Pro-RL-BPW2.5-00004-of-00008.gguf]=48611845920
  [MiMo-V2.6-Pro-RL-BPW2.5-00005-of-00008.gguf]=49602775072
  [MiMo-V2.6-Pro-RL-BPW2.5-00006-of-00008.gguf]=49235709504
  [MiMo-V2.6-Pro-RL-BPW2.5-00007-of-00008.gguf]=48158861088
  [MiMo-V2.6-Pro-RL-BPW2.5-00008-of-00008.gguf]=26235779072
)

ready() {
  [ -x "$BIN" ] && [ -x "$BENCH" ] || return 1
  local name
  for name in "${!EXPECT[@]}"; do
    [ -f "$DEST/$name" ] && [ "$(stat -c%s "$DEST/$name")" = "${EXPECT[$name]}" ] || return 1
  done
}

echo "waiting for BPW2.5 shards and llama.cpp b11149 $(date -Is)"
until ready; do
  sleep 30
done
echo "ready $(date -Is)"

cd "$ROOT"
export LLAMA_SERVER="$BIN"
export LLAMA_BENCH="$BENCH"
set +e
flock -n /tmp/v100_exclusive.lock /usr/bin/python3 scripts/benchmarks/well_known_suite.py \
  mimo-v26-pro-bpw2.5 \
  --profile dual-layer \
  --model-release-date 2026-09-21 \
  --model-release-source https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL \
  --out "$REPORT"
suite_rc=$?
set -e
echo "suite_exit $suite_rc $(date -Is)"

python3 - << 'PY'
import json
from pathlib import Path
report = Path("/media/knight2/EDS2/projects/virtualv_llm/reports/well_known_suite_20260917.json")
data = json.loads(report.read_text())
for row in data.get("results", []):
    if row.get("model") != "mimo-v26-pro-bpw2.5":
        continue
    row["model_source"] = "https://huggingface.co/AesSedai/MiMo-V2.6-Pro-RL-GGUF"
    row["source_repo"] = "AesSedai/MiMo-V2.6-Pro-RL-GGUF"
    row["source_revision"] = "a0af536b79b0b312eaddb7fea6a3f56e50e739c0"
    row["quantization"] = "BPW2.5"
    row["weight_bytes"] = 319743531488
    row["base_model"] = "XiaomiMiMo/MiMo-V2.6-Pro-RL"
    row["llama_cpp"] = "b11149"
tmp = report.with_suffix(".json.tmp")
tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
tmp.replace(report)
PY

VIRTUALV_DASHBOARD_OUT="$DASH_NUMERAI" python3 scripts/benchmarks/build_dual_v100_html.py
cp -a "$DASH_NUMERAI" "$ROOT/reports/dual_v100_nvlink_benchmark.html"
echo "dashboard refreshed $(date -Is)"
exit "$suite_rc"
