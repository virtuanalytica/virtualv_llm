# Source this file to export CUDA_VISIBLE_DEVICES for the NVLinked Tesla V100 pair, found by name.
# The GPU inventory changed on 2026-10-05 (V100s moved from PCI indices 1,2 to 3,4), so serve
# scripts must not hard-code indices. Indices, not UUIDs: 1Cat-vLLM 1.5 parses the variable as
# integers ("invalid literal for int()" with a GPU UUID).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mapfile -t _V100_IDX < <(nvidia-smi --query-gpu=index,name --format=csv,noheader | awk -F, '/Tesla V100-SXM2-32GB/ {gsub(/[[:space:]]/, "", $1); print $1}')
if [[ ${#_V100_IDX[@]} -ne 2 ]]; then
  echo "Expected two healthy Tesla V100-SXM2-32GB GPUs, found ${#_V100_IDX[@]}" >&2
  exit 75
fi
export CUDA_VISIBLE_DEVICES="${_V100_IDX[0]},${_V100_IDX[1]}"
unset _V100_IDX
