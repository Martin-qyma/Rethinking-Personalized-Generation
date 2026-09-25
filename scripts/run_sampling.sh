#!/usr/bin/env bash
# Data-parallel candidate sampling: one vLLM process per GPU, then merge.
#
#   GPUS=0,1,2,3 bash scripts/run_sampling.sh LaMP_7 dev
#   GPUS=0,1,2,3 N=16 bash scripts/run_sampling.sh XRec_amazon ctx   # context split (App. B.3)
#
# Environment: GPUS (default 0), N (samples per prompt, default 256),
# DATA_ROOT (default data), LOG_DIR (default logs/sampling).
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET=${1:?dataset}; SPLIT=${2:?split}
GPUS=${GPUS:-0}; N=${N:-256}; DATA_ROOT=${DATA_ROOT:-data}; LOG_DIR=${LOG_DIR:-logs/sampling}
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
SHARDS=${#GPU_ARR[@]}
mkdir -p "$LOG_DIR"

pids=()
for i in "${!GPU_ARR[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPU_ARR[$i]} python scripts/sample.py --dataset "$DATASET" --split "$SPLIT" \
    --data_root "$DATA_ROOT" --num_samples "$N" --num_shards "$SHARDS" --shard "$i" \
    > "$LOG_DIR/${DATASET}_${SPLIT}_shard$i.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
python scripts/sample.py --dataset "$DATASET" --split "$SPLIT" --data_root "$DATA_ROOT" \
  --num_shards "$SHARDS" --merge
