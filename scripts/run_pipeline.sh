#!/usr/bin/env bash
# End-to-end pipeline for one dataset (after data preparation, see README).
#
#   GPUS=0,1,2,3 bash scripts/run_pipeline.sh LaMP_7
#
# 1. sample 256 candidates per prompt (train + eval split)
# 2. label candidates with ROUGE-1/ROUGE-L/BLEU
# 3. extract h_x, h_u, h_y for the first 64 candidates
# 4. train the ranker (four sizes) and evaluate Best-of-N
# 5. score the evaluation pools with the four generalist reward models
# 6. ranking-guided generation with the default ranker
set -euo pipefail
cd "$(dirname "$0")/.."
DS=${1:?dataset}; GPUS=${GPUS:-0}
GPU0=${GPUS%%,*}
EVAL=$(python -c "from rethinking.datasets import get; print(get('$DS').eval_split)")

for SPLIT in train "$EVAL"; do
  [ -f "data/$DS/${SPLIT}_samples.json" ] || GPUS=$GPUS bash scripts/run_sampling.sh "$DS" "$SPLIT"
  [ -f "data/$DS/${SPLIT}_samples_scores.json" ] || python scripts/score_candidates.py --dataset "$DS" --split "$SPLIT"
  [ -f "data/$DS/${SPLIT}_cand_embeddings.npz" ] || \
    CUDA_VISIBLE_DEVICES=$GPU0 python scripts/embed.py --dataset "$DS" --split "$SPLIT"
done

for SIZE in 1M 3M 10M 30M; do
  CUDA_VISIBLE_DEVICES=$GPU0 python scripts/train_ranker.py --dataset "$DS" --size "$SIZE"
done

IFS=',' read -r -a G <<< "$GPUS"
i=0
for RM in skywork internlm urm armorm; do
  CUDA_VISIBLE_DEVICES=${G[$((i % ${#G[@]}))]} python scripts/score_reward_model.py --rm "$RM" --dataset "$DS" &
  i=$((i + 1))
  (( i % ${#G[@]} == 0 )) && wait
done
wait

CUDA_VISIBLE_DEVICES=$GPU0 python scripts/guided_generation.py --dataset "$DS" \
  --ckpt "checkpoints/ranker/$DS/3M_mse" --grid sensitivity
