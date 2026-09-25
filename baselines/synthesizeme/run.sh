#!/bin/bash
# SynthesizeMe baseline pipeline on XRec. Run from the repository root.
#
# Prerequisites
#   - A Python env with the SynthesizeMe package (built against dspy 2.6.12; the
#     workarounds in common.py make it run under dspy 3.x):
#         pip install SynthesizeMe pandas python-dotenv platformdirs
#     Point PYBIN at it (default: python).
#   - One or more vLLM servers of the judge, e.g. on ports 8001-8004:
#         vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8001 --max-model-len 32768
#   - data/XRec_<ds>/{ctx,test}_* splits with candidates, scores and embeddings
#     (see baselines/README.md); results/baselines/personalized_rms/ for the subset
#     comparison in `collect` (optional).
#
# Stages
#   prep      per-user preference pairs + 1,000-prompt eval subset   (prep.py)
#   fit       persona judge per eval user, sharded over the servers  (fit.py)
#   bon       knockout Best-of-64, judge = synthme | default          (bon.py)
#   collect   score tournaments, compare on the subset                (collect.py)
#
#   bash baselines/synthesizeme/run.sh prep
#   bash baselines/synthesizeme/run.sh fit
#   bash baselines/synthesizeme/run.sh bon synthme
#   bash baselines/synthesizeme/run.sh bon default
#   bash baselines/synthesizeme/run.sh collect
set -e
PYBIN=${PYBIN:-python}
STAGE=${1:-fit}
SHARDS=${SHARDS:-12}                         # shards per dataset
PORTS=(${PORTS:-8001 8002 8003 8004})
DATASETS=${DATASETS:-"XRec_amazon XRec_yelp XRec_google"}
LOGS=${LOGS:-logs/synthesizeme}
DIR=baselines/synthesizeme
mkdir -p "$LOGS"

case $STAGE in
prep)
  for ds in $DATASETS; do python $DIR/prep.py --dataset $ds; done
  ;;
fit|bon)
  JUDGE=${2:-synthme}
  i=0
  for ds in $DATASETS; do
    for s in $(seq 0 $((SHARDS - 1))); do
      port=${PORTS[$((i % ${#PORTS[@]}))]}
      if [ "$STAGE" = fit ]; then
        nohup $PYBIN $DIR/fit.py --dataset $ds --shard $s --num_shards $SHARDS --port $port \
          --num_workers 4 --num_search_candidates 5 > $LOGS/fit_${ds}_s${s}.log 2>&1 &
      else
        nohup $PYBIN $DIR/bon.py --dataset $ds --judge $JUDGE --shard $s --num_shards $SHARDS \
          --port $port --threads 3 > $LOGS/bon_${JUDGE}_${ds}_s${s}.log 2>&1 &
      fi
      i=$((i + 1))
    done
  done
  echo "launched $i $STAGE shards (resumable; re-run the stage to fill gaps)"
  ;;
collect)
  python $DIR/collect.py
  ;;
*)
  echo "usage: run.sh {prep|fit|bon [synthme|default]|collect}"; exit 1
  ;;
esac
