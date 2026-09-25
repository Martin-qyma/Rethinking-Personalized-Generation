#!/usr/bin/env python3
"""Score candidate pools with a reward model (generalist baselines or the finetuned comparator).

Writes ``<split>_rmscores_<tag>.shard<i>of<n>.jsonl`` with one ``{"id", "scores"}``
line per prompt (raw scores of the first ``--n_cand`` candidates), so Best-of-N
can be computed afterwards for any N and any metric. Resumable across shards.

Paper populations: the first 1,000 evaluation prompts of each LaMP and XRec
dataset and all LaMP-QA evaluation prompts for the four generalist models
(``--max_examples 1000``); the full evaluation split for the finetuned comparator.

    CUDA_VISIBLE_DEVICES=0 python scripts/score_reward_model.py --rm skywork --dataset LaMP_7
    # finetuned comparator (scripts/finetune_reward_model.py)
    CUDA_VISIBLE_DEVICES=0 python scripts/score_reward_model.py --rm skywork \
        --model_path checkpoints/rm_ft/LaMP_7 --tag skywork_ft --dataset LaMP_7 \
        --max_examples 0 --max_length 2048 --fit_response
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rethinking import datasets as D
from rethinking.reward_models import REWARD_MODELS, RewardModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rm", required=True, choices=list(REWARD_MODELS))
    ap.add_argument("--dataset", required=True, choices=list(D.DATASETS))
    ap.add_argument("--split", default=None, help="default: the dataset's evaluation split")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--model_path", default=None, help="local checkpoint overriding the hub model")
    ap.add_argument("--tag", default=None, help="name in the output file (default: --rm)")
    ap.add_argument("--n_cand", type=int, default=64)
    ap.add_argument("--max_examples", type=int, default=1000, help="0 = all prompts")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--fit_response", action="store_true",
                    help="left-truncate the prompt so the candidate always fits (finetuned comparator)")
    args = ap.parse_args()

    split = args.split or D.get(args.dataset).eval_split
    tag = args.tag or args.rm
    base = D.path(args.data_root, args.dataset, split, f"rmscores_{tag}")
    done = set(D.read_jsonl_scores(base))

    questions = D.load_questions(args.data_root, args.dataset, split)
    if args.max_examples:
        questions = questions[: args.max_examples]
    pools = D.load_samples(args.data_root, args.dataset, split)
    todo = [q for q in questions[args.shard::args.num_shards]
            if str(q["id"]) in pools and str(q["id"]) not in done]
    print(f"[{tag} shard {args.shard}] {len(todo)} prompts to score", flush=True)
    if not todo:
        return

    prompts = D.PromptBuilder(args.dataset)
    rm = RewardModel(args.rm, args.model_path, device=torch.cuda.current_device(),
                     batch_size=args.batch_size, max_length=args.max_length,
                     fit_response=args.fit_response)
    out = f"{base}.shard{args.shard}of{args.num_shards}.jsonl"
    from tqdm import tqdm
    with open(out, "a") as f:
        for q in tqdm(todo, mininterval=30):
            qid = str(q["id"])
            scores = rm.score(prompts.user_text(q), pools[qid][: args.n_cand])
            f.write(json.dumps({"id": qid, "scores": scores}) + "\n")
            f.flush()
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
