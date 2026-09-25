#!/usr/bin/env python3
"""Prepare LaMP-QA (Arts, Lifestyle, Society) for sampling.

* Inputs: the LaMP-QA test split (``alireza7/LaMP-QA`` on the Hugging Face Hub).
* Prompt: the benchmark's retrieval-augmented template with the six past
  questions of the asker retrieved by BM25.
* References: LaMP-QA ships rubrics, not reference answers. We use the per-user
  reference answer (``chosen``) released with Personalized-RewardBench
  (``QiyaoMa/Personalized-RewardBench``), whose question ids coincide with the
  LaMP-QA test set.
* Split: the questions are shuffled with seed 0 and split 90/10 into ranker
  training (``train``) and evaluation (``dev``); 77 / 99 / 108 evaluation prompts.

    python scripts/prepare_lampqa.py
"""
import argparse
import os
import random
import sys

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from rank_bm25 import BM25Okapi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rethinking.datasets import save_json

CATEGORIES = ["Art_and_Entertainment", "Lifestyle_and_Personal_Development", "Society_and_Culture"]

SYSTEM = """You are a helpful assistant designed to generate personalized responses to user questions. Your task is to answer a user's question from a post in a personalized way by considering this user's past post questions and detailed descriptions of these questions.
# Your input:
    - The user's current question from a post.
    - The user's past post questions and detailed descriptions of these questions.
# Your task: Answer the user's current question in a personalized way by considering this user's past post questions and detailed descriptions of these questions, to learn about the user's preferences.
# Your output: Generate a personalized answer to the user's current question considering this user's past post questions and detailed descriptions to learn about their preferences."""

USER = """
# Past post questions and detailed descriptions of these questions:
{profile}
# Current post question:
{question}
"""


def bm25_topk(question, profile, k):
    texts = [p.get("text", "") for p in profile]
    if len(texts) <= k:
        return texts
    return BM25Okapi([t.split() for t in texts]).get_top_n(question.split(), texts, n=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="data")
    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--train_frac", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for cat in CATEGORIES:
        qa = load_dataset("alireza7/LaMP-QA", data_dir=f"data/{cat}", split="test")
        prb = pd.read_parquet(hf_hub_download("QiyaoMa/Personalized-RewardBench",
                                              f"{cat}/test-00000-of-00001.parquet", repo_type="dataset"))
        reference = dict(zip(prb["id"].astype(str), prb["chosen"].astype(str)))

        questions = []
        for e in qa:
            qid = str(e["id"])
            assert qid in reference, f"{cat}: question {qid} has no Personalized-RewardBench reference"
            profile = "\n\n".join(bm25_topk(e["question"], e["profile"] or [], args.top_k))
            questions.append({"id": qid, "system": SYSTEM,
                              "input": USER.format(profile=profile, question=e["question"]),
                              "q_text": e["question"].strip(), "u_text": profile.strip(),
                              "history_size": len(e["profile"] or [])})

        ids = [q["id"] for q in questions]
        rng = random.Random(args.seed)
        shuffled = ids[:]
        rng.shuffle(shuffled)
        n_train = int(len(shuffled) * args.train_frac)
        parts = {"train": set(shuffled[:n_train]), "dev": set(shuffled[n_train:])}

        name = f"LaMPQA_{cat}"
        d = os.path.join(args.out_root, name)
        for split, keep in parts.items():
            qs = [q for q in questions if q["id"] in keep]
            save_json(qs, os.path.join(d, f"{split}_questions.json"))
            save_json({"task": name, "golds": [{"id": q["id"], "output": reference[q["id"]]} for q in qs]},
                      os.path.join(d, f"{split}_outputs.json"))
            print(f"{name}/{split}: {len(qs)} questions")


if __name__ == "__main__":
    main()
