#!/usr/bin/env python3
"""Extract the recycled embeddings h_x, h_u, h_y for a split (Sec. 3.2).

Each embedding is the frozen generator's final-layer hidden state at the last
token of a text: the task query (h_x), the profile text that enters the prompt
(h_u; BM25-retrieved items for LaMP / LaMP-QA, the user summary for XRec), and
each of the first ``--max_cand`` candidates (h_y). We obtain them with vLLM's
last-token pooling in a pass separate from generation; a serving stack that
exposes the generator's hidden states can read them off the generation pass.

Writes ``<split>_cand_embeddings.npz`` with ids (P,), question_embs (P, D),
profile_embs (P, D), answer_embs (P, S, D) float16, n_samples (P,).

    CUDA_VISIBLE_DEVICES=0 python scripts/embed.py --dataset LaMP_7 --split dev
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rethinking import datasets as D


def embed(llm, texts, batch=8192):
    texts = [t if t and t.strip() else " " for t in texts]   # vLLM rejects empty inputs
    out = []
    for i in range(0, len(texts), batch):
        out += [o.outputs.embedding for o in llm.embed(texts[i:i + batch])]
    return np.asarray(out, dtype=np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(D.DATASETS))
    ap.add_argument("--split", required=True)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--model", default=D.GENERATOR)
    ap.add_argument("--max_cand", type=int, default=64)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = ap.parse_args()

    from vllm import LLM

    pools = D.load_samples(args.data_root, args.dataset, args.split)
    questions = [q for q in D.load_questions(args.data_root, args.dataset, args.split)
                 if str(q["id"]) in pools]
    ids, q_texts, u_texts, cands, n = [], [], [], [], []
    for q in questions:
        x, u = D.embedding_texts(args.dataset, q)
        c = pools[str(q["id"])][: args.max_cand]
        ids.append(str(q["id"])); q_texts.append(x); u_texts.append(u)
        cands += c; n.append(len(c))

    llm = LLM(model=args.model, runner="pooling", convert="embed", dtype="bfloat16",
              tensor_parallel_size=1, trust_remote_code=True,
              gpu_memory_utilization=args.gpu_memory_utilization)
    print(f"{args.dataset}/{args.split}: {len(ids)} prompts, {len(cands)} candidates", flush=True)
    hx, hu, flat = embed(llm, q_texts), embed(llm, u_texts), embed(llm, cands)

    hy = np.zeros((len(ids), max(n), hx.shape[1]), dtype=np.float16)
    pos = 0
    for i, k in enumerate(n):
        hy[i, :k] = flat[pos:pos + k]
        pos += k
    out = D.path(args.data_root, args.dataset, args.split, "cand_embeddings.npz")
    np.savez_compressed(out, ids=np.array(ids), question_embs=hx, profile_embs=hu,
                        answer_embs=hy, n_samples=np.array(n))
    print(f"saved -> {out}  (dim {hx.shape[1]})")


if __name__ == "__main__":
    main()
