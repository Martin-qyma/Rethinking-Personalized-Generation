#!/usr/bin/env python3
"""Mean-pooled joint (prompt, candidate) embeddings for the GPO baseline.

GPO (Zhao et al., ICLR 2024, App. D) represents a (query, response) option by the
AVERAGE hidden state over all tokens of their concatenation. This embeds
``"<system>\\n\\n<input>\\n\\n<candidate>"`` with the frozen generator and MEAN pooling
and writes

    <data_root>/<dataset>/<split>_meanpool_embeddings.npz
        ids (P,), cand_embs (P, S, D) fp16, n_samples (P,)

(Our ranker and the other baselines use the last-token states written by
scripts/embed.py instead.)

    python baselines/embed_meanpool.py --dataset XRec_amazon --split ctx --max_cand 16
    python baselines/embed_meanpool.py --dataset XRec_amazon --split test --max_cand 64
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rethinking import datasets as D


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=[n for n, d in D.DATASETS.items() if d.family == "xrec"])
    ap.add_argument("--split", required=True, choices=["ctx", "test"])
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--model", default=D.GENERATOR)
    ap.add_argument("--max_cand", type=int, default=64)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--batch", type=int, default=4096)
    args = ap.parse_args()

    out_path = D.path(args.data_root, args.dataset, args.split, "meanpool_embeddings.npz")
    if os.path.exists(out_path):
        print(f"[skip] {out_path} exists")
        return
    samples = D.load_samples(args.data_root, args.dataset, args.split)
    questions = [q for q in D.load_questions(args.data_root, args.dataset, args.split)
                 if str(q["id"]) in samples]

    ids, texts, n_samples = [], [], []
    for q in questions:
        prompt = (q.get("system", "") + "\n\n" + q["input"]).strip()
        cands = samples[str(q["id"])][:args.max_cand]
        ids.append(str(q["id"]))
        n_samples.append(len(cands))
        texts.extend(prompt + "\n\n" + c for c in cands)

    from vllm import LLM
    from vllm.config import PoolerConfig
    print(f"Embedding {len(texts)} (prompt, candidate) sequences, mean pooled ...", flush=True)
    llm = LLM(model=args.model, runner="pooling", convert="embed", tensor_parallel_size=1,
              trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_memory_utilization,
              pooler_config=PoolerConfig(pooling_type="MEAN"))
    flat = []
    for i in range(0, len(texts), args.batch):
        chunk = [t if t.strip() else " " for t in texts[i:i + args.batch]]
        flat.extend(o.outputs.embedding for o in llm.embed(chunk))
        print(f"  {min(i + args.batch, len(texts))}/{len(texts)}", flush=True)
    flat = np.array(flat, dtype=np.float16)

    cand = np.zeros((len(ids), max(n_samples), flat.shape[1]), np.float16)
    pos = 0
    for i, n in enumerate(n_samples):
        cand[i, :n] = flat[pos:pos + n]
        pos += n
    np.savez_compressed(out_path, ids=np.array(ids), cand_embs=cand, n_samples=np.array(n_samples, np.int64))
    print(f"saved -> {out_path} {cand.shape}")


if __name__ == "__main__":
    main()
