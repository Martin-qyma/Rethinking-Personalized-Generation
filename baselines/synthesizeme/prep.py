#!/usr/bin/env python3
"""Per-user preference data and the evaluation subset for the SynthesizeMe baseline.

Uses exactly the context pairs of baselines/run_personalized_rms.py (same mining,
same order, same cap of 16 pairs per user), converted to the text format the
SynthesizeMe package expects:

    {"context": [{role, content}, ...], "chosen": {role, content},
     "rejected": {role, content}, "flip": bool}

and samples the 1,000-prompt evaluation subset (seed 0) from the same test
population (users with at least one context pair). The LLM tournament is too
expensive for the full test set; collect.py reports every method on this subset.

Writes ``<out_dir>/<dataset>/{user_prefs.json, eval_subset.json}``.

    python baselines/synthesizeme/prep.py --dataset XRec_amazon
"""
import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from baselines.run_personalized_rms import MAX_PAIRS, METRIC, mine_pairs
from rethinking import datasets as D

FLIP_SEED = 42


def load_split(data_root, name, split):
    npz = np.load(D.path(data_root, name, split, "cand_embeddings.npz"), allow_pickle=True)
    questions = {str(q["id"]): q for q in D.load_questions(data_root, name, split)}
    return dict(ids=[str(x) for x in npz["ids"]], n=npz["n_samples"].astype(np.int64),
                q=questions, uid={k: int(q["uid"]) for k, q in questions.items()},
                scores=D.load_scores(data_root, name, split),
                samples=D.load_samples(data_root, name, split))


def conversation(q):
    return [{"role": "system", "content": q["system"]}, {"role": "user", "content": q["input"]}]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=[n for n, d in D.DATASETS.items() if d.family == "xrec"])
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results/baselines/synthesizeme")
    args = ap.parse_args()

    ctx = load_split(args.data_root, args.dataset, "ctx")
    tst = load_split(args.data_root, args.dataset, "test")
    by_user = mine_pairs(ctx)
    users = sorted(by_user)
    P = min(MAX_PAIRS, max(len(p) for p in by_user.values()))

    rng = random.Random(FLIP_SEED)
    prefs = {}
    for u in users:
        rows = []
        for i, a, b in by_user[u][:P]:
            cid = ctx["ids"][i]
            cands, r = ctx["samples"][cid], ctx["scores"][cid][METRIC]
            rows.append({
                "context": conversation(ctx["q"][cid]),
                "chosen": {"role": "assistant", "content": cands[a]},
                "rejected": {"role": "assistant", "content": cands[b]},
                "flip": rng.random() < 0.5,
                "meta": {"qid": cid, "chosen_idx": a, "rejected_idx": b,
                         "rougeL_chosen": float(r[a]), "rougeL_rejected": float(r[b])},
            })
        prefs[str(u)] = rows

    uset = set(users)
    pop = [tid for tid in tst["ids"] if tid in tst["scores"] and tst["uid"].get(tid) in uset]
    pos = {tid: k for k, tid in enumerate(pop)}
    eval_ids = sorted(random.Random(args.seed).sample(pop, min(args.n_eval, len(pop))), key=pos.get)
    eval_uid = {tid: tst["uid"][tid] for tid in eval_ids}
    eval_users = sorted(set(eval_uid.values()))

    d = os.path.join(args.out_dir, args.dataset)
    D.save_json(prefs, os.path.join(d, "user_prefs.json"))
    D.save_json({"ids": eval_ids, "uid": eval_uid, "population": len(pop), "seed": args.seed},
                os.path.join(d, "eval_subset.json"))
    npairs = [len(prefs[str(u)]) for u in eval_users]
    print(f"{args.dataset}: users with pairs {len(users)} (P={P}); test population {len(pop)}; "
          f"eval subset {len(eval_ids)} prompts / {len(eval_users)} users; pairs per eval user "
          f"median {int(np.median(npairs))} min {min(npairs)} max {max(npairs)} -> {d}")


if __name__ == "__main__":
    main()
