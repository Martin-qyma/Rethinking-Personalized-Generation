#!/usr/bin/env python3
"""Prepare XRec (Amazon, Yelp, Google) for sampling.

Reads the XRec data release (https://github.com/HKUDS/XRec, ``data/<ds>/{trn,tst}.pkl``
DataFrames with uid, iid, title, user_summary, item_summary, explanation) and writes
``data/XRec_<ds>/<split>_{questions,outputs}.json``. The prompt is XRec's explainer
template with its textual user and item profiles, without the collaborative
embedding tokens specific to XRec's own model.

Splits used in the paper:
    test   the full test set (3,000 interactions)                      -> evaluation
    train  the first 3,000 training interactions                        -> ranker training
    ctx    up to 8 training interactions of every test user              -> personalized-RM
           baselines only (per-user preference pairs, App. B.3)

    python scripts/prepare_xrec.py --src_dir /path/to/XRec/data
"""
import argparse
import collections
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rethinking.datasets import save_json

# XRec's per-dataset instruction and item noun (explainer/utils/data_handler.py)
XREC = {
    "amazon": ("Explain why the user would buy with the book within 50 words.", "book"),
    "yelp": ("Explain why the user would enjoy the business within 50 words.", "business"),
    "google": ("Explain why the user would enjoy the business within 50 words.", "business"),
}


def to_questions(df, ds, prefix):
    system, noun = XREC[ds]
    questions, golds = [], []
    for i in range(len(df)):
        r = df.iloc[i]
        qid = f"{prefix}{i}"
        questions.append({
            "id": qid, "uid": int(r["uid"]), "iid": int(r["iid"]), "system": system,
            "input": (f"{noun} name: {r['title']}\nuser profile: {r['user_summary']}\n"
                      f"{noun} profile: {r['item_summary']}"),
            # texts behind the query and user embeddings
            "q_text": f"{noun} name: {r['title']}\n{noun} profile: {r['item_summary']}",
            "u_text": f"user profile: {r['user_summary']}",
        })
        golds.append({"id": qid, "output": str(r["explanation"])})
    return questions, golds


def context_rows(trn, test_uids, k):
    picked = collections.defaultdict(list)
    for row, u in enumerate(trn["uid"].tolist()):
        if u in test_uids and len(picked[u]) < k:
            picked[u].append(row)
    return trn.iloc[[i for u in sorted(picked) for i in picked[u]]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True, help="XRec data dir containing <ds>/{trn,tst}.pkl")
    ap.add_argument("--datasets", nargs="+", default=list(XREC))
    ap.add_argument("--out_root", default="data")
    ap.add_argument("--train_cap", type=int, default=3000)
    ap.add_argument("--context_k", type=int, default=8)
    args = ap.parse_args()

    for ds in args.datasets:
        load = lambda f: pickle.load(open(os.path.join(args.src_dir, ds, f), "rb")).reset_index(drop=True)
        trn, tst = load("trn.pkl"), load("tst.pkl")
        splits = {
            "test": (tst, f"{ds}_"),
            "train": (trn.iloc[:args.train_cap].reset_index(drop=True), f"{ds}_"),
            "ctx": (context_rows(trn, set(tst["uid"].tolist()), args.context_k), f"{ds}_ctx_"),
        }
        for split, (df, prefix) in splits.items():
            q, g = to_questions(df, ds, prefix)
            d = os.path.join(args.out_root, f"XRec_{ds}")
            save_json(q, os.path.join(d, f"{split}_questions.json"))
            save_json({"task": f"XRec_{ds}", "golds": g}, os.path.join(d, f"{split}_outputs.json"))
            print(f"XRec_{ds}/{split}: {len(q)} questions")


if __name__ == "__main__":
    main()
