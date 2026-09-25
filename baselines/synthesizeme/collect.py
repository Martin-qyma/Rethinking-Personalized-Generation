#!/usr/bin/env python3
"""Score the SynthesizeMe tournaments and compare all methods on the same subset.

For each XRec dataset, reads ``<prep_dir>/<dataset>/bon_{synthme,default}.jsonl``
(bon.py) and, if present, the per-question selections of the trained reward models
(results/baselines/personalized_rms/<dataset>.json) and GPO, and reports Best-of-N
ROUGE-L on the 1,000-prompt evaluation subset, with oracle / random references over
the same first-64 pools.

Writes ``<prep_dir>/results.json`` (consumed by baselines/collect_table.py).

    python baselines/synthesizeme/collect.py
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from rethinking import datasets as D
from rethinking.datasets import KS

XREC = [n for n, d in D.DATASETS.items() if d.family == "xrec"]


def curve(sel, ids, scores, metric):
    """Mean label of the selected candidate at every N, over ``ids``."""
    return {str(k): float(np.mean([scores[i][metric][sel[i][str(k)]] for i in ids])) for k in KS}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--prep_dir", default="results/baselines/synthesizeme")
    ap.add_argument("--rm_dirs", nargs="*", default=["results/baselines/personalized_rms", "results/baselines/gpo"])
    ap.add_argument("--datasets", nargs="+", default=XREC)
    args = ap.parse_args()

    out = {}
    for name in args.datasets:
        d = os.path.join(args.prep_dir, name)
        if not os.path.exists(os.path.join(d, "eval_subset.json")):
            continue
        ids = D.load_json(os.path.join(d, "eval_subset.json"))["ids"]
        scores = D.load_scores(args.data_root, name, "test")
        res, n_pool = {}, {}
        for judge in ["synthme", "default"]:
            p = os.path.join(d, f"bon_{judge}.jsonl")
            if not os.path.exists(p):
                continue
            with open(p) as f:
                recs = {r["id"]: r for r in map(json.loads, filter(str.strip, f))}
            done = [i for i in ids if i in recs]
            sel = {i: recs[i]["sel"] for i in done}
            n_pool.update({i: recs[i]["n_cands"] for i in done})
            res[judge] = dict(n=len(done), rougeL=curve(sel, done, scores, "rougeL"),
                              rouge1=curve(sel, done, scores, "rouge1"),
                              ties=sum(recs[i]["ties"] for i in done),
                              matches=sum(recs[i]["matches"] for i in done),
                              fallbacks=sum(int(recs[i].get("fallback", False)) for i in done))
        # trained reward models (full-test runs) restricted to the same prompts
        for rd in args.rm_dirs:
            p = os.path.join(rd, f"{name}.json")
            if not os.path.exists(p):
                continue
            for method, r in D.load_json(p).items():
                pq = r["full"]["per_q"]
                pos = {qid: j for j, qid in enumerate(pq["ids"])}
                sub = [i for i in ids if i in pos]
                sel = {i: {k: v[pos[i]] for k, v in pq["sel_idx"].items()} for i in sub}
                res[method] = dict(n=len(sub), rougeL=curve(sel, sub, scores, "rougeL"),
                                   rouge1=curve(sel, sub, scores, "rouge1"))
        first = [scores[i]["rougeL"][:n_pool.get(i, 64)] for i in ids]
        res["oracle_rougeL"] = float(np.mean([max(v) for v in first]))
        res["mean_rougeL"] = float(np.mean([np.mean(v) for v in first]))
        out[name] = res

        print(f"## {name} (subset of {len(ids)} prompts; oracle {res['oracle_rougeL']:.4f}, "
              f"random {res['mean_rougeL']:.4f})")
        for m, r in res.items():
            if isinstance(r, dict):
                v = r["rougeL"]["64"]
                cap = 100 * (v - res["mean_rougeL"]) / (res["oracle_rougeL"] - res["mean_rougeL"])
                extra = f"  ties {r['ties']}/{r['matches']}, fallback prompts {r['fallbacks']}" if "ties" in r else ""
                print(f"  {m:8s} n={r['n']:4d}  ROUGE-L@64 {v:.4f} ({cap:.0f}% of headroom){extra}")
    D.save_json(out, os.path.join(args.prep_dir, "results.json"), indent=1)


if __name__ == "__main__":
    main()
