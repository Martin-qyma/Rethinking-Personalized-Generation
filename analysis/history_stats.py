#!/usr/bin/env python3
"""How much history users have (history statistics table).

Summarizes the number of historical interactions per user (median, mean, 10th
and 90th percentiles, maximum) and lists the profile text that actually enters
the generation prompt and h_u. History sizes are defined per family in
``common.history_sizes``:

    LaMP     len(profile) of each evaluation question (one user per question)
    LaMP-QA  past questions of the asker in the benchmark profile, over all
             questions of the category (ranker training + evaluation splits);
             read from Personalized-RewardBench on the Hugging Face Hub
    XRec     interactions of each distinct test user in our ctx + test splits
             (a lower bound; the benchmark reports 18--25 per user)

``--population eval`` instead reports one value per evaluation prompt for every
dataset.

    python analysis/history_stats.py --data_root data
"""
import argparse
import os

import numpy as np

from common import ALL_DATASETS, D, history_sizes, history_unit, markdown_table, profile_description


def summarize(v):
    v = np.asarray(v)
    return dict(n=int(len(v)), median=float(np.median(v)), mean=float(v.mean()),
                p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)),
                min=int(v.min()), max=int(v.max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    ap.add_argument("--population", default="users", choices=["users", "eval"],
                    help="users: one value per benchmark user (paper); eval: one per evaluation prompt")
    ap.add_argument("--out", default=None, help="JSON output (default: <results_dir>/analysis/history_stats.json)")
    args = ap.parse_args()

    res, rows = {}, []
    for name in args.datasets:
        per_q, per_user = history_sizes(args.data_root, name)
        s = summarize(per_user if args.population == "users" else list(per_q.values()))
        res[name] = dict(**s, unit=history_unit(name), profile_text=profile_description(name))
        rows.append([D.get(name).label, res[name]["unit"], s["n"], f"{s['median']:.0f}", f"{s['mean']:.1f}",
                     f"{s['p10']:.0f}", f"{s['p90']:.0f}", s["max"], res[name]["profile_text"]])
    print(f"## History size per user (population: {args.population})\n")
    print(markdown_table(["Dataset", "History unit", "n", "Median", "Mean", "p10", "p90", "Max",
                          "Profile text entering the prompt and h_u"], rows))

    out = args.out or os.path.join(args.results_dir, "analysis", "history_stats.json")
    D.save_json(res, out, indent=1)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
