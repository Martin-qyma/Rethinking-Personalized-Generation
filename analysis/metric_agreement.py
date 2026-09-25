#!/usr/bin/env python3
"""Agreement between ROUGE-L and BLEU at the candidate level (metric agreement table).

For each evaluation prompt the two metrics are compared over the first N
candidates of its pool:

    within-pool Spearman / Pearson   correlation over the N candidates, averaged over prompts
    same argmax                      share of pools where both metrics pick the same best
                                     candidate (first index on ties)
    across-prompt Pearson            correlation of the per-prompt mean scores

Pools that are constant (std < 1e-9) under either metric, i.e. with no
ordering to compare, are excluded from every statistic. Only the label file
``<split>_samples_scores.json`` is needed.

    python analysis/metric_agreement.py --data_root data
"""
import argparse
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr

from common import ALL_DATASETS, D, column_labels, markdown_table


def agreement(scores, a="rougeL", b="bleu", n=64):
    sp, pe, same, mean_a, mean_b = [], [], [], [], []
    for s in scores.values():
        x = np.asarray(s[a][:n], dtype=float)
        y = np.asarray(s[b][:n], dtype=float)
        if x.std() < 1e-9 or y.std() < 1e-9:   # constant up to float round-off
            continue
        sp.append(spearmanr(x, y)[0]); pe.append(pearsonr(x, y)[0])
        same.append(int(np.argmax(x) == np.argmax(y)))
        mean_a.append(x.mean()); mean_b.append(y.mean())
    return dict(within_spearman=float(np.mean(sp)), within_pearson=float(np.mean(pe)),
                same_argmax=float(np.mean(same)), across_pearson=float(pearsonr(mean_a, mean_b)[0]),
                n=len(sp), n_total=len(scores))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    ap.add_argument("--n", type=int, default=64, help="candidates per pool")
    ap.add_argument("--out", default=None, help="JSON output (default: <results_dir>/analysis/metric_agreement.json)")
    args = ap.parse_args()

    res = {name: agreement(D.load_scores(args.data_root, name, D.get(name).eval_split), n=args.n)
           for name in args.datasets}
    rows = [["Within-pool Spearman"] + [f"{res[n]['within_spearman']:.3f}" for n in args.datasets],
            ["Within-pool Pearson"] + [f"{res[n]['within_pearson']:.3f}" for n in args.datasets],
            ["Same argmax"] + [f"{100 * res[n]['same_argmax']:.0f}%" for n in args.datasets],
            ["Across-prompt Pearson"] + [f"{res[n]['across_pearson']:.3f}" for n in args.datasets],
            ["Prompts (used / total)"] + [f"{res[n]['n']}/{res[n]['n_total']}" for n in args.datasets]]
    print(f"## ROUGE-L vs BLEU agreement over the first {args.n} candidates\n")
    print(markdown_table(["Statistic"] + column_labels(args.datasets), rows))

    out = args.out or os.path.join(args.results_dir, "analysis", "metric_agreement.json")
    D.save_json(res, out, indent=1)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
