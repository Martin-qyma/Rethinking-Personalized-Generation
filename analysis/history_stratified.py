#!/usr/bin/env python3
"""Performance stratified by available history (stratified history table).

Buckets the evaluation prompts by their user's history size (definitions in
``common.history_sizes``) and reports the Best-of-N ROUGE-L of the default
ranker within each bucket, so differences across buckets reflect the history
available to the pipeline rather than a change of model. XRec users have at
most 22 recorded interactions and fall into the two lowest buckets. Cells with
fewer than ``--min_support`` prompts are marked with a dagger.

Inputs: the ranker's per-question selections
(``<results_dir>/ranker/<dataset>/<run>.json``), the label file
``<split>_samples_scores.json``, and the question files behind the history sizes.

    python analysis/history_stratified.py --data_root data --results_dir results
"""
import argparse
import os

import numpy as np

from common import ALL_DATASETS, D, column_labels, history_sizes, load_selection, markdown_table

BUCKETS = [(0, 10, "<10"), (10, 25, "10-24"), (25, 50, "25-49"), (50, 100, "50-99"), (100, None, "100+")]


def bucket_of(h):
    for lo, hi, label in BUCKETS:
        if h >= lo and (hi is None or h < hi):
            return label


def stratify(selection, scores, sizes, metric="rougeL", n=64):
    groups = {label: [] for _, _, label in BUCKETS}
    for qid, b in selection.items():
        v = np.asarray(scores[qid][metric][:n], dtype=float)
        groups[bucket_of(sizes[qid])].append((v[b], v.max(), v.mean()))
    out = {}
    for label, g in groups.items():
        if not g:
            out[label] = None
            continue
        sel, orc, rnd = (float(np.mean(x)) for x in zip(*g))
        out[label] = dict(value=sel, oracle=orc, random=rnd, n=len(g),
                          headroom=(sel - rnd) / max(orc - rnd, 1e-9))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    ap.add_argument("--run", default="3M_mse", help="ranker run name under results/ranker/<dataset>/")
    ap.add_argument("--metric", default="rougeL", choices=["rougeL", "rouge1", "bleu"])
    ap.add_argument("--n", type=int, default=64, help="Best-of-N pool size")
    ap.add_argument("--min_support", type=int, default=10, help="dagger below this many prompts")
    ap.add_argument("--out", default=None, help="JSON output (default: <results_dir>/analysis/history_stratified.json)")
    args = ap.parse_args()

    res = {}
    for name in args.datasets:
        split = D.get(name).eval_split
        per_q, _ = history_sizes(args.data_root, name)
        scores = D.load_scores(args.data_root, name, split)
        selection = load_selection(args.results_dir, name, args.run, args.n)
        missing = [q for q in selection if q not in per_q]
        if missing:
            raise KeyError(f"{name}: {len(missing)} evaluated prompts without a history size")
        res[name] = stratify(selection, scores, per_q, args.metric, args.n)

    def cell(r, with_n=False):
        if r is None:
            return "--"
        s = f"{r['value']:.4f}" + ("†" if r["n"] < args.min_support else "")
        return s + (f" ({100 * r['headroom']:.0f}%, n={r['n']})" if with_n else "")

    header = ["History size"] + column_labels(args.datasets)
    print(f"## Best-of-{args.n} {args.metric} by user history size (run {args.run})\n")
    print(markdown_table(header, [[label] + [cell(res[n][label]) for n in args.datasets]
                                  for _, _, label in BUCKETS]))
    print(f"\n† fewer than {args.min_support} prompts.\n\n### With headroom captured and support\n")
    print(markdown_table(header, [[label] + [cell(res[n][label], True) for n in args.datasets]
                                  for _, _, label in BUCKETS]))

    out = args.out or os.path.join(args.results_dir, "analysis", "history_stratified.json")
    D.save_json(res, out, indent=1)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
