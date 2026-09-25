#!/usr/bin/env python3
"""Cross-metric generalization (BLEU of the Best-of-64 selection).

Every selector picks the argmax of its scores over the first N candidates of a
pool; the selected candidate is then evaluated with a metric the rankers were
not tuned toward (BLEU by default). Our ranker is the ROUGE-L-trained default
run (``results/ranker/<dataset>/3M_mse.json``); the four generalist reward
models use their raw scores ``<split>_rmscores_<rm>.jsonl``.

Prompt populations (``--population``):

    paper   (default, as in the paper table) each selector on the prompts it was
            evaluated on: the reward models on the prompts they scored (the first
            1,000 evaluation prompts; all of them for LaMP-QA), our ranker on the
            full evaluation split (prompts with a non-constant ROUGE-L pool).
    common  every selector on the prompts shared by our ranker and all reward
            models, so all rows of a column average over the same prompts.

    python analysis/bleu_transfer.py --data_root data --results_dir results
    python analysis/bleu_transfer.py --population common --metric rougeL
"""
import argparse
import os

import numpy as np

from common import ALL_DATASETS, D, column_labels, load_selection, markdown_table

RMS = {"skywork": "Skywork", "internlm": "InternLM2", "urm": "URM", "armorm": "ArmoRM"}


def evaluate(selected, scores, metric, n):
    """Mean metric of the selected candidate, plus oracle and random references."""
    sel, orc, rnd = [], [], []
    for qid, b in selected.items():
        v = np.asarray(scores[qid][metric][:n], dtype=float)
        sel.append(v[min(b, len(v) - 1)]); orc.append(v.max()); rnd.append(v.mean())
    return dict(value=float(np.mean(sel)), oracle=float(np.mean(orc)),
                random=float(np.mean(rnd)), n=len(sel))


def dataset_selections(args, name):
    """{selector: {question id: selected index}} on the requested population."""
    split = D.get(name).eval_split
    scores = D.load_scores(args.data_root, name, split)
    sel = {"ours": {q: b for q, b in load_selection(args.results_dir, name, args.run, args.n).items()
                    if q in scores}}
    for rm in args.rms:
        rm_scores = D.load_rm_scores(args.data_root, name, split, rm)
        sel[rm] = {q: int(np.argmax(s[: args.n])) for q, s in rm_scores.items() if q in scores}
    if args.population == "common":
        shared = set.intersection(*(set(s) for s in sel.values()))
        sel = {k: {q: b for q, b in s.items() if q in shared} for k, s in sel.items()}
    return sel, scores


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    ap.add_argument("--run", default="3M_mse", help="ranker run name under results/ranker/<dataset>/")
    ap.add_argument("--rms", nargs="+", default=list(RMS), choices=list(RMS))
    ap.add_argument("--metric", default="bleu", choices=["bleu", "rougeL", "rouge1"])
    ap.add_argument("--n", type=int, default=64, help="Best-of-N pool size")
    ap.add_argument("--population", default="paper", choices=["paper", "common"])
    ap.add_argument("--out", default=None, help="JSON output (default: <results_dir>/analysis/"
                                                "bleu_transfer_<metric>_<population>.json)")
    args = ap.parse_args()

    table = {}
    for name in args.datasets:
        sel, scores = dataset_selections(args, name)
        table[name] = {k: evaluate(s, scores, args.metric, args.n) for k, s in sel.items() if s}

    methods = ["ours"] + args.rms
    rows = []
    for m in methods:
        cells = []
        for name in args.datasets:
            r = table[name].get(m)
            if r is None:
                cells.append("--"); continue
            best = max(v["value"] for v in table[name].values())
            cells.append(f"**{r['value']:.4f}**" if r["value"] == best else f"{r['value']:.4f}")
        rows.append([{"ours": "Ours", **RMS}[m]] + cells)
    for ref in ("oracle", "random"):
        rows.append([f"{ref.capitalize()} (ours' prompts)"]
                    + [f"{table[n]['ours'][ref]:.4f}" if "ours" in table[n] else "--" for n in args.datasets])
    print(f"## {args.metric} of the Best-of-{args.n} selection (population: {args.population})\n")
    print(markdown_table(["Method"] + column_labels(args.datasets), rows))
    print("\nPrompts per selector: " + "; ".join(
        f"{D.get(n).label} " + "/".join(str(table[n][m]["n"]) if m in table[n] else "0" for m in methods)
        for n in args.datasets) + f"  ({'/'.join(methods)})")

    out = args.out or os.path.join(args.results_dir, "analysis",
                                   f"bleu_transfer_{args.metric}_{args.population}.json")
    D.save_json(table, out, indent=1)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
