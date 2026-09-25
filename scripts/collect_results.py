#!/usr/bin/env python3
"""Assemble the numbers behind the paper figures into ``results/figdata.json``.

Per dataset:
* Best-of-N curves (Figures 1, 3 and appendix counterparts), on the prompts
  scored by all four generalist reward models: oracle (metric-selected), random
  (mean candidate), each reward model, the best reward model at each N, our
  default ranker (``results/ranker/<dataset>/3M_mse.json``), and the ranking-guided
  generation value (``results/guided/<dataset>.json``, setting "default").
* Size scaling (Figure 4 and appendix counterpart): Best-of-64 of the 1M/3M/10M/30M
  rankers and of the finetuned 8B reward model on the prompts the latter scored.

Also prints the headroom-captured summary quoted in Sec. 4.2 / 4.3.

    python scripts/collect_results.py
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rethinking import datasets as D
from rethinking.datasets import KS
from rethinking.pools import headroom_captured
from rethinking.ranker import SIZES

GENERALIST_RMS = ["skywork", "internlm", "urm", "armorm"]


def ours_curve(run, labels, ids, metric):
    """Curve of a ranker run restricted to ``ids``, from its per-question selections."""
    pos = {q: j for j, q in enumerate(run["per_q"]["ids"])}
    common = [i for i in ids if i in pos]
    curve = [float(np.mean([labels[i][metric][:64][run["per_q"]["sel_idx"][str(k)][pos[i]]]
                            for i in common])) for k in KS]
    return curve, len(common)


def selector_curve(scores, labels, ids, metric):
    vals = {k: [] for k in KS}
    for i in ids:
        v, s = np.array(labels[i][metric][:64]), np.array(scores[i][:64])
        for k in KS:
            vals[k].append(v[int(np.argmax(s[:min(k, len(s), len(v))]))])
    return [float(np.mean(vals[k])) for k in KS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--ranker_run", default="3M_mse")
    ap.add_argument("--ft_tag", default="skywork_ft", help="rmscores tag of the finetuned 8B comparator")
    args = ap.parse_args()

    fig = {}
    for name, ds in D.DATASETS.items():
        split = ds.eval_split
        labels = D.load_scores(args.data_root, name, split)
        res = {"label": ds.label}

        # --- Best-of-N curves on the reward-model population ----------------
        rms = {rm: D.load_rm_scores(args.data_root, name, split, rm) for rm in GENERALIST_RMS}
        ids = sorted(set(labels).intersection(*[set(v) for v in rms.values()]))
        run_path = os.path.join(args.results_dir, "ranker", name, f"{args.ranker_run}.json")
        run = D.load_json(run_path) if os.path.exists(run_path) else None
        if ids:
            res["n"] = len(ids)
            for metric in ("rougeL", "rouge1"):
                m = {"oracle": [float(np.mean([max(labels[i][metric][:k]) for i in ids])) for k in KS],
                     "random": float(np.mean([np.mean(labels[i][metric][:64]) for i in ids]))}
                for rm in GENERALIST_RMS:
                    m[rm] = selector_curve(rms[rm], labels, ids, metric)
                m["best_rm"] = list(np.max([m[rm] for rm in GENERALIST_RMS], axis=0))
                if run:
                    m["ours"], res["ours_n"] = ours_curve(run, labels, ids, metric)
                res[metric] = m
        guided = os.path.join(args.results_dir, "guided", f"{name}.json")
        if os.path.exists(guided):
            res["guided_rougeL"] = D.load_json(guided)["default"]["rougeL"]

        # --- size scaling vs the finetuned 8B reward model ------------------
        sweep = {s: D.load_json(p) for s in SIZES
                 if os.path.exists(p := os.path.join(args.results_dir, "ranker", name, f"{s}_mse.json"))}
        ft = D.load_rm_scores(args.data_root, name, split, args.ft_tag)
        if sweep:
            pop = sweep[max(sweep, key=lambda s: list(SIZES).index(s))]["per_q"]["ids"]
            if ft:
                pop = [i for i in pop if i in ft and len(ft[i]) >= 64]
            res["scaling"] = {"n": len(pop),
                              "mlp": {s: ours_curve(r, labels, pop, "rougeL")[0][-1] for s, r in sweep.items()},
                              "oracle": float(np.mean([max(labels[i]["rougeL"][:64]) for i in pop])),
                              "random": float(np.mean([np.mean(labels[i]["rougeL"][:64]) for i in pop]))}
            if ft:
                res["scaling"]["rm8b"] = selector_curve(ft, labels, pop, "rougeL")[-1]
        fig[name] = res

    out = os.path.join(args.results_dir, "figdata.json")
    D.save_json(fig, out, indent=1)

    print("| dataset | n | oracle@64 | best RM@64 | ours@64 | guided | best RM captures | ours captures |")
    print("|---|---|---|---|---|---|---|---|")
    for name, r in fig.items():
        if "rougeL" not in r:
            continue
        m = r["rougeL"]
        base, orc = m["oracle"][0], m["oracle"][-1]      # headroom over the single-sample point
        ours = m.get("ours", [float("nan")])[-1]
        print(f"| {r['label']} | {r['n']} | {orc:.4f} | {m['best_rm'][-1]:.4f} | {ours:.4f} | "
              f"{r.get('guided_rougeL', float('nan')):.4f} | "
              f"{100 * headroom_captured(m['best_rm'][-1], base, orc):.0f}% | "
              f"{100 * headroom_captured(ours, base, orc):.0f}% |")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
