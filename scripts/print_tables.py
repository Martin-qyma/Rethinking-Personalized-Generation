#!/usr/bin/env python3
"""Print the ranker and guided-generation tables as markdown.

    objectives   ROUGE-L@64 per training objective (results/ranker/<ds>/3M_<loss>.json)
    sizes        ROUGE-L@64 per ranker size, full evaluation split
    guided       guided-generation sensitivity (results/guided/<ds>.json)

    python scripts/print_tables.py objectives
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rethinking import datasets as D
from rethinking.losses import LOSSES
from rethinking.ranker import SIZES


def table(rows, header, fmt=lambda v: f"{v:.4f}"):
    names = list(D.DATASETS)
    print("| " + header + " | " + " | ".join(D.get(n).label for n in names) + " |")
    print("|---" * (len(names) + 1) + "|")
    for label, get in rows:
        cells = []
        for n in names:
            v = get(n)
            cells.append("--" if v is None else fmt(v))
        print(f"| {label} | " + " | ".join(cells) + " |")


def ranker_value(results_dir, name, run):
    p = os.path.join(results_dir, "ranker", name, f"{run}.json")
    return D.load_json(p)["rougeL"][-1] if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table", choices=["objectives", "sizes", "guided"])
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()
    r = args.results_dir

    if args.table == "objectives":
        table([(l, lambda n, l=l: ranker_value(r, n, f"3M_{l}")) for l in LOSSES], "Objective")
    elif args.table == "sizes":
        table([(s, lambda n, s=s: ranker_value(r, n, f"{s}_mse")) for s in SIZES], "Ranker size")
    else:
        def load(n):
            p = os.path.join(r, "guided", f"{n}.json")
            return D.load_json(p) if os.path.exists(p) else {}
        data = {n: load(n) for n in D.DATASETS}
        settings = list(dict.fromkeys(k for v in data.values() for k in v))
        table([(s, lambda n, s=s: data[n].get(s, {}).get("rougeL")) for s in settings], "Setting")
        print()
        table([(s, lambda n, s=s: data[n].get(s, {}).get("steered_frac")) for s in settings],
              "Steered steps", fmt=lambda v: f"{100 * v:.0f}%")


if __name__ == "__main__":
    main()
