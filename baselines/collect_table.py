#!/usr/bin/env python3
"""Print Table B.3 (personalized reward-model baselines, BoN ROUGE-L at N=64) from
the results of run_personalized_rms.py, gpo.py and synthesizeme/collect.py.

Ours / VPL / PAL / PReF / LoRe are evaluated on every test interaction of users with
at least one context pair, GPO on those of users with at least two ctx interactions,
and SynthesizeMe (and its generic-judge control) on the 1,000-prompt subset.
``--detail`` adds, per dataset, the share of oracle headroom each method captures and
the "swap" diagnostic: the change in ROUGE-L@64 when every user is given another
user's parameters or context set.

    python baselines/collect_table.py
    python baselines/collect_table.py --detail
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rethinking import datasets as D
from rethinking.pools import headroom_captured

COLUMNS = [("ours", "Ours"), ("vpl", "VPL"), ("pal", "PAL"), ("pref", "PReF"), ("lore", "LoRe"),
           ("gpo", "GPO"), ("synthme", "SynthesizeMe")]
XREC = [n for n, d in D.DATASETS.items() if d.family == "xrec"]


def load(results_dir, name):
    rows = {}
    for sub in ("personalized_rms", "gpo"):
        p = os.path.join(results_dir, sub, f"{name}.json")
        if os.path.exists(p):
            for m, r in D.load_json(p).items():
                full, sw = r["full"], r.get("swapped")
                rows[m] = dict(v=full["rougeL"][-1], n=full["n"],
                               cap=headroom_captured(full["rougeL"][-1], full["mean_rougeL"], full["oracle_rougeL"]),
                               swap=None if sw is None else sw["rougeL"][-1] - full["rougeL"][-1])
    p = os.path.join(results_dir, "synthesizeme", "results.json")
    s = D.load_json(p).get(name, {}) if os.path.exists(p) else {}
    for m in ("synthme", "default"):
        if m in s:
            v = s[m]["rougeL"]["64"]
            rows[m] = dict(v=v, n=s[m]["n"], swap=None,
                           cap=headroom_captured(v, s["mean_rougeL"], s["oracle_rougeL"]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="results/baselines")
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()

    res = {name: load(args.results_dir, name) for name in XREC}
    print("| Dataset | " + " | ".join(lbl for _, lbl in COLUMNS) + " |")
    print("|---" * (len(COLUMNS) + 1) + "|")
    for name in XREC:
        cells = [f"{res[name][m]['v']:.4f}" if m in res[name] else "-" for m, _ in COLUMNS]
        print(f"| {D.get(name).label} | " + " | ".join(cells) + " |")
    if not args.detail:
        return
    for name in XREC:
        print(f"\n{D.get(name).label}:  method  ROUGE-L@64  headroom  swap delta  n")
        for m, lbl in COLUMNS + [("default", "Generic judge")]:
            if m in res[name]:
                r = res[name][m]
                sw = f"{r['swap']:+.4f}" if r["swap"] is not None else "   n/a"
                print(f"  {lbl:14s} {r['v']:.4f}  {100 * r['cap']:4.0f}%  {sw}  {r['n']}")


if __name__ == "__main__":
    main()
