#!/usr/bin/env python3
"""Draw the paper figures from ``results/figdata.json`` (scripts/collect_results.py).

    oracle.png          Fig. 1   oracle vs best generalist RM, averaged per task
    ranking.png         Fig. 3   Best-of-N: our ranker vs four RMs (+ guided line), per task
    scale.png           Fig. 4   ranker size vs finetuned 8B RM at N=64, per task (min/max bars)
    headroom_full.png   Fig. 5   Fig. 1 per dataset
    rank_full.png       Fig. 6   Fig. 3 per dataset
    scale_full.png      Fig. 7   Fig. 4 per dataset

    python scripts/plot_figures.py --out_dir results/figures
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rethinking import datasets as D
from rethinking.datasets import KS, TASK_GROUPS

ORACLE_COLOR, RMLINE_COLOR, FILL_COLOR = "#08306b", "#9ecae1", "#bdd9eb"
OURS_COLOR, GUIDED_COLOR = "#0096C7", "#90E0EF"
RM_STYLE = {"skywork": ("Skywork", "#8c8c8c", "--", "s"),
            "urm": ("URM", "#ababab", (0, (5, 2)), "^"),
            "armorm": ("ArmoRM", "#6b6b6b", "-.", "D"),
            "internlm": ("InternLM2", "#4d4d4d", ":", "X")}
MLP_SHADES = ["#a6cee3", "#6baed6", "#3182bd", "#08519c"]
LINE_COLOR, STAR_COLOR = "#3182bd", "#08306b"
SIZES = ["1M", "3M", "10M", "30M"]
X_MLP, X_8B = [0, 1, 2, 3], 5
CURVE_ORDER = ["Personalized Ranking Model", "Skywork", "URM", "ArmoRM", "InternLM2",
               "Ranking Guided Generation"]
HEADROOM_ORDER = ["Oracle (ceiling)", "SOTA Reward Model", "Remaining Headroom"]
SHORT = {"Short-form Generation": "Short-form", "Long-form QA": "Long-form QA",
         "Explainable Recommendation": "Explainable Rec."}


def row_label(group):
    return f"ROUGE-L: {SHORT[group]}"


def style_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS], fontsize=13)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(alpha=0.3, lw=0.5, ls=":")


def legend_top(fig, ax, order, ncol, size=12):
    h, l = ax.get_legend_handles_labels()
    idx = [l.index(x) for x in order if x in l]
    fig.legend([h[i] for i in idx], [l[i] for i in idx], loc="upper center", ncol=ncol, frameon=True,
               prop={"size": size, "weight": "bold"}, bbox_to_anchor=(0.5, 1.03))


def headroom_panel(ax, oracle, rm):
    ax.fill_between(KS, rm, oracle, color=FILL_COLOR, label="Remaining Headroom", zorder=1)
    ax.plot(KS, rm, "--", color=RMLINE_COLOR, marker="s", ms=5, lw=1.5, label="SOTA Reward Model", zorder=3)
    ax.plot(KS, oracle, "-", color=ORACLE_COLOR, marker="o", ms=6, lw=2.6, label="Oracle (ceiling)", zorder=4)


def curves_panel(ax, d):
    if d.get("guided") is not None:
        ax.axhline(d["guided"], color=GUIDED_COLOR, ls="--", lw=3.2, label="Ranking Guided Generation", zorder=2)
    for rm, (lbl, c, ls, mk) in RM_STYLE.items():
        ax.plot(KS, d[rm], linestyle=ls, color=c, marker=mk, ms=5, lw=1.5, label=lbl, zorder=3)
    if d.get("ours") is not None:
        ax.plot(KS, d["ours"], "-", color=OURS_COLOR, marker="o", ms=6, lw=2.6,
                label="Personalized Ranking Model", zorder=4)


def task_mean(fd, members, key):
    vals = [fd[t]["rougeL"][key] for t in members if key in fd[t].get("rougeL", {})]
    return np.mean(vals, axis=0) if len(vals) == len(members) else None


def grid_axes(title_fn, xlabel, draw, figsize=(15, 10.8)):
    fig, axes = plt.subplots(3, 3, figsize=figsize, squeeze=False)
    for r, (group, members) in enumerate(TASK_GROUPS):
        for c, t in enumerate(members):
            ax = axes[r][c]
            draw(ax, t)
            ax.set_title(D.get(t).label, fontsize=16, fontweight="bold")
            if r == 2:
                ax.set_xlabel(xlabel, fontsize=16, fontweight="bold")
            if c == 0:
                ax.set_ylabel(title_fn(group), fontsize=14, fontweight="bold")
    return fig, axes


def pad_ylim(ax, lo, hi, top=0.17, bottom=0.12):
    span = max(hi - lo, 1e-6)
    ax.set_ylim(lo - bottom * span, hi + top * span)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))


def label_points(ax, xs, ys, colors, fontsize=9.5):
    """Value labels placed above each point unless that would cross the polyline."""
    w_in, h_in = ax.figure.get_size_inches()
    pos = ax.get_position()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    sx, sy = (x1 - x0) / (pos.width * w_in * 72), (y1 - y0) / (pos.height * h_in * 72)
    tw, th = 0.58 * fontsize * 5 * sx, 0.72 * fontsize * sy
    cands = [((0, 9), "center"), ((0, -9 - 0.72 * fontsize), "center"), ((-6, 5), "right"),
             ((6, 5), "left"), ((-6, -5 - 0.72 * fontsize), "right"), ((6, -5 - 0.72 * fontsize), "left")]

    def clear(i, xyt, ha):
        xl = xs[i] + xyt[0] * sx - {"center": tw / 2, "right": tw, "left": 0}[ha]
        xr, yb = xl + tw, ys[i] + xyt[1] * sy
        yt = yb + th
        for j in (i - 1, i + 1):
            if not 0 <= j < len(xs):
                continue
            lo, hi = max(xl, min(xs[i], xs[j])), min(xr, max(xs[i], xs[j]))
            for x in np.linspace(lo, hi, 12) if lo < hi else []:
                y = ys[i] + (ys[j] - ys[i]) * (x - xs[i]) / (xs[j] - xs[i])
                if yb - 0.15 * th <= y <= yt + 0.15 * th:
                    return False
        return True

    for i, (x, y, c) in enumerate(zip(xs, ys, colors)):
        xyt, ha = next(((p, h) for p, h in cands if clear(i, p, h)), cands[0])
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=xyt, ha=ha,
                    fontsize=fontsize, color=c, fontweight="bold", zorder=5)


def delta_label(ax, y_mlp, y_8b, fontsize=11, x=4.0):
    ax.plot([3, X_8B], [y_mlp, y_8b], "--", color="#aaaaaa", lw=1.2, zorder=2)
    ax.annotate(f"$\\Delta$ = {y_8b - y_mlp:+.3f}", xy=(x, y_mlp + (y_8b - y_mlp) * (x - 3) / (X_8B - 3)),
                fontsize=fontsize, color="#555555", ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdata", default="results/figdata.json")
    ap.add_argument("--out_dir", default="results/figures")
    args = ap.parse_args()
    fd = D.load_json(args.figdata)
    os.makedirs(args.out_dir, exist_ok=True)

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(args.out_dir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {args.out_dir}/{name}.png")

    have_curves = all("rougeL" in fd[t] for _, m in TASK_GROUPS for t in m)
    if have_curves:
        # Fig. 1: task-averaged headroom
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.1), squeeze=False)
        for col, (group, members) in enumerate(TASK_GROUPS):
            ax = axes[0][col]
            headroom_panel(ax, task_mean(fd, members, "oracle"), task_mean(fd, members, "best_rm"))
            style_axis(ax)
            ax.set_title(group, fontsize=15, fontweight="bold")
            ax.set_xlabel("Number of Generations", fontsize=16, fontweight="bold")
            if col == 0:
                ax.set_ylabel("ROUGE-L", fontsize=16, fontweight="bold")
        legend_top(fig, axes[0][0], HEADROOM_ORDER, 3)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        save(fig, "oracle")

        # Fig. 5: per-dataset headroom
        def draw_headroom(ax, t):
            headroom_panel(ax, fd[t]["rougeL"]["oracle"], fd[t]["rougeL"]["best_rm"])
            style_axis(ax)
        fig, axes = grid_axes(row_label, "Number of Generations", draw_headroom)
        legend_top(fig, axes[0][0], HEADROOM_ORDER, 3, size=13)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        save(fig, "headroom_full")

        # Fig. 3: task-averaged Best-of-N curves
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.1), squeeze=False)
        for col, (group, members) in enumerate(TASK_GROUPS):
            ax = axes[0][col]
            d = {rm: task_mean(fd, members, rm) for rm in RM_STYLE}
            d["ours"] = task_mean(fd, members, "ours")
            g = [fd[t].get("guided_rougeL") for t in members]
            d["guided"] = float(np.mean(g)) if None not in g else None
            curves_panel(ax, d)
            style_axis(ax)
            ax.set_title(group, fontsize=16, fontweight="bold")
            ax.set_xlabel("Number of Generations", fontsize=16, fontweight="bold")
            if col == 0:
                ax.set_ylabel("ROUGE-L", fontsize=16, fontweight="bold")
        legend_top(fig, axes[0][0], CURVE_ORDER, 6, size=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        save(fig, "ranking")

        # Fig. 6: per-dataset Best-of-N curves
        def draw_curves(ax, t):
            curves_panel(ax, {**fd[t]["rougeL"], "guided": fd[t].get("guided_rougeL")})
            style_axis(ax)
        fig, axes = grid_axes(row_label, "Number of Generations", draw_curves)
        legend_top(fig, axes[0][0], CURVE_ORDER, 6, size=13)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        save(fig, "rank_full")

    if not all("scaling" in fd[t] for _, m in TASK_GROUPS for t in m):
        print("size-sweep results missing for some datasets; skipping scale figures")
        return

    def mlp(t):
        return [fd[t]["scaling"]["mlp"][s] for s in SIZES]

    def star(t):
        return fd[t]["scaling"].get("rm8b")

    # Fig. 7: per-dataset size scaling
    def draw_scale(ax, t):
        vals, ys = mlp(t), list(mlp(t))
        ax.plot(X_MLP, vals, "-", color=LINE_COLOR, lw=2.2, zorder=3)
        for x, y, c in zip(X_MLP, vals, MLP_SHADES):
            ax.plot([x], [y], "o", color=c, ms=9, zorder=4, label="Personalized Ranking Model" if x == 3 else None)
        if star(t) is not None:
            ax.plot([X_8B], [star(t)], "*", color=STAR_COLOR, ms=17, zorder=4, label="Finetuned SOTA Reward Model")
            ax.annotate(f"{star(t):.3f}", (X_8B, star(t)), textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=10, color=STAR_COLOR, fontweight="bold", zorder=5)
            delta_label(ax, vals[-1], star(t))
            ys.append(star(t))
        ax.set_xlim(-0.6, 5.8)
        pad_ylim(ax, min(ys), max(ys))
        label_points(ax, X_MLP, vals, MLP_SHADES)
        ax.set_xticks(X_MLP + [X_8B])
        ax.set_xticklabels(SIZES + ["8B"], fontsize=13)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(alpha=0.3, lw=0.5, ls=":")
    fig, axes = grid_axes(row_label, "Model Size (Parameters)", draw_scale)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=True, prop={"size": 13, "weight": "bold"},
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, "scale_full")

    # Fig. 4: task means with min/max bars
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), squeeze=False)
    for col, (group, members) in enumerate(TASK_GROUPS):
        ax = axes[0][col]
        m = np.array([mlp(t) for t in members])
        mean, lo, hi = m.mean(0), m.min(0), m.max(0)
        ax.plot(X_MLP, mean, "-", color=LINE_COLOR, lw=2.2, zorder=3)
        for x, y, l_, h_, c in zip(X_MLP, mean, lo, hi, MLP_SHADES):
            ax.errorbar([x], [y], yerr=[[y - l_], [h_ - y]], fmt="o", color=c, ecolor=c, elinewidth=1.6,
                        capsize=4, ms=9, zorder=4)
            ax.annotate(f"{y:.3f}", (x, h_), textcoords="offset points", xytext=(0, 6), ha="center",
                        fontsize=10, color=c, fontweight="bold", zorder=5)
        lo_all, hi_all = lo.min(), hi.max()
        stars = [star(t) for t in members]
        if None not in stars:
            s = np.array(stars)
            ax.errorbar([X_8B], [s.mean()], yerr=[[s.mean() - s.min()], [s.max() - s.mean()]], fmt="*",
                        color=STAR_COLOR, ecolor=STAR_COLOR, elinewidth=1.6, capsize=4, ms=17, zorder=4)
            ax.annotate(f"{s.mean():.3f}", (X_8B, s.max()), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=10.5, color=STAR_COLOR, fontweight="bold", zorder=5)
            delta_label(ax, mean[-1], s.mean(), fontsize=10, x=4.1)
            lo_all, hi_all = min(lo_all, s.min()), max(hi_all, s.max())
        ax.set_xticks(X_MLP + [X_8B])
        ax.set_xticklabels(SIZES + ["8B"], fontsize=14)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_xlim(-0.6, 5.8)
        pad_ylim(ax, lo_all, hi_all, top=0.16, bottom=0.07)
        ax.set_title(group, fontsize=15, fontweight="bold")
        ax.set_xlabel("Model Size (Parameters)", fontsize=16, fontweight="bold")
        if col == 0:
            ax.set_ylabel("ROUGE-L @ Best-of-64", fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3, lw=0.5, ls=":")
    dots = tuple(Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markeredgecolor=c, ms=8)
                 for c in MLP_SHADES)
    star_h = Line2D([0], [0], marker="*", color="none", markerfacecolor=STAR_COLOR,
                    markeredgecolor=STAR_COLOR, ms=16)
    fig.legend([dots, star_h, Line2D([0], [0], color="#08519c", lw=1.6)],
               ["Personalized Ranking Model", "Finetuned SOTA Reward Model",
                "Vertical I-bar: min/max across datasets"],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.35)}, handlelength=3.2,
               loc="upper center", ncol=3, frameon=True, prop={"size": 11, "weight": "bold"},
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "scale")


if __name__ == "__main__":
    main()
