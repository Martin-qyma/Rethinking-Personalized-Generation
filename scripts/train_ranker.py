#!/usr/bin/env python3
"""Train the personalized ranking model and evaluate Best-of-N selection.

Protocol of the paper (Sec. 4.1, App. C.2): pointwise MSE to within-pool
z-scored ROUGE-L, AdamW (lr 5e-4, weight decay 0.01), 15 epochs, batch 64,
gradient clipping 1.0, seed 0, training pools of 64 candidates. The same
trainer produces every ranker number in the paper; the flags below switch
between the main run and the ablations:

    --size {1M,3M,10M,30M}         ranker size (Fig. 4)                default 3M
    --loss {mse,bt,ranknet,...}    objective (Table 2)                 default mse
    --inputs {xuy,xy,y}            input ablation                      default xuy
    --train_cands K                candidates per training pool        default 64

Writes ``<out_dir>/<dataset>/<run>.json`` (curves + per-question selections,
consumed by collect_results.py) and ``<ckpt_dir>/<dataset>/<run>/`` (weights,
consumed by guided_generation.py).

    python scripts/train_ranker.py --dataset LaMP_7
    python scripts/train_ranker.py --dataset XRec_amazon --size 30M
    for l in mse bt ranknet listnet listmle lambdarank; do
        python scripts/train_ranker.py --dataset LaMP_7 --loss $l; done
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rethinking import datasets as D
from rethinking.losses import LOSSES, ranking_loss
from rethinking.pools import best_of_n, headroom_captured, load_pools, predict, standardize_within_pool
from rethinking.ranker import SIZES, PersonalizedRanker


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(D.DATASETS))
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--size", default="3M", choices=list(SIZES))
    ap.add_argument("--loss", default="mse", choices=LOSSES)
    ap.add_argument("--inputs", default="xuy", choices=["xuy", "xy", "y"])
    ap.add_argument("--train_cands", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", default=None, help="run name (default derived from the flags)")
    ap.add_argument("--out_dir", default="results/ranker")
    ap.add_argument("--ckpt_dir", default="checkpoints/ranker")
    args = ap.parse_args()

    ds = D.get(args.dataset)
    run = args.run or "_".join(
        [args.size, args.loss] + ([args.inputs] if args.inputs != "xuy" else [])
        + ([f"pool{args.train_cands}"] if args.train_cands != 64 else []))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr = load_pools(args.data_root, args.dataset, ds.train_split)
    ev = load_pools(args.data_root, args.dataset, ds.eval_split)
    if args.train_cands < tr["mask"].shape[1]:
        tr["mask"][:, args.train_cands:] = False
    targets = tr["rougeL"] if args.loss == "mse_abs" else standardize_within_pool(tr["rougeL"], tr["mask"])
    print(f"{args.dataset} [{run}]: train={len(tr['ids'])} eval={len(ev['ids'])} dim={tr['dim']}", flush=True)

    torch.manual_seed(args.seed)
    model = PersonalizedRanker(tr["dim"], inputs=args.inputs, **SIZES[args.size]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    P = len(tr["ids"])
    for ep in range(args.epochs):
        model.train()
        perm, tot, nb = torch.randperm(P), 0.0, 0
        for s in range(0, P, args.batch):
            b = perm[s:s + args.batch]
            pred = model(tr["hx"][b].to(device), tr["hu"][b].to(device), tr["hy"][b].to(device))
            loss = ranking_loss(args.loss, pred, targets[b].to(device), tr["mask"][b].to(device), gen)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss); nb += 1
        if (ep + 1) % 5 == 0:
            print(f"  epoch {ep + 1}: loss {tot / max(nb, 1):.4f}", flush=True)

    scores = predict(model, ev, device)
    labels = {m: ev[m].numpy() for m in ("rougeL", "rouge1", "bleu") if m in ev}
    res = best_of_n(scores, labels, ev["mask"].numpy(), ev["ids"], metrics=tuple(labels))
    res.update(dataset=args.dataset, run=run, params=model.num_parameters(), args=vars(args))
    out = os.path.join(args.out_dir, args.dataset, f"{run}.json")
    D.save_json(res, out)
    model.save(os.path.join(args.ckpt_dir, args.dataset, run), protocol=vars(args))
    cap = 100 * headroom_captured(res["rougeL"][-1], res["mean_rougeL"], res["oracle_rougeL"])
    print(f"[{run}] params={res['params'] / 1e6:.2f}M  ROUGE-L@64={res['rougeL'][-1]:.4f} "
          f"({cap:.0f}% of headroom; mean {res['mean_rougeL']:.4f}, oracle {res['oracle_rougeL']:.4f}) -> {out}")


if __name__ == "__main__":
    main()
