#!/usr/bin/env python3
"""Seen-user comparison with PAL / VPL / PReF / LoRe on XRec (App. B.3, Table B.3).

Protocol (favourable to the baselines: they get per-user preference labels, our
ranker gets none):
  1. Context pairs. For every test user, the ``ctx`` split holds up to eight of their
     training interactions with 16 on-policy candidates each. From each interaction
     we mine up to two (chosen, rejected) pairs by ROUGE-L against the user's
     reference: (best, worst) and (2nd best, 2nd worst), kept if the ROUGE-L gap
     exceeds 0.02. At most 16 pairs per user.
  2. Training. Each method trains its shared components on all users' pairs, jointly
     with the free per-user theta_u for PAL / PReF / LoRe; VPL infers z_u from the
     pairs with its set encoder. Full-batch Adam (lr 1e-3), 200 epochs, Bradley-Terry
     loss with a fresh random orientation of every pair each epoch.
     Our ranker is trained on the same ctx interactions with its native objective
     (pointwise MSE to within-pool z-scored ROUGE-L over the whole candidate pool,
     no pairs, no user ids) and conditions on the profile embedding.
  3. Evaluation. Best-of-N on each user's held-out test interaction (disjoint from
     the context), first N of the 64 test candidates. We also report a "swapped"
     diagnostic in which every user receives another user's theta.

Writes ``<out_dir>/<dataset>.json``: {method: {"full": curves, "swapped": curves,
"params": n}}, curves as returned by rethinking.pools.best_of_n (incl. per-question
selections, which baselines/synthesizeme/collect.py uses to score the 1,000-prompt
subset).

    python baselines/run_personalized_rms.py --dataset XRec_yelp
    python baselines/run_personalized_rms.py --dataset XRec_amazon --models ours vpl
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from baselines.models import LABELS, MODELS, build, count_params
from rethinking import datasets as D
from rethinking.pools import best_of_n, headroom_captured

METRIC = "rougeL"
MARGIN = 0.02          # minimum ROUGE-L gap of a context pair
MAX_PAIRS = 16         # context pairs per user


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_split(data_root, name, split):
    npz = np.load(D.path(data_root, name, split, "cand_embeddings.npz"), allow_pickle=True)
    return dict(ids=[str(x) for x in npz["ids"]],
                uid={str(q["id"]): int(q["uid"]) for q in D.load_questions(data_root, name, split)},
                scores=D.load_scores(data_root, name, split),
                hx=npz["question_embs"].astype(np.float32),
                hu=npz["profile_embs"].astype(np.float32),
                hy=npz["answer_embs"].astype(np.float32),
                n=npz["n_samples"].astype(np.int64))


def mine_pairs(ctx):
    """{uid: [(ctx row, chosen idx, rejected idx), ...]} in ctx file order.

    Shared with baselines/synthesizeme/prep.py so that every method sees the same
    pairs. Scores are kept in float64 as read from JSON (the ordering of near-ties
    and the margin test depend on it).
    """
    by_user = {}
    for i, cid in enumerate(ctx["ids"]):
        if cid not in ctx["scores"] or cid not in ctx["uid"]:
            continue
        r = np.array(ctx["scores"][cid][METRIC][: int(ctx["n"][i])])
        if len(r) < 2 or r.std() < 1e-6:
            continue
        order = np.argsort(-r)                                # best -> worst
        for a, b in [(0, -1), (1, -2)]:
            if len(order) > max(a, abs(b)) and r[order[a]] - r[order[b]] > MARGIN:
                by_user.setdefault(ctx["uid"][cid], []).append((i, int(order[a]), int(order[b])))
    return by_user


def build_dataset(data_root, name):
    ctx = load_split(data_root, name, "ctx")
    tst = load_split(data_root, name, "test")
    by_user = mine_pairs(ctx)
    users = sorted(by_user)
    uindex = {u: k for k, u in enumerate(users)}
    U, P, Dm = len(users), min(MAX_PAIRS, max(len(p) for p in by_user.values())), ctx["hx"].shape[1]

    # per-user context pairs, padded to P
    cx, cc, cr = (np.zeros((U, P, Dm), np.float32) for _ in range(3))
    cm = np.zeros((U, P), bool)
    for u in users:
        k = uindex[u]
        for j, (i, a, b) in enumerate(by_user[u][:P]):
            cx[k, j], cc[k, j], cr[k, j], cm[k, j] = ctx["hx"][i], ctx["hy"][i, a], ctx["hy"][i, b], True

    # held-out test interaction(s) of the same users
    t_idx = np.array([i for i, tid in enumerate(tst["ids"])
                      if tst["uid"].get(tid) in uindex and tid in tst["scores"]])
    t_ids = [tst["ids"][i] for i in t_idx]
    tn = tst["n"][t_idx]
    S = int(tn.max())
    mask = np.arange(S)[None, :] < tn[:, None]
    metrics = [m for m in ("rougeL", "rouge1", "bleu") if m in tst["scores"][t_ids[0]]]
    labels = {m: np.zeros((len(t_idx), S), np.float32) for m in metrics}
    for j, tid in enumerate(t_ids):
        for m in metrics:
            labels[m][j, :tn[j]] = tst["scores"][tid][m][:tn[j]]

    # our ranker's training data: every candidate of the users' ctx interactions,
    # ROUGE-L z-scored within its pool (population std)
    keep = [i for i, cid in enumerate(ctx["ids"])
            if cid in ctx["scores"] and ctx["uid"].get(cid) in uindex]
    Sc = int(ctx["n"][keep].max())
    ry, rm = np.zeros((len(keep), Sc), np.float32), np.zeros((len(keep), Sc), bool)
    for j, i in enumerate(keep):
        ni = int(ctx["n"][i])
        v = np.array(ctx["scores"][ctx["ids"][i]][METRIC][:ni], np.float32)
        if v.std() > 1e-6:
            v = (v - v.mean()) / v.std()
        ry[j, :ni], rm[j, :ni] = v, True

    t = torch.from_numpy
    return dict(
        users=users, dim=Dm,
        ctx=dict(hx=t(cx), hc=t(cc), hr=t(cr), mask=t(cm)),
        ours_ctx=dict(hx=t(ctx["hx"][keep]), hu=t(ctx["hu"][keep]), hy=t(ctx["hy"][keep]),
                      y=t(ry), mask=t(rm)),
        # the user's profile embedding (ours), from their first context interaction
        prof=t(np.stack([ctx["hu"][by_user[u][0][0]] for u in users])),
        test=dict(ids=t_ids, hx=t(tst["hx"][t_idx]), hy=t(tst["hy"][t_idx]),
                  user=torch.tensor([uindex[tst["uid"][x]] for x in t_ids]),
                  mask=mask, labels=labels),
    )


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def bt_loss(logit, flip):
    """Bradley-Terry loss; ``logit`` = score(chosen) - score(rejected), and a pair
    with flip=True is presented in the opposite orientation."""
    target = torch.where(flip, torch.zeros_like(logit), torch.ones_like(logit))
    logit = torch.where(flip, -logit, logit)
    return F.binary_cross_entropy_with_logits(logit, target)


def train(name, model, data, device, epochs=200, lr=1e-3, seed=0, log_every=50):
    """Train the shared parameters (and the free theta_u of PAL / PReF / LoRe).

    Returns theta (U, k): fitted parameters, VPL's inferred posterior means, or the
    profile embeddings for ours.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    model.to(device).train()
    U = len(data["users"])
    theta = model.init_theta(U, device=device) if model.theta_is_fitted else None
    opt = torch.optim.Adam(list(model.parameters()) + ([theta] if theta is not None else []), lr=lr)
    hx, hc, hr, m = (data["ctx"][k].to(device) for k in ("hx", "hc", "hr", "mask"))
    torch.rand(m.shape, device=device, generator=gen)   # unused draw, kept so seeds reproduce the paper runs

    for ep in range(epochs):
        flip = torch.rand(m.shape, device=device, generator=gen) < 0.5
        opt.zero_grad()
        if name == "vpl":
            z, mu, lv = model.compute_theta(hc, hr, m, sample=True)
            loss = bt_loss(model.pair_logit(hx, hc, hr, z)[m], flip[m]) + model.beta * model.kl(mu, lv)
        elif name == "ours":
            # pointwise MSE on within-pool z-scored ROUGE-L, minibatch of 512 pools
            oc = data["ours_ctx"]
            sel = torch.randint(0, oc["hy"].shape[0], (min(512, oc["hy"].shape[0]),),
                                generator=gen, device=device).cpu()
            bm = oc["mask"][sel].to(device)
            pred = model.score(oc["hx"][sel].to(device), oc["hy"][sel].to(device), oc["hu"][sel].to(device))
            loss = (((pred - oc["y"][sel].to(device)) ** 2) * bm).sum() / bm.sum().clamp(min=1)
        elif name == "lore":
            # alternating minimization: theta step with V frozen, then V step
            loss_w = bt_loss(model.pair_logit(hx, hc, hr, theta)[m], flip[m])
            loss_w.backward(inputs=[theta])
            opt.step()
            opt.zero_grad()
            reg = min(1.0, max(0.0, (ep - 0.2 * epochs) / (0.6 * epochs))) * 0.1
            loss = bt_loss(model.pair_logit(hx, hc, hr, theta)[m], flip[m]) + reg * model.v_alignment_reg()
            loss.backward(inputs=list(model.parameters()))
            opt.step()
        else:   # pal, pref
            loss = bt_loss(model.pair_logit(hx, hc, hr, theta)[m], flip[m])
            if name == "pref":
                loss = loss + model.l2 * theta.pow(2).sum(-1).mean()
        if name != "lore":
            loss.backward()
            opt.step()
        if log_every and (ep + 1) % log_every == 0:
            print(f"  [{name}] epoch {ep + 1}: loss {loss.item():.4f}", flush=True)

    model.eval()
    if theta is not None:
        return theta.detach()
    if name == "vpl":
        with torch.no_grad():
            return model.compute_theta(hc, hr, m, sample=False)[0]
    return data["prof"].to(device)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, theta, data, device, swap_seed=None, batch=256):
    """Best-of-N on the held-out test interactions. With ``swap_seed`` every user is
    scored with a randomly permuted user's theta (user-conditioning diagnostic)."""
    t = data["test"]
    if swap_seed is not None:
        perm = torch.randperm(theta.shape[0], generator=torch.Generator().manual_seed(swap_seed))
        theta = theta[perm.to(theta.device)]
    model.eval()
    scores = []
    for s in range(0, t["hy"].shape[0], batch):
        sl = slice(s, s + batch)
        th = theta[t["user"][sl].to(theta.device)]
        scores.append(model.score(t["hx"][sl].to(device), t["hy"][sl].to(device), th).float().cpu())
    scores = torch.cat(scores).numpy()
    return best_of_n(scores, t["labels"], t["mask"], t["ids"], metrics=tuple(t["labels"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=[n for n, d in D.DATASETS.items() if d.family == "xrec"])
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results/baselines/personalized_rms")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = build_dataset(args.data_root, args.dataset)
    npairs = data["ctx"]["mask"].sum(1)
    print(f"{args.dataset}: users={len(data['users'])}  ctx pairs/user median={int(npairs.median())} "
          f"max={int(npairs.max())}  test interactions={len(data['test']['ids'])}", flush=True)

    out = os.path.join(args.out_dir, f"{args.dataset}.json")
    results = D.load_json(out) if os.path.exists(out) else {}
    for name in args.models:
        torch.manual_seed(args.seed)
        model = build(name, data["dim"])
        print(f"[{LABELS[name]}] params={count_params(model) / 1e6:.2f}M", flush=True)
        theta = train(name, model, data, device, epochs=args.epochs, lr=args.lr, seed=args.seed)
        full = evaluate(model, theta, data, device)
        swapped = evaluate(model, theta, data, device, swap_seed=args.seed)
        results[name] = dict(full=full, swapped=swapped, params=count_params(model), args=vars(args))
        cap = 100 * headroom_captured(full["rougeL"][-1], full["mean_rougeL"], full["oracle_rougeL"])
        print(f"  -> ROUGE-L@64 {full['rougeL'][-1]:.4f} ({cap:.0f}% of headroom; swapped "
              f"{swapped['rougeL'][-1]:.4f}; mean {full['mean_rougeL']:.4f}, oracle {full['oracle_rougeL']:.4f})",
              flush=True)
        D.save_json(results, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
