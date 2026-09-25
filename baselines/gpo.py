#!/usr/bin/env python3
"""GPO (Group Preference Optimization, Zhao et al., ICLR 2024) on XRec (App. B.3).

Port of the official implementation (github.com/jamqd/Group-Preference-Optimization,
models/tnp.py + models/gpo.py, configs/gpo.yaml) to Best-of-N candidate selection:

- Groups are users; a "question" is one interaction with its candidate pool (the
  analogue of their survey question with answer options).
- x = mean-pooled generator embedding of the concatenated (prompt, candidate), their
  Appendix D choice (``--emb avg``); produced by baselines/embed_meanpool.py.
- y = the candidate's ROUGE-L normalized to a distribution within its question (their
  prob_y is a distribution over a question's options; linear normalization keeps the
  argmax).
- Architecture as in their code: MLP embedder over concat(x, y), a transformer
  encoder WITHOUT positional encodings whose attention mask lets every position
  attend only to context positions, predictor -> (mean, std), Gaussian NLL on targets
  after a per-question softmax over predicted means (their GPO.forward).
  Config (configs/gpo.yaml): d_model=128, emb_depth=4, dim_feedforward=128, nhead=4,
  dropout=0, num_layers=6.
- Meta-training: each episode samples one user and splits a random permutation of
  their ctx questions into context (uniform count, at most 6) and targets (at most 4),
  their collate logic adapted to our smaller per-user question counts. Adam, lr 1e-4,
  20,000 episodes.
- Test: context = the user's first 6 ctx questions with their labels; targets = the
  first 64 candidates of the held-out test interaction; BoN@N = argmax of the
  predicted mean over the first N candidates. A "swapped" diagnostic gives every user
  another user's context set.

Users with at least two ctx questions are evaluated (GPO needs no mined pairs), so
the test population differs slightly from run_personalized_rms.py.

    python baselines/gpo.py --dataset XRec_yelp
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rethinking import datasets as D
from rethinking.pools import best_of_n, headroom_captured

CFG = dict(d_model=128, emb_depth=4, dim_feedforward=128, nhead=4, dropout=0.0, num_layers=6)


def build_mlp(dim_in, dim_hid, dim_out, depth):
    modules = [nn.Linear(dim_in, dim_hid), nn.ReLU()]
    for _ in range(depth - 2):
        modules += [nn.Linear(dim_hid, dim_hid), nn.ReLU()]
    modules.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*modules)


class GPO(nn.Module):
    def __init__(self, dim_x, dim_y=1, d_model=128, emb_depth=4, dim_feedforward=128,
                 nhead=4, dropout=0.0, num_layers=6):
        super().__init__()
        self.embedder = build_mlp(dim_x + dim_y, d_model, d_model, emb_depth)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.predictor = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.ReLU(),
                                       nn.Linear(dim_feedforward, dim_y * 2))

    def encode(self, xc, yc, xt):
        """Input = [context (x, y); targets (x, 0)]; targets attend only to context."""
        tar = torch.cat((xt, torch.zeros(xt.shape[0], xt.shape[1], 1, device=xt.device)), dim=-1)
        inp = torch.cat((torch.cat((xc, yc), dim=-1), tar), dim=1)
        n_ctx, n_all = xc.shape[1], xc.shape[1] + xt.shape[1]
        mask = torch.full((n_all, n_all), float("-inf"), device=xt.device)
        mask[:, :n_ctx] = 0.0
        return self.encoder(self.embedder(inp), mask=mask)[:, -xt.shape[1]:]

    def forward_loss(self, xc, yc, xt, yt, tar_q_len):
        mean, std = torch.chunk(self.predictor(self.encode(xc, yc, xt)), 2, dim=-1)
        sm = torch.zeros_like(mean)
        start = 0
        for n in tar_q_len:                       # softmax within each target question
            sm[:, start:start + n] = F.softmax(mean[:, start:start + n], dim=1)
            start += n
        return -Normal(sm, torch.exp(std)).log_prob(yt).sum(-1).sum(-1).mean()

    @torch.no_grad()
    def predict_mean(self, xc, yc, xt):
        return torch.chunk(self.predictor(self.encode(xc, yc, xt)), 2, dim=-1)[0].squeeze(-1)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_split(data_root, name, split):
    npz = np.load(D.path(data_root, name, split, "meanpool_embeddings.npz"), allow_pickle=True)
    return dict(ids=[str(x) for x in npz["ids"]],
                uid={str(q["id"]): int(q["uid"]) for q in D.load_questions(data_root, name, split)},
                scores=D.load_scores(data_root, name, split),
                emb=npz["cand_embs"],                       # (P, S, D) fp16
                n=npz["n_samples"].astype(np.int64))


def norm_dist(r):
    r = np.asarray(r, np.float64)
    s = r.sum()
    return np.full_like(r, 1.0 / len(r)) if s <= 1e-8 else r / s


def group_by_user(ctx, metric="rougeL"):
    """{uid: [(x (S, D) fp16, y (S,)), ...]}, one entry per ctx question, file order."""
    by_user = {}
    for i, cid in enumerate(ctx["ids"]):
        if cid not in ctx["scores"] or cid not in ctx["uid"]:
            continue
        ni = int(ctx["n"][i])
        r = np.array(ctx["scores"][cid][metric][:ni], np.float32)
        if len(r) < 2:
            continue
        by_user.setdefault(ctx["uid"][cid], []).append((ctx["emb"][i, :ni], norm_dist(r).astype(np.float32)))
    return by_user


def _stack(qs):
    x = np.concatenate([e for e, _ in qs]).astype(np.float32)
    y = np.concatenate([y for _, y in qs]).astype(np.float32)[:, None]
    return x, y


def episode(by_user, uids, rng, device, max_ctx_q=6, max_tar_q=4):
    """One meta-training episode: a batch of one user."""
    for _ in range(50):
        u = uids[rng.integers(len(uids))]
        if len(by_user[u]) >= 2:
            break
    qs = by_user[u]
    order = rng.permutation(len(qs))
    n_ctx = int(rng.integers(1, min(max_ctx_q, len(qs) - 1) + 1))
    tar_q = [qs[j] for j in order[n_ctx:n_ctx + max_tar_q]]
    xc, yc = _stack([qs[j] for j in order[:n_ctx]])
    xt, yt = _stack(tar_q)
    t = lambda a: torch.from_numpy(a)[None].to(device)
    return t(xc), t(yc), t(xt), t(yt), [len(y) for _, y in tar_q]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=[n for n, d in D.DATASETS.items() if d.family == "xrec"])
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_ctx_q", type=int, default=6)
    ap.add_argument("--max_cand", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results/baselines/gpo")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    ctx = load_split(args.data_root, args.dataset, "ctx")
    tst = load_split(args.data_root, args.dataset, "test")
    by_user = group_by_user(ctx)
    uids = sorted(u for u, qs in by_user.items() if len(qs) >= 2)
    print(f"{args.dataset}: users={len(uids)} ctx questions/user median="
          f"{int(np.median([len(by_user[u]) for u in uids]))}", flush=True)

    model = GPO(dim_x=ctx["emb"].shape[-1], **CFG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GPO params: {n_params / 1e6:.2f}M", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    for step in range(args.steps):
        xc, yc, xt, yt, tlen = episode(by_user, uids, rng, device, max_ctx_q=args.max_ctx_q)
        opt.zero_grad()
        loss = model.forward_loss(xc, yc, xt, yt, tlen)
        loss.backward()
        opt.step()
        if (step + 1) % 2000 == 0:
            print(f"  step {step + 1}: loss {loss.item():.4f}", flush=True)

    # ---- Best-of-N on the held-out test interactions -----------------------------
    model.eval()
    swap = dict(zip(uids, (int(v) for v in rng.permutation(uids))))
    rows = [(i, tid) for i, tid in enumerate(tst["ids"])
            if tid in tst["scores"] and tst["uid"].get(tid) in swap]
    S = min(args.max_cand, tst["emb"].shape[1])
    n = np.array([min(S, int(tst["n"][i])) for i, _ in rows])
    mask = np.arange(S)[None, :] < n[:, None]
    ids = [tid for _, tid in rows]
    metrics = [m for m in ("rougeL", "rouge1", "bleu") if m in tst["scores"][ids[0]]]
    labels = {m: np.zeros((len(rows), S), np.float32) for m in metrics}
    for j, tid in enumerate(ids):
        for m in metrics:
            labels[m][j, :n[j]] = tst["scores"][tid][m][:n[j]]

    results = {}
    for mode in ["full", "swapped"]:
        scores = np.full((len(rows), S), -np.inf, np.float32)
        for j, (i, tid) in enumerate(rows):
            u = tst["uid"][tid]
            xc, yc = _stack(by_user[u if mode == "full" else swap[u]][:args.max_ctx_q])
            xt = tst["emb"][i, :n[j]].astype(np.float32)
            t = lambda a: torch.from_numpy(a)[None].to(device)
            scores[j, :n[j]] = model.predict_mean(t(xc), t(yc), t(xt))[0].cpu().numpy()
        results[mode] = best_of_n(scores, labels, mask, ids, metrics=tuple(metrics))

    full = results["full"]
    cap = 100 * headroom_captured(full["rougeL"][-1], full["mean_rougeL"], full["oracle_rougeL"])
    print(f"GPO ROUGE-L@64 {full['rougeL'][-1]:.4f} ({cap:.0f}% of headroom; swapped "
          f"{results['swapped']['rougeL'][-1]:.4f}; n={full['n']})", flush=True)
    out = os.path.join(args.out_dir, f"{args.dataset}.json")
    D.save_json({"gpo": dict(**results, params=n_params, config=CFG, args=vars(args))}, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
