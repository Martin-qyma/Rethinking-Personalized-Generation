"""Candidate pools as tensors, and Best-of-N evaluation of a scorer on them."""
import numpy as np
import torch

from . import datasets as D
from .datasets import KS

METRICS = ("rougeL", "rouge1", "bleu")


def load_pools(data_root, name, split, max_cand=64):
    """Embeddings and per-candidate labels of a split, aligned by question.

    Returns a dict with tensors hx, hu (P, D), hy (P, S, D), mask (P, S), one
    (P, S) label tensor per available metric, and the question ids.
    """
    npz = np.load(D.path(data_root, name, split, "cand_embeddings.npz"), allow_pickle=True)
    ids = [str(x) for x in npz["ids"]]
    scores = D.load_scores(data_root, name, split)
    keep = [i for i, x in enumerate(ids) if x in scores]
    S = min(max_cand, npz["answer_embs"].shape[1])
    n = npz["n_samples"][keep]
    mask = np.zeros((len(keep), S), bool)
    labels = {m: np.zeros((len(keep), S), np.float32) for m in METRICS
              if m in scores[ids[keep[0]]]}
    for j, i in enumerate(keep):
        ni = min(int(n[j]), S)
        mask[j, :ni] = True
        for m in labels:
            labels[m][j, :ni] = scores[ids[i]][m][:ni]
    out = dict(ids=[ids[i] for i in keep], dim=int(npz["question_embs"].shape[1]),
               hx=torch.from_numpy(npz["question_embs"][keep].astype(np.float32)),
               hu=torch.from_numpy(npz["profile_embs"][keep].astype(np.float32)),
               hy=torch.from_numpy(npz["answer_embs"][keep][:, :S].astype(np.float32)),
               mask=torch.from_numpy(mask))
    out.update({m: torch.from_numpy(v) for m, v in labels.items()})
    return out


def standardize_within_pool(y, mask):
    """z-score each row over its valid candidates (target standardization, Sec. 3.3)."""
    z = y.clone()
    for i in range(y.shape[0]):
        v = y[i][mask[i]]
        if v.numel() > 1 and v.std() > 1e-6:
            z[i][mask[i]] = (v - v.mean()) / v.std()
    return z


@torch.no_grad()
def predict(model, pools, device, batch=256):
    """Scores (P, S) of every candidate; invalid positions are left as is."""
    model.eval()
    out = []
    for s in range(0, pools["hy"].shape[0], batch):
        sl = slice(s, s + batch)
        out.append(model(pools["hx"][sl].to(device), pools["hu"][sl].to(device),
                         pools["hy"][sl].to(device)).float().cpu())
    return torch.cat(out).numpy()


def best_of_n(scores, labels, mask, ids, ks=KS, metrics=("rougeL", "rouge1")):
    """Best-of-N selection of the first N candidates by ``scores``.

    Pools with fewer than two candidates or constant labels carry no selection
    signal and are skipped. Returns curves per metric, oracle/mean references,
    and per-question selected indices so curves can be recomputed on any subset.
    """
    per_q = {"ids": [], "sel_idx": {str(k): [] for k in ks}}
    sel = {m: {k: [] for k in ks} for m in metrics}
    oracle, mean = [], []
    for j in range(scores.shape[0]):
        mm = mask[j]
        yl = labels["rougeL"][j][mm]
        if len(yl) < 2 or yl.std() < 1e-9:
            continue
        pp = scores[j][mm]
        per_q["ids"].append(ids[j])
        oracle.append(float(yl.max())); mean.append(float(yl.mean()))
        for k in ks:
            b = int(np.argmax(pp[:min(k, len(pp))]))
            per_q["sel_idx"][str(k)].append(b)
            for m in metrics:
                sel[m][k].append(float(labels[m][j][mm][b]))
    res = {"ks": list(ks), "n": len(per_q["ids"]),
           "oracle_rougeL": float(np.mean(oracle)), "mean_rougeL": float(np.mean(mean))}
    for m in metrics:
        res[m] = [float(np.mean(sel[m][k])) for k in ks]
    res["per_q"] = per_q
    return res


def headroom_captured(selected, mean, oracle):
    """Share of the oracle gain over a random pick that a selector recovers."""
    return (selected - mean) / max(oracle - mean, 1e-9)
