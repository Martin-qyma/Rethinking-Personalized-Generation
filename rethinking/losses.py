"""Ranking objectives compared in Appendix B.1 (pointwise MSE is the default).

All losses take predictions and targets of shape (B, N) plus a validity mask;
targets are the within-pool standardized metric except for ``mse_abs``.

    mse         pointwise regression on within-pool z-scored ROUGE-L (ours)
    mse_abs     pointwise regression on raw ROUGE-L
    bt          Bradley-Terry on a random 25% of ordered candidate pairs
    ranknet     Bradley-Terry over all ordered pairs
    lambdarank  pairwise, weighted by |delta NDCG| of swapping the pair
    listnet     softmax cross-entropy against softmax(targets)
    listmle     Plackett-Luce likelihood of the target ordering
"""
import torch
import torch.nn.functional as F

LOSSES = ["mse", "mse_abs", "bt", "ranknet", "listnet", "listmle", "lambdarank"]


def ranking_loss(name, pred, tgt, mask, generator=None):
    neg = torch.finfo(pred.dtype).min
    if name in ("mse", "mse_abs"):
        return (((pred - tgt) ** 2) * mask).sum() / mask.sum().clamp(min=1)

    if name in ("bt", "ranknet", "lambdarank"):
        valid = mask.unsqueeze(2) & mask.unsqueeze(1)                 # (B, N, N)
        dt = tgt.unsqueeze(2) - tgt.unsqueeze(1)
        dp = pred.unsqueeze(2) - pred.unsqueeze(1)
        pos = valid & (dt > 1e-6)                                      # i truly better than j
        if pos.sum() == 0:
            return pred.sum() * 0.0
        if name == "bt":
            keep = pos & (torch.rand(pos.shape, device=pred.device, generator=generator) < 0.25)
            keep = keep if keep.sum() > 0 else pos
            return -F.logsigmoid(dp[keep]).mean()
        if name == "ranknet":
            return -F.logsigmoid(dp[pos]).mean()
        with torch.no_grad():                                          # lambdarank weights
            rank = pred.masked_fill(~mask, neg).argsort(1, descending=True).argsort(1).float()
            disc = 1.0 / torch.log2(rank + 2)
            gain = 2 ** tgt.clamp(min=0) - 1
            w = ((gain.unsqueeze(2) - gain.unsqueeze(1)).abs()
                 * (disc.unsqueeze(2) - disc.unsqueeze(1)).abs())[pos]
            w = w / w.mean().clamp(min=1e-8)
        return -(w * F.logsigmoid(dp[pos])).mean()

    if name == "listnet":
        p = torch.softmax(pred.masked_fill(~mask, neg), dim=1)
        q = torch.softmax(tgt.masked_fill(~mask, neg), dim=1)
        return -(q * torch.log(p.clamp(min=1e-9))).sum(1).mean()

    if name == "listmle":
        order = tgt.masked_fill(~mask, neg).argsort(1, descending=True)
        ps, ms = pred.gather(1, order), mask.gather(1, order)
        ps = ps.masked_fill(~ms, neg)
        rev_lse = torch.logcumsumexp(ps.flip(1), dim=1).flip(1)       # log sum_{j>=i} exp(s_j)
        ll = ((ps - rev_lse) * ms).sum(1) / ms.sum(1).clamp(min=1)
        return -ll.mean()

    raise ValueError(f"unknown loss {name!r}; choose from {LOSSES}")
