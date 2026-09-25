"""Personalized reward-model baselines of App. B.3 (PAL, VPL, PReF, LoRe) and our ranker.

Each baseline is a port of the official implementation that preserves the mechanism
that defines the method, namely **how the per-user parameters theta_u are obtained**:

  PAL-B [Chen et al., 2024]      theta_u = mixture weights over K prompt-conditioned
                                 prototypes (softmax); score = scaled cosine between the
                                 item projection and the user's ideal point. theta_u is
                                 fit by gradient descent (official main_pal_b_unseen.py).
                                 Default config: K=2, pref_learner_type='angle',
                                 learnable logit scale.
  VPL   [Poddar et al., 2024]    theta_u = latent z INFERRED by a pair encoder and a
                                 self-attention set encoder over the user's labeled
                                 (chosen, rejected) pairs; never fit per user. VAE
                                 objective (BT likelihood + beta * KL).
  PReF  [Shenfeld et al., 2025]  theta_u = free coefficients lambda_u over J=16 basis
                                 reward features phi(x, y); logit uses the pair
                                 difference phi(x, y1) - phi(x, y2) (their Model.forward);
                                 users fit by L2-regularized logistic regression.
  LoRe  [Bose et al., 2025]      theta_u = simplex weights over a shared low-rank basis
                                 V (rank 8), LINEAR in the response embedding; trained
                                 by alternating minimization (theta step, then V step).
  Ours                           theta_u = the user's profile embedding h_u; no per-user
                                 parameters and zero preference labels
                                 (rethinking.ranker.PersonalizedRanker).

All models read the same frozen generator embeddings as our ranker (h_x query, h_u
profile, h_y candidate; last-token states of Qwen2.5-7B-Instruct). VPL and LoRe
explicitly support precomputed embeddings, and PAL is designed as MLP heads over a
penultimate-layer representation. Hidden width 256 matches our ranker's ~2.8M budget.

Common interface used by run_personalized_rms.py:
    theta_is_fitted                 True if theta_u is a free parameter (PAL/PReF/LoRe)
    init_theta(n_users, device)     -> nn.Parameter (U, theta_dim)
    compute_theta(c, r, mask)       VPL only: infer z from a user's context pairs
    score(h_x, h_y, theta)          -> (B, N) reward used for Best-of-N selection
    pair_logit(h_x, h_a, h_b, th)   -> logit that a is preferred to b

Module construction order is part of the reproduction: with ``torch.manual_seed(0)``
before ``build(...)`` the initial weights are identical to those of the paper runs.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rethinking.ranker import PersonalizedRanker


def mlp(in_dim, hidden, out_dim, layers=2, dropout=0.1, act=nn.GELU):
    if layers <= 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))
    mods, d = [], in_dim
    for _ in range(layers - 1):
        mods += [nn.Linear(d, hidden), act(), nn.Dropout(dropout)]
        d = hidden
    mods += [nn.Linear(d, out_dim)]
    return nn.Sequential(*mods)


def _bcast(h_x, h_y):
    """h_x (B, D) or (B, N, D); h_y (B, D) or (B, N, D) -> both (B, N, D)."""
    if h_y.dim() == 2:
        h_y = h_y.unsqueeze(1)
    B, N, D = h_y.shape
    if h_x.dim() == 2:
        h_x = h_x.unsqueeze(1).expand(B, N, D)
    return h_x, h_y


class UserParamModel(nn.Module):
    theta_is_fitted = True     # False for amortized methods (VPL, ours)
    theta_simplex = False      # True -> softmax applied to theta

    def init_theta(self, n_users, device="cpu"):
        return nn.Parameter(torch.randn(n_users, self.theta_dim, device=device) * 0.01)

    def _theta(self, theta):
        return F.softmax(theta, dim=-1) if self.theta_simplex else theta

    def pair_logit(self, h_x, h_y_a, h_y_b, theta):
        return self.score(h_x, h_y_a, theta) - self.score(h_x, h_y_b, theta)


class PAL(UserParamModel):
    """PAL-B: r_u(x, y) = s * cos(f(x, y), sum_k w_uk g_k(x)), w_u = softmax(theta_u)."""
    theta_simplex = True

    def __init__(self, D, K=2, d_pref=128, hidden=256, dropout=0.1):
        super().__init__()
        self.theta_dim = K
        self.f = mlp(2 * D, hidden, d_pref, layers=2, dropout=dropout)            # item projector
        self.g = nn.ModuleList([mlp(D, hidden, d_pref, layers=2, dropout=dropout)
                                for _ in range(K)])                               # prototypes g_k(x)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def score(self, h_x, h_y, theta):
        hx, hy = _bcast(h_x, h_y)
        f = self.f(torch.cat([hx, hy], -1))                    # (B, N, P)
        protos = torch.stack([g(hx) for g in self.g], dim=2)   # (B, N, K, P)
        w = self._theta(theta)                                 # (B, K)
        u = (w.unsqueeze(1).unsqueeze(-1) * protos).sum(2)     # (B, N, P) ideal point
        f, u = F.normalize(f, dim=-1), F.normalize(u, dim=-1)
        return self.logit_scale.exp() * (f * u).sum(-1)        # cosine ("angle")


class PReF(UserParamModel):
    """PReF: r_u(x, y) = <lambda_u, phi(x, y)>, phi in R^J."""
    l2 = 1e-3                  # L2 on the user vectors (regularized logistic regression)

    def __init__(self, D, J=16, hidden=256, dropout=0.1):
        super().__init__()
        self.theta_dim = J
        self.phi = mlp(2 * D, hidden, J, layers=2, dropout=dropout)

    def score(self, h_x, h_y, theta):
        hx, hy = _bcast(h_x, h_y)
        return (self.phi(torch.cat([hx, hy], -1)) * theta.unsqueeze(1)).sum(-1)


class LoRe(UserParamModel):
    """LoRe: logit = <e_chosen - e_rejected, V softmax(w_u)> / temp, V in R^{D x r}.

    The prompt embedding is shared by both candidates of a pair and cancels in the
    difference, so the feature is the response embedding; within-prompt Best-of-N
    ranking is unaffected. ``V_sft`` is LoRe's base-RM direction for the cosine
    alignment regularizer; no base RM is used in the paper runs, so it is zero and
    the regularizer is inactive.
    """
    theta_simplex = True

    def __init__(self, D, rank=8, temp=100.0, V_sft=None):
        super().__init__()
        self.theta_dim = rank
        self.V = nn.Parameter(torch.randn(D, rank) / (D ** 0.5))
        self.temp = temp
        self.register_buffer("V_sft", V_sft if V_sft is not None else torch.zeros(D, 1))

    def score(self, h_x, h_y, theta):
        _, hy = _bcast(h_x, h_y)
        Vw = self._theta(theta) @ self.V.t()                   # (B, D)
        return (hy * Vw.unsqueeze(1)).sum(-1) / self.temp

    def v_alignment_reg(self):
        """mean(1 - cos) between the columns of V and the base-RM direction."""
        if self.V_sft.abs().sum() == 0:
            return torch.zeros((), device=self.V.device)
        cos = (F.normalize(self.V, dim=0) * F.normalize(self.V_sft, dim=0)[:, :1]).sum(0)
        return (1 - cos).mean()


class VPLPairEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self._model = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, e_c, e_r):
        return self._model(torch.cat([e_c, e_r], dim=-1))


class VPLSequenceEncoder(nn.Module):
    """Self-attention over the user's pair set, mean-pooled -> (mu, logvar).
    Batched form of the official vae_utils.SequenceEncoder (which loops over users)."""

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.w_q = nn.Linear(input_dim, input_dim)
        self.w_k = nn.Linear(input_dim, input_dim)
        self.w_v = nn.Linear(input_dim, input_dim)
        self.mean_layer = nn.Linear(input_dim, latent_dim)
        self.log_var_layer = nn.Linear(input_dim, latent_dim)

    def forward(self, seq, mask):
        # seq (U, P, Dp), mask (U, P) bool
        q, k, v = self.w_q(seq), self.w_k(seq), self.w_v(seq)
        att = torch.matmul(q, k.transpose(1, 2)) / (seq.shape[-1] ** 0.5)
        att = att.masked_fill(~mask.unsqueeze(1), float("-inf"))
        att = torch.nan_to_num(torch.softmax(att, dim=-1))
        weighted = torch.matmul(att, v)
        m = mask.unsqueeze(-1).float()
        pooled = (weighted * m).sum(1) / m.sum(1).clamp(min=1)
        return self.mean_layer(pooled), self.log_var_layer(pooled)


class VPL(UserParamModel):
    """VPL: r(x, y, z) with z ~ q(z | user's labeled pairs); decoder is an MLP."""
    theta_is_fitted = False

    def __init__(self, D, latent_dim=64, hidden=256, dropout=0.1, beta=0.01):
        super().__init__()
        self.theta_dim = self.latent_dim = latent_dim
        self.beta = beta
        self.pair_encoder = VPLPairEncoder(D, hidden, latent_dim)
        self.seq_encoder = VPLSequenceEncoder(latent_dim, latent_dim)
        self.decoder = mlp(2 * D + latent_dim, hidden, 1, layers=3, dropout=dropout)

    def compute_theta(self, ctx_c, ctx_r, ctx_mask, sample=False):
        """ctx_c / ctx_r (U, P, D) chosen / rejected embeddings; ctx_mask (U, P)."""
        mu, logvar = self.seq_encoder(self.pair_encoder(ctx_c, ctx_r), ctx_mask)
        mu, logvar = torch.clamp(mu, -1, 1), torch.clamp(logvar, -8, 2)
        z = mu
        if sample and self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return z, mu, logvar

    def score(self, h_x, h_y, theta):
        hx, hy = _bcast(h_x, h_y)
        z = theta.unsqueeze(1).expand(hy.shape[0], hy.shape[1], self.latent_dim)
        return self.decoder(torch.cat([hx, hy, z], -1)).squeeze(-1)

    @staticmethod
    def kl(mu, logvar):
        return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()


class Ours(PersonalizedRanker):
    """Our ranker under the common interface: theta_u is the profile embedding h_u."""
    theta_is_fitted = False

    def score(self, h_x, h_y, theta):
        return self(h_x, theta, h_y)


MODELS = {"ours": Ours, "vpl": VPL, "pal": PAL, "pref": PReF, "lore": LoRe}
LABELS = {"ours": "Ours", "vpl": "VPL", "pal": "PAL", "pref": "PReF", "lore": "LoRe"}


def build(name, D, **kw):
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {list(MODELS)}")
    if name == "ours":
        kw = {"hidden": 256, "layers": 3, "dropout": 0.1, **kw}
    return MODELS[name](D, **kw)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
