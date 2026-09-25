"""The personalized ranking model: a small MLP over recycled generator states.

    r_u(x, y) = w^T g_L o ... o g_1([h_x || h_u || h_y])          (paper Eq. 2)

h_x, h_u, h_y are final-layer last-token hidden states of the frozen generator
for the task query, the user profile, and the candidate. No per-user parameters.
"""
import json
import os

import torch
import torch.nn as nn

# (hidden width, number of linear layers) -> parameter count at d = 3584
SIZES = {
    "1M": dict(hidden=96, layers=3),     # 1.04M
    "3M": dict(hidden=256, layers=3),    # 2.82M (default)
    "10M": dict(hidden=1024, layers=3),  # 12.06M
    "30M": dict(hidden=2048, layers=4),  # 30.42M
}


class PersonalizedRanker(nn.Module):
    """MLP with GELU activations and dropout; scores (B, N) candidates per prompt.

    ``inputs`` selects the input ablation: "xuy" (full), "xy" (profile zeroed),
    "y" (candidate only; query and profile zeroed).
    """

    def __init__(self, dim, hidden=256, layers=3, dropout=0.1, inputs="xuy"):
        super().__init__()
        assert inputs in ("xuy", "xy", "y")
        self.config = dict(dim=dim, hidden=hidden, layers=layers, dropout=dropout, inputs=inputs)
        self.inputs = inputs
        mods, d = [], 3 * dim
        for _ in range(layers - 1):
            mods += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        mods += [nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*mods)

    def forward(self, h_x, h_u, h_y):
        """h_x, h_u: (B, D) or (B, N, D); h_y: (B, N, D) or (B, D). Returns (B, N)."""
        if h_y.dim() == 2:
            h_y = h_y.unsqueeze(1)
        B, N, D = h_y.shape
        if h_x.dim() == 2:
            h_x = h_x.unsqueeze(1).expand(B, N, D)
        if h_u.dim() == 2:
            h_u = h_u.unsqueeze(1).expand(B, N, D)
        if self.inputs != "xuy":
            h_u = torch.zeros_like(h_u)
        if self.inputs == "y":
            h_x = torch.zeros_like(h_x)
        return self.mlp(torch.cat([h_x, h_u, h_y], -1)).squeeze(-1)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -- checkpoints ---------------------------------------------------------
    def save(self, out_dir, **meta):
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(out_dir, "model.pt"))
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump({**self.config, **meta}, f, indent=1)

    @classmethod
    def load(cls, ckpt_dir, device="cpu"):
        with open(os.path.join(ckpt_dir, "config.json")) as f:
            cfg = json.load(f)
        kw = {k: cfg[k] for k in ("hidden", "layers", "dropout", "inputs") if k in cfg}
        m = cls(cfg["dim"], **kw)
        m.load_state_dict(torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu"))
        return m.to(device).eval()
