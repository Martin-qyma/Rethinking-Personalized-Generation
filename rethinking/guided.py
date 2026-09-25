"""Ranking-guided generation (Sec. 3.4): single-pass greedy decoding in which the
ranker steers the next token whenever the generator is uncertain.

    H_t = -sum_v p_t(v) log p_t(v)                                  (entropy gate)
    if H_t <= tau:  y_t = argmax l_t
    else:           g_t = d r_u(x, y) / d h_t,   u_t = W g_t            (W = LM head)
                    alpha_t = alpha0 * clip((H_t - tau) / (H_max - tau), 0, 1)
                    y_t = argmax(l_t + alpha_t u_t)

h_t is the generator's final-layer state at the current position, i.e. the
running representation of the partial answer. The first ``warmup`` tokens are
decoded without steering.
"""
from dataclasses import dataclass

import torch


@dataclass
class GuidanceConfig:
    tau: float = 1.0       # entropy threshold (nats)
    h_max: float = 8.0     # entropy at which the step size saturates
    alpha0: float = 5.0    # maximum step size; 0 disables steering (greedy decoding)
    warmup: int = 3        # unsteered opening tokens


@torch.no_grad()
def _forward(model, ids, past):
    return model(ids, past_key_values=past, use_cache=True, output_hidden_states=True)


def generate_guided(model, tokenizer, ranker, prompt_ids, h_x, h_u, cfg: GuidanceConfig,
                    max_new_tokens=48, W=None):
    """Decode one response. Returns (text, stats) with the number of decoding steps,
    the number of steered steps and the mean next-token entropy.

    ``W`` is the LM head weight in float32 (V, d); pass it to avoid recasting per call."""
    if W is None:
        W = model.lm_head.weight.detach().float()
    past, cur, out, step = None, prompt_ids, [], -1
    n_steered, entropy_sum = 0, 0.0
    for step in range(max_new_tokens):
        o = _forward(model, cur, past)
        past = o.past_key_values
        logits = o.logits[:, -1, :].float()                       # (1, V)
        h = o.hidden_states[-1][:, -1, :].float()                 # (1, d)
        logp = torch.log_softmax(logits[0], -1)
        H = float(-(logp.exp() * logp).sum())
        entropy_sum += H
        if step >= cfg.warmup and cfg.alpha0 > 0 and H > cfg.tau:
            alpha = cfg.alpha0 * min(max((H - cfg.tau) / max(cfg.h_max - cfg.tau, 1e-6), 0.0), 1.0)
            with torch.enable_grad():
                hp = h.detach().requires_grad_(True)
                (g,) = torch.autograd.grad(ranker(h_x, h_u, hp.unsqueeze(1)).squeeze(), hp)
            logits = logits + alpha * (g @ W.T)                    # u_t = W g_t
            n_steered += 1
        nxt = int(logits.argmax(-1))
        if nxt == tokenizer.eos_token_id:
            break
        out.append(nxt)
        cur = torch.tensor([[nxt]], device=prompt_ids.device)
    n = max(step + 1, 1)
    return tokenizer.decode(out, skip_special_tokens=True), dict(
        tokens=n, steered=n_steered, mean_entropy=entropy_sum / n)
