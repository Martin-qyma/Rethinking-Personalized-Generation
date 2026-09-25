"""Uniform interface to the four generalist reward-model baselines (Sec. 4.1).

    skywork   Skywork/Skywork-Reward-V2-Llama-3.1-8B   sequence classifier, logits[:, 0]
    internlm  internlm/internlm2-7b-reward             remote code, model.get_scores(chats)
    urm       LxzGordon/URM-LLaMa-3.1-8B               sequence classifier, logits[:, 0]
    armorm    RLHFlow/ArmoRM-Llama3-8B-v0.1            remote code, output.score

Every model scores the generation prompt (user turn: query + retrieved profile)
followed by the candidate as the assistant turn, i.e. exactly what our ranker sees.
"""
import torch

REWARD_MODELS = {
    "skywork": "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    "internlm": "internlm/internlm2-7b-reward",
    "urm": "LxzGordon/URM-LLaMa-3.1-8B",
    "armorm": "RLHFlow/ArmoRM-Llama3-8B-v0.1",
}


def _compat_patches(tag):
    """Keep the remote-code models importable under recent transformers releases."""
    import transformers.models.llama.modeling_llama as llama
    # ArmoRM's remote code imports docstring constants that newer transformers removed
    for sym in ("LLAMA_INPUTS_DOCSTRING", "LLAMA_START_DOCSTRING"):
        if not hasattr(llama, sym):
            setattr(llama, sym, "")
    try:
        import transformers.models.mistral.modeling_mistral as mistral
        for sym in ("MISTRAL_INPUTS_DOCSTRING", "MISTRAL_START_DOCSTRING"):
            if not hasattr(mistral, sym):
                setattr(mistral, sym, "")
    except ImportError:
        pass
    if tag == "internlm":
        # internlm2 calls DynamicCache.from_legacy_cache(None), rejected by newer releases
        from transformers.cache_utils import DynamicCache
        orig = DynamicCache.from_legacy_cache.__func__

        def from_legacy_cache(cls, past_key_values=None, *a, **kw):
            return cls() if past_key_values is None else orig(cls, past_key_values, *a, **kw)

        DynamicCache.from_legacy_cache = classmethod(from_legacy_cache)


def chat_text(tok, prompt, response):
    text = tok.apply_chat_template([{"role": "user", "content": prompt},
                                    {"role": "assistant", "content": response}], tokenize=False)
    if tok.bos_token and text.startswith(tok.bos_token):   # the tokenizer adds it again
        text = text[len(tok.bos_token):]
    return text


def fit_prompt(tok, prompt, response, max_length, margin=8):
    """Left-truncate the prompt so that the whole response fits in ``max_length``.

    Long-form QA prompts reach ~4k tokens; right-truncating the chat would drop the
    candidate. Used by the finetuned comparator (training and scoring alike)."""
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    overhead = len(tok(chat_text(tok, "", response), add_special_tokens=False)["input_ids"])
    budget = max(max_length - overhead - margin, 32)
    return prompt if len(p_ids) <= budget else tok.decode(p_ids[-budget:])


class RewardModel:
    """``RewardModel(tag).score(prompt, responses) -> list[float]``.

    ``model_path`` loads a local (e.g. finetuned) checkpoint of the same family.
    """

    def __init__(self, tag, model_path=None, device="cuda", batch_size=32, max_length=4096,
                 fit_response=False):
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
        _compat_patches(tag)
        self.tag, self.bs, self.max_length, self.fit_response = tag, batch_size, max_length, fit_response
        name = model_path or REWARD_MODELS[tag]
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if tag == "armorm":
            # its gating reads the assistant-turn tokens at the END of the sequence
            self.tok.truncation_side = "left"
        kw = dict(dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
        if tag == "internlm":
            self.model = AutoModel.from_pretrained(name, **kw)
        elif tag == "skywork":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                name, num_labels=1, attn_implementation="sdpa", **kw)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(name, **kw)
        self.model.eval()

    @torch.no_grad()
    def score(self, prompt, responses):
        out = []
        for i in range(0, len(responses), self.bs):
            chunk = responses[i:i + self.bs]
            if self.tag == "internlm":
                chats = [[{"role": "user", "content": prompt}, {"role": "assistant", "content": r}]
                         for r in chunk]
                s = self.model.get_scores(self.tok, chats)
                out += [float(x) for x in (s if isinstance(s, (list, tuple)) else [s])]
                continue
            texts = [chat_text(self.tok, fit_prompt(self.tok, prompt, r, self.max_length)
                               if self.fit_response else prompt, r) for r in chunk]
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_length).to(self.model.device)
            res = self.model(**enc)
            s = res.score if self.tag == "armorm" else res.logits[:, 0]
            out += s.float().cpu().tolist()
        return out
