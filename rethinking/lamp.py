"""LaMP prompt construction with BM25 profile retrieval (tasks 4, 5 and 7).

Adapted from the official LaMP code (https://github.com/LaMP-Benchmark/LaMP,
``LaMP/prompts/prompts.py`` and ``LaMP/prompts/utils.py``), restricted to the
three generation tasks used in the paper. The logic is unchanged so that
prompts match the benchmark's retrieval-augmented protocol exactly.
"""
from rank_bm25 import BM25Okapi

TASKS = ("LaMP-4", "LaMP-5", "LaMP-7")


# ---------------------------------------------------------------------------
# query / corpus extraction
# ---------------------------------------------------------------------------
def _after(text, marker):
    i = text.find(marker)
    return None if i == -1 else text[i + len(marker):].strip()


def query_and_corpus(task, inp, profile):
    """Return (retrieval corpus, query) for a LaMP input and its user profile."""
    if task == "LaMP-4":   # news headline generation
        return [f'{x["title"]} {x["text"]}' for x in profile], _after(inp, "article:")
    if task == "LaMP-5":   # scholarly title generation
        return [f'{x["title"]} {x["abstract"]}' for x in profile], _after(inp, "paper:")
    if task == "LaMP-7":   # tweet paraphrasing
        return [x["text"] for x in profile], _after(inp, ":")
    raise ValueError(f"unsupported LaMP task {task}")


def retrieve(task, inp, profile, k):
    """Top-k profile items by BM25 against the task query. Returns (query, items)."""
    corpus, query = query_and_corpus(task, inp, profile)
    bm25 = BM25Okapi([x.split() for x in corpus])
    return query, bm25.get_top_n(query.split(), profile, n=k)


def profile_text(task, items):
    """Serialized retrieved profile, the text behind the user embedding h_u."""
    if task == "LaMP-4":
        return "\n".join(f'{x["title"]} {x["text"]}' for x in items)
    if task == "LaMP-5":
        return "\n".join(f'{x["title"]} {x["abstract"]}' for x in items)
    if task == "LaMP-7":
        return "\n".join(x["text"] for x in items)
    raise ValueError(f"unsupported LaMP task {task}")


# ---------------------------------------------------------------------------
# profile-augmented prompts (token-budgeted, as in the benchmark)
# ---------------------------------------------------------------------------
def _truncate(tokenizer, text, budget):
    ids = tokenizer(text, max_length=budget, truncation=True)["input_ids"]
    return tokenizer.batch_decode([ids], skip_special_tokens=True)[0], len(ids)


def _news_prompt(inp, profile, max_length, tok):
    per_p = (max_length - 1 - 2 * (len(profile) - 1)) // len(profile)
    saved, parts = 0, []
    for p in profile:
        need = len(tok(f'"{p["title"]}" is the title for " " ')["input_ids"])
        text, n = _truncate(tok, p["text"], per_p + saved - need)
        saved += per_p - n - need
        parts.append(f'"{p["title"]}" is the title for "{text}" ')
    return f'{", and ".join(parts)}. {inp}'


def _paper_prompt(inp, profile, max_length, tok):
    per_p = (max_length - 1 - 2 * (len(profile) - 1)
             - len(tok("Following the given patterns")["input_ids"])) // len(profile)
    saved, parts = 0, []
    for p in profile:
        need = len(tok(f'"{p["title"]}" is a title " " ')["input_ids"])
        text, n = _truncate(tok, p["abstract"], per_p + saved - need)
        saved += per_p - n - need
        parts.append(f'"{p["title"]}" is a title for "{text}" ')
    return f'{", and ".join(parts)}. Following the given patterns {inp}'


def _tweet_prompt(inp, profile, max_length, tok):
    per_p = (max_length - 1 - 2 * (len(profile) - 1)
             - len(tok("are written by user. Following the given patterns")["input_ids"])) // len(profile)
    saved, parts = 0, []
    for p in profile:
        need = len(tok('"" ')["input_ids"])
        text, n = _truncate(tok, p["text"], per_p + saved - need)
        saved += per_p - n - need
        parts.append(f'"{text}" ')
    return f'{", and ".join(parts)} are written by a person. Following the given patterns {inp}'


_BUILDERS = {"LaMP-4": _news_prompt, "LaMP-5": _paper_prompt, "LaMP-7": _tweet_prompt}


def make_prompt_fn(tokenizer, num_retrieve=3, max_length=512):
    """Return ``fn(inp, profile, task) -> prompt`` (BM25 top-k, token-budgeted).

    ``tokenizer`` is the generator's tokenizer; it only sets the profile budget.
    """
    def fn(inp, profile, task):
        _, selected = retrieve(task, inp, profile, num_retrieve)
        factor = 0.6
        while True:
            try:
                budget = max_length - min(len(tokenizer(inp)["input_ids"]), int(factor * max_length))
                return _BUILDERS[task](inp, selected, budget, tokenizer)
            except Exception:
                factor -= 0.1
                if factor < 0:
                    return inp
    return fn
