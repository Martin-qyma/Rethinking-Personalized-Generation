"""Dataset registry, file layout, and prompt construction shared by every stage.

Every dataset lives in ``<data_root>/<name>/`` with one set of files per split:

    <split>_questions.json            [{id, input, ...}]           (prepared inputs)
    <split>_outputs.json              {task, golds: [{id, output}]} (user-written references)
    <split>_samples.json              [{id, samples: [str, ...]}]   (candidate pools)
    <split>_samples_scores.json       [{id, rouge1, rougeL, bleu}]  (per-candidate labels)
    <split>_cand_embeddings.npz       ids, question_embs, profile_embs, answer_embs, n_samples
    <split>_rmscores_<rm>.jsonl       {id, scores}                  (reward-model scores)

LaMP question files are the official benchmark files ({id, input, profile}); the
prompt is built on the fly with BM25 retrieval. LaMP-QA and XRec question files
are produced by ``scripts/prepare_lampqa.py`` / ``scripts/prepare_xrec.py`` and
carry the final prompt ({id, system, input}) plus the texts behind the query and
user embeddings ({q_text, u_text}).
"""
import glob
import json
import os
from dataclasses import dataclass

from . import lamp

GENERATOR = "Qwen/Qwen2.5-7B-Instruct"
KS = [1, 2, 4, 8, 16, 32, 64]          # Best-of-N pool sizes reported in the paper


@dataclass(frozen=True)
class Dataset:
    name: str
    family: str          # "lamp" | "lampqa" | "xrec"
    label: str           # short name used in the paper
    train_split: str
    eval_split: str
    max_new_tokens: int  # sampling budget per candidate
    top_k: int = 3       # retrieved profile items (LaMP)


DATASETS = {d.name: d for d in [
    Dataset("LaMP_4", "lamp", "News", "train", "dev", 512),
    Dataset("LaMP_5", "lamp", "Scholarly", "train", "dev", 512),
    Dataset("LaMP_7", "lamp", "Tweet", "train", "dev", 512),
    Dataset("LaMPQA_Art_and_Entertainment", "lampqa", "Arts", "train", "dev", 512),
    Dataset("LaMPQA_Lifestyle_and_Personal_Development", "lampqa", "Lifestyle", "train", "dev", 512),
    Dataset("LaMPQA_Society_and_Culture", "lampqa", "Society", "train", "dev", 512),
    Dataset("XRec_amazon", "xrec", "Amazon", "train", "test", 128),
    Dataset("XRec_yelp", "xrec", "Yelp", "train", "test", 128),
    Dataset("XRec_google", "xrec", "Google", "train", "test", 128),
]}

TASK_GROUPS = [
    ("Short-form Generation", ["LaMP_4", "LaMP_5", "LaMP_7"]),
    ("Long-form QA", ["LaMPQA_Art_and_Entertainment",
                      "LaMPQA_Lifestyle_and_Personal_Development",
                      "LaMPQA_Society_and_Culture"]),
    ("Explainable Recommendation", ["XRec_amazon", "XRec_yelp", "XRec_google"]),
]


def get(name):
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    return DATASETS[name]


# ---------------------------------------------------------------------------
# file IO
# ---------------------------------------------------------------------------
def path(data_root, name, split, kind):
    return os.path.join(data_root, name, f"{split}_{kind}")


def load_json(p):
    with open(p) as f:
        return json.load(f)


def save_json(obj, p, indent=None):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=indent)


def load_questions(data_root, name, split):
    return load_json(path(data_root, name, split, "questions.json"))


def load_golds(data_root, name, split):
    return {str(g["id"]): g["output"]
            for g in load_json(path(data_root, name, split, "outputs.json"))["golds"]}


def load_samples(data_root, name, split):
    return {str(s["id"]): s["samples"]
            for s in load_json(path(data_root, name, split, "samples.json"))}


def load_scores(data_root, name, split):
    """{id: {"rouge1": [...], "rougeL": [...], "bleu": [...]}} per candidate."""
    return {str(s["id"]): s
            for s in load_json(path(data_root, name, split, "samples_scores.json"))}


def read_jsonl_scores(pattern_base):
    """Merge ``<base>.jsonl`` and ``<base>.shard*.jsonl`` into {id: scores}."""
    out = {}
    for p in [f"{pattern_base}.jsonl"] + sorted(glob.glob(f"{pattern_base}.shard*.jsonl")):
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:   # truncated last line of an interrupted run
                    continue
                out.setdefault(str(r["id"]), r["scores"])
    return out


def load_rm_scores(data_root, name, split, tag):
    return read_jsonl_scores(path(data_root, name, split, f"rmscores_{tag}"))


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
class PromptBuilder:
    """Builds the generation prompt of a question exactly as at sampling time.

    ``user_text(q)`` is the user turn (what the reward models score against);
    ``messages(q)`` adds the system prompt used for generation (LaMP-QA, XRec).
    """

    def __init__(self, name, tokenizer=None, max_prompt_length=512):
        self.ds = get(name)
        self._fn = None
        if self.ds.family == "lamp":
            if tokenizer is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(GENERATOR)
            # the profile budget is measured with the generator's tokenizer
            self._fn = lamp.make_prompt_fn(tokenizer, self.ds.top_k, max_prompt_length)
            self._task = name.replace("_", "-")

    def user_text(self, q):
        if self._fn is not None:
            return self._fn(q["input"], q["profile"], self._task)
        return q["input"]

    def messages(self, q):
        m = []
        if self.ds.family != "lamp" and q.get("system"):
            m.append({"role": "system", "content": q["system"]})
        m.append({"role": "user", "content": self.user_text(q)})
        return m


def embedding_texts(name, q):
    """(query text, profile text) whose last-token states become h_x and h_u."""
    ds = get(name)
    if ds.family == "lamp":
        task = name.replace("_", "-")
        query, items = lamp.retrieve(task, q["input"], q["profile"], ds.top_k)
        return query, lamp.profile_text(task, items)
    return q["q_text"], q["u_text"]
