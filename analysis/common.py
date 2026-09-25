"""Helpers shared by the appendix analyses: ranker selections, history sizes, tables."""
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rethinking import datasets as D  # noqa: E402

ALL_DATASETS = [name for _, names in D.TASK_GROUPS for name in names]


# ---------------------------------------------------------------------------
# ranker selections
# ---------------------------------------------------------------------------
def load_selection(results_dir, name, run="3M_mse", n=64):
    """{question id: index of the candidate our ranker selects among the first n}.

    Read from ``<results_dir>/ranker/<name>/<run>.json`` written by
    ``scripts/train_ranker.py``. Its ids are the evaluation prompts with a
    non-constant ROUGE-L pool (the population of every ranker number in the paper).
    """
    res = D.load_json(os.path.join(results_dir, "ranker", name, f"{run}.json"))
    pq = res["per_q"]
    return dict(zip([str(i) for i in pq["ids"]], pq["sel_idx"][str(n)]))


# ---------------------------------------------------------------------------
# user history sizes
# ---------------------------------------------------------------------------
HISTORY_UNIT = {"lamp": {"LaMP_4": "article--headline pairs", "LaMP_5": "abstract--title pairs",
                         "LaMP_7": "past tweets"},
                "lampqa": "past questions", "xrec": "past reviews"}
PROFILE_TEXT = {"lamp": "top-3 BM25-retrieved {items}", "lampqa": "top-6 BM25-retrieved questions",
                "xrec": "pre-written user summary"}


def history_unit(name):
    unit = HISTORY_UNIT[D.get(name).family]
    return unit[name] if isinstance(unit, dict) else unit


def profile_description(name):
    ds = D.get(name)
    items = {"LaMP_7": "tweets"}.get(name, "pairs")
    return PROFILE_TEXT[ds.family].format(items=items)


def _lampqa_profile_sizes(name):
    """{question id: number of past questions of the asker} for a LaMP-QA category.

    Read from the Personalized-RewardBench release, the same file
    ``scripts/prepare_lampqa.py`` takes the references from.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download
    cat = name[len("LaMPQA_"):]
    df = pd.read_parquet(hf_hub_download("QiyaoMa/Personalized-RewardBench",
                                         f"{cat}/test-00000-of-00001.parquet", repo_type="dataset"))
    return {str(i): len(p) for i, p in zip(df["id"], df["profile"])}


def history_sizes(data_root, name):
    """Number of historical interactions behind each question of a dataset.

    Returns ``(per_question, per_user)``:

    * ``per_question`` maps every evaluation prompt to its user's history size;
    * ``per_user`` is the list of sizes over the benchmark's users, one value per
      user (the population of the history statistics table).

    Definitions per family:

    * LaMP: ``len(profile)`` of the question (one user per question); users are
      the evaluation-split questions.
    * LaMP-QA: number of past questions of the asker in the benchmark profile;
      users are all questions of the category (ranker training and evaluation
      splits together, one asker per question).
    * XRec: interactions of the user present in our ``ctx`` + ``test`` splits,
      a lower bound of the platform history (the benchmark reports 18--25 per
      user); users are the distinct users of the test split.
    """
    ds = D.get(name)
    eval_q = D.load_questions(data_root, name, ds.eval_split)
    if ds.family == "lamp":
        per_q = {str(q["id"]): len(q.get("profile") or []) for q in eval_q}
        return per_q, list(per_q.values())
    if ds.family == "lampqa":
        qs = [q for split in (ds.train_split, ds.eval_split)
              for q in D.load_questions(data_root, name, split)]
        ids = [str(q["id"]) for q in qs]
        # written by scripts/prepare_lampqa.py; older question files fall back to the hub
        sizes = ({str(q["id"]): q["history_size"] for q in qs} if all("history_size" in q for q in qs)
                 else _lampqa_profile_sizes(name))
        missing = [i for i in ids if i not in sizes]
        if missing:
            raise KeyError(f"{name}: {len(missing)} questions without a benchmark profile")
        return {str(q["id"]): sizes[str(q["id"])] for q in eval_q}, [sizes[i] for i in ids]
    # xrec
    counts = collections.Counter()
    for split in ("ctx", ds.eval_split):
        counts.update(int(q["uid"]) for q in D.load_questions(data_root, name, split))
    per_q = {str(q["id"]): counts[int(q["uid"])] for q in eval_q}
    users = {int(q["uid"]) for q in eval_q}
    return per_q, [counts[u] for u in sorted(users)]


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def column_labels(names):
    return [D.get(n).label for n in names]
