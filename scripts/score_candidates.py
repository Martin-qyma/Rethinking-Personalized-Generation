#!/usr/bin/env python3
"""Label every candidate against the user-written reference.

Writes ``<split>_samples_scores.json`` = [{id, rouge1: [...], rougeL: [...], bleu: [...]}]:
ROUGE-1 / ROUGE-L F-measure (``rouge_score``, Porter stemming, as in HF
``evaluate``) are the ranker's training targets and the evaluation metric;
sentence BLEU (``sacrebleu``, effective order) is the held-out metric of the cross-metric analysis.

    python scripts/score_candidates.py --dataset LaMP_7 --split dev
"""
import argparse
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rethinking import datasets as D

_ROUGE = _BLEU = None


def _init():
    global _ROUGE, _BLEU
    from rouge_score import rouge_scorer
    from sacrebleu.metrics import BLEU
    _ROUGE = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    # effective_order avoids zero scores for short texts without 4-gram matches
    _BLEU = BLEU(effective_order=True)


def _score(job):
    qid, cands, gold = job
    gold = (gold or "").strip()
    r1, rl, bl = [], [], []
    for c in cands:
        c = (c or "").strip()
        s = _ROUGE.score(gold, c)
        r1.append(s["rouge1"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
        bl.append(_BLEU.sentence_score(c, [gold]).score / 100.0 if c and gold else 0.0)
    return {"id": qid, "rouge1": r1, "rougeL": rl, "bleu": bl}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(D.DATASETS))
    ap.add_argument("--split", required=True)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--max_cand", type=int, default=None, help="score only the first K candidates")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    samples = D.load_json(D.path(args.data_root, args.dataset, args.split, "samples.json"))
    golds = D.load_golds(args.data_root, args.dataset, args.split)
    jobs = [(s["id"], s["samples"][: args.max_cand], golds.get(str(s["id"]), "")) for s in samples]
    print(f"{args.dataset}/{args.split}: {len(jobs)} pools, {sum(len(j[1]) for j in jobs)} candidates")
    with Pool(args.workers, initializer=_init) as pool:
        results = pool.map(_score, jobs, chunksize=16)
    out = D.path(args.data_root, args.dataset, args.split, "samples_scores.json")
    D.save_json(results, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
