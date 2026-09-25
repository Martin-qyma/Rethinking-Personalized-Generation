#!/usr/bin/env python3
"""Ranking-guided generation on the first ``--n`` evaluation prompts (Sec. 3.4, App. C.3).

``--grid default`` decodes with the paper's setting (tau=1, H_max=8, alpha0=5,
warmup 3) and with plain greedy decoding (alpha0=0) as reference; this is the
dashed line of Figure 3. ``--grid sensitivity`` adds the one-at-a-time sweep of
Table 7 (tau in {0.5,1,2,4}, H_max in {4,16}, alpha0 in {1,2,10}, warmup in {0,8}).
Paper setting: 100 prompts (all LaMP-QA evaluation prompts where fewer), greedy,
at most 48 new tokens.

Writes ``results/guided/<dataset>.json`` = {setting: {rougeL, steered_frac, mean_entropy, cfg, n}}.

    CUDA_VISIBLE_DEVICES=0 python scripts/guided_generation.py --dataset LaMP_7 \
        --ckpt checkpoints/ranker/LaMP_7/3M_mse --grid sensitivity
"""
import argparse
import os
import sys
from dataclasses import asdict, replace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rethinking import datasets as D
from rethinking.guided import GuidanceConfig, generate_guided
from rethinking.ranker import PersonalizedRanker


def settings(grid):
    base = GuidanceConfig()
    out = {"greedy (alpha0=0)": replace(base, alpha0=0.0), "default": base}
    if grid == "sensitivity":
        out.update({f"tau={t}": replace(base, tau=t) for t in (0.5, 2.0, 4.0)})
        out.update({f"H_max={h}": replace(base, h_max=h) for h in (4.0, 16.0)})
        out.update({f"alpha0={a}": replace(base, alpha0=a) for a in (1.0, 2.0, 10.0)})
        out.update({f"warmup={w}": replace(base, warmup=w) for w in (0, 8)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(D.DATASETS))
    ap.add_argument("--ckpt", required=True, help="ranker checkpoint dir (scripts/train_ranker.py)")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--model", default=D.GENERATOR)
    ap.add_argument("--grid", choices=["default", "sensitivity"], default="default")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from rouge_score import rouge_scorer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    split = D.get(args.dataset).eval_split
    npz = np.load(D.path(args.data_root, args.dataset, split, "cand_embeddings.npz"), allow_pickle=True)
    ids = [str(x) for x in npz["ids"]][: args.n]
    hx = torch.from_numpy(npz["question_embs"][: args.n].astype(np.float32)).to(device)
    hu = torch.from_numpy(npz["profile_embs"][: args.n].astype(np.float32)).to(device)
    questions = {str(q["id"]): q for q in D.load_questions(args.data_root, args.dataset, split)}
    golds = D.load_golds(args.data_root, args.dataset, split)

    tok = AutoTokenizer.from_pretrained(args.model)
    builder = D.PromptBuilder(args.dataset, tok)
    prompts = [tok(tok.apply_chat_template(builder.messages(questions[i]), tokenize=False,
                                           add_generation_prompt=True), return_tensors="pt").input_ids.to(device)
               for i in ids]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=device).eval()
    W = model.lm_head.weight.detach().float()
    ranker = PersonalizedRanker.load(args.ckpt, device)
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    results = {}
    for name, cfg in settings(args.grid).items():
        scores, stats = [], []
        for j, i in enumerate(ids):
            text, st = generate_guided(model, tok, ranker, prompts[j], hx[j:j + 1], hu[j:j + 1], cfg,
                                       args.max_new_tokens, W=W)
            scores.append(rouge.score(golds.get(i, ""), text)["rougeL"].fmeasure)
            stats.append(st)
        results[name] = dict(rougeL=float(np.mean(scores)), n=len(scores), cfg=asdict(cfg),
                             steered_frac=float(np.mean([s["steered"] / s["tokens"] for s in stats])),
                             mean_entropy=float(np.mean([s["mean_entropy"] for s in stats])))
        r = results[name]
        print(f"{name:20s} ROUGE-L {r['rougeL']:.4f}  steered {100 * r['steered_frac']:4.1f}% of steps  "
              f"mean entropy {r['mean_entropy']:.2f}", flush=True)

    out = args.out or os.path.join("results", "guided", f"{args.dataset}.json")
    D.save_json(results, out, indent=1)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
