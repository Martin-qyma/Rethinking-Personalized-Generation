#!/usr/bin/env python3
"""Finetune Skywork-Reward-V2-Llama-3.1-8B on a dataset's candidate pools (Sec. 4.3).

This is the model-size comparator of Figure 4: the strongest generalist reward
model, fully finetuned per dataset with the same pointwise objective as our
ranker. Recipe (App. C.2):

* data: 1,500 training prompts (all available for LaMP-QA), 8 candidates each
  (best, worst and 6 random of the first 64), targets = ROUGE-L z-scored within
  the 64-candidate pool;
* loss: MSE between the group-centred predictions and the targets (the scalar
  head has no bias, so a prompt-dependent offset cannot be learned; Best-of-N is
  invariant to it). Before training, the head is divided by the within-pool std
  of the pretrained scores so predictions start on the target scale;
* optimisation: full finetune in bf16 with FSDP, AdamW lr 2e-5, 10% linear
  warmup, weight decay 0.01, gradient clipping 1.0, 24 prompts per optimizer
  step, 1 epoch (2 for LaMP-QA), seed 42, sequences truncated to 2,048 tokens
  with the prompt truncated from the left so the candidate always fits.

    accelerate launch --config_file configs/fsdp.yaml --num_processes 2 \
        scripts/finetune_reward_model.py --dataset LaMP_7 --out checkpoints/rm_ft/LaMP_7
    # LaMP-QA: --epochs 2 --prompts_per_micro 3 --grad_accum 4

Then score with ``scripts/score_reward_model.py --model_path <out> --fit_response``.
"""
import argparse
import json
import math
import os
import random
import sys

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, set_seed
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rethinking import datasets as D
from rethinking.reward_models import REWARD_MODELS, chat_text, fit_prompt


def build_items(args, prompt_of):
    """Training prompts with (candidate, z-scored ROUGE-L) targets."""
    split = D.get(args.dataset).train_split
    questions = D.load_questions(args.data_root, args.dataset, split)
    golds = D.load_golds(args.data_root, args.dataset, split)
    pools = D.load_samples(args.data_root, args.dataset, split)
    labels = D.load_scores(args.data_root, args.dataset, split)
    rng = random.Random(args.seed)
    rng.shuffle(questions)
    items, n_seen = [], 0
    for q in questions:
        i = str(q["id"])
        if i not in pools or i not in labels or not golds.get(i, "").strip():
            continue
        ranked = sorted([(c, s) for c, s in zip(pools[i][:64], labels[i]["rougeL"][:64])
                         if c != golds[i] and c.strip()], key=lambda x: x[1])
        if len(ranked) < 2:
            continue
        # the prompt budget counts every usable prompt; degenerate pools are dropped below
        n_seen += 1
        vals = np.array([s for _, s in ranked])
        if len(ranked) >= 4 and vals.std() > 1e-6:
            pick = [ranked[-1], ranked[0]] + rng.sample(ranked[1:-1], min(args.cands_per_prompt - 2,
                                                                          len(ranked) - 2))
            items.append({"prompt": prompt_of(q),
                          "cands": [(c, float((s - vals.mean()) / vals.std())) for c, s in pick]})
        if n_seen >= args.n_prompts:
            break
    return items


class Collator:
    def __init__(self, tok, max_length):
        self.tok, self.max_length = tok, max_length

    def __call__(self, batch):
        texts, sizes, targets = [], [], []
        for it in batch:
            for c, z in it["cands"]:
                texts.append(chat_text(self.tok, fit_prompt(self.tok, it["prompt"], c, self.max_length), c))
                targets.append(z)
            sizes.append(len(it["cands"]))
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "targets": torch.tensor(targets), "sizes": sizes}


def centred_mse(scores, targets, sizes):
    loss, off = 0.0, 0
    for k in sizes:
        s = scores[off:off + k]
        loss = loss + ((s - s.mean() - targets[off:off + k]) ** 2).mean()
        off += k
    return loss / len(sizes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(D.DATASETS))
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--model", default=REWARD_MODELS["skywork"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_prompts", type=int, default=1500)
    ap.add_argument("--cands_per_prompt", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--prompts_per_micro", type=int, default=6, help="prompts per forward pass per process")
    ap.add_argument("--grad_accum", type=int, default=2, help="2 processes x 6 x 2 = 24 prompts per step")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--calib_prompts", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    acc = Accelerator(gradient_accumulation_steps=args.grad_accum, mixed_precision="bf16")
    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    builder = D.PromptBuilder(args.dataset)
    items = build_items(args, builder.user_text)
    collator = Collator(tok, args.max_length)
    loader = DataLoader(items, batch_size=args.prompts_per_micro, shuffle=True, collate_fn=collator)
    acc.print(f"{args.dataset}: {len(items)} training prompts")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", num_labels=1)
    model.config.pad_token_id = tok.pad_token_id

    # scale calibration: divide the (bias-free) head by the within-pool std of the
    # pretrained scores, measured on the main process before FSDP shards the model
    calib = None
    if acc.is_main_process:
        model.to(acc.device).eval()
        stds = []
        with torch.no_grad():
            for it in items[: args.calib_prompts]:
                b = collator([it])
                s = model(input_ids=b["input_ids"].to(acc.device),
                          attention_mask=b["attention_mask"].to(acc.device)).logits[:, 0].float()
                stds.append(float(s.std(unbiased=False)))
        sigma = float(np.sqrt(np.mean(np.square(stds))))
        model.score.weight.data.div_(sigma)
        model.to("cpu").train()
        torch.cuda.empty_cache()
        calib = {"sigma_within": sigma, "n": len(stds)}
    calib = broadcast_object_list([calib], from_process=0)[0]
    acc.print(f"calibration: within-pool std {calib['sigma_within']:.3f} over {calib['n']} pools")

    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = math.ceil(math.ceil(len(loader) / acc.num_processes) / args.grad_accum) * args.epochs
    # accelerate steps a prepared scheduler once per process, so size it in ticks
    ticks = acc.num_processes
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * steps) * ticks, max(steps, 1) * ticks)
    model, opt, loader, sched = acc.prepare(model, opt, loader, sched)

    step = 0
    for ep in range(args.epochs):
        model.train()
        for batch in loader:
            with acc.accumulate(model):
                s = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, 0].float()
                loss = centred_mse(s, batch["targets"].to(s.device), batch["sizes"])
                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step(); sched.step(); opt.zero_grad()
            if acc.sync_gradients:
                step += 1
                if step % 5 == 0:
                    acc.print(f"epoch {ep + 1} step {step}/{steps} loss {float(loss):.4f}", flush=True)

    acc.wait_for_everyone()
    state = acc.get_state_dict(model)
    if acc.is_main_process:
        acc.unwrap_model(model).save_pretrained(args.out, state_dict=state, safe_serialization=True)
        tok.save_pretrained(args.out)
        with open(os.path.join(args.out, "recipe.json"), "w") as f:
            json.dump({**vars(args), "train_prompts": len(items), "optimizer_steps": steps,
                       "processes": acc.num_processes, "calibration": calib}, f, indent=1)
        print(f"saved -> {args.out}")
    acc.wait_for_everyone()


if __name__ == "__main__":
    main()
