#!/usr/bin/env python3
"""Sample candidate pools with the generator (vLLM), sharded and resumable.

Paper setting (App. C.2): Qwen2.5-7B-Instruct, 256 samples per prompt,
temperature 1.0; at most 512 new tokens (LaMP, LaMP-QA) or 128 (XRec). Every
downstream stage uses the first 64 candidates of each pool in stored order.

One process per GPU generates ``questions[shard::num_shards]`` and appends one
``{"id", "samples"}`` line per question to a shard JSONL, so an interrupted run
resumes where it stopped. ``--merge`` joins the shards into ``<split>_samples.json``
in question order. ``scripts/run_sampling.sh`` launches all shards and merges.

    CUDA_VISIBLE_DEVICES=0 python scripts/sample.py --dataset LaMP_7 --split dev \
        --num_shards 4 --shard 0
    python scripts/sample.py --dataset LaMP_7 --split dev --num_shards 4 --merge
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rethinking import datasets as D


def shard_path(args, i):
    return D.path(args.data_root, args.dataset, args.split, f"samples.shard{i}of{args.num_shards}.jsonl")


def done_ids(p):
    out = set()
    if os.path.exists(p):
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("samples"):
                out.add(r["id"])
    return out


def merge(args):
    records, seen = [], set()
    for i in range(args.num_shards):
        for line in open(shard_path(args, i)):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["id"] not in seen:
                seen.add(r["id"])
                records.append(r)
    order = {q["id"]: k for k, q in enumerate(D.load_questions(args.data_root, args.dataset, args.split))}
    records.sort(key=lambda r: order.get(r["id"], len(order)))
    out = D.path(args.data_root, args.dataset, args.split, "samples.json")
    D.save_json(records, out)
    print(f"merged {len(records)} pools -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(D.DATASETS))
    ap.add_argument("--split", required=True)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--model", default=D.GENERATOR)
    ap.add_argument("--num_samples", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=None, help="default: per-dataset budget")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--chunk_size", type=int, default=512, help="questions per generate() call")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--merge", action="store_true", help="merge finished shards and exit")
    args = ap.parse_args()
    if args.merge:
        return merge(args)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    ds = D.get(args.dataset)
    questions = D.load_questions(args.data_root, args.dataset, args.split)[: args.max_examples]
    out = shard_path(args, args.shard)
    done = done_ids(out)
    todo = [q for q in questions[args.shard::args.num_shards] if q["id"] not in done]
    print(f"[shard {args.shard}/{args.num_shards}] {len(todo)} to generate ({len(done)} done)", flush=True)
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = D.PromptBuilder(args.dataset, tok)
    llm = LLM(model=args.model, tensor_parallel_size=1, dtype="bfloat16", trust_remote_code=True,
              gpu_memory_utilization=args.gpu_memory_utilization)
    params = SamplingParams(n=args.num_samples, temperature=args.temperature,
                            max_tokens=args.max_tokens or ds.max_new_tokens)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for start in range(0, len(todo), args.chunk_size):
        chunk = todo[start:start + args.chunk_size]
        texts = [tok.apply_chat_template(prompts.messages(q), tokenize=False, add_generation_prompt=True)
                 for q in chunk]
        results = llm.generate(texts, params)
        with open(out, "a") as f:
            for q, r in zip(chunk, results):
                f.write(json.dumps({"id": q["id"], "samples": [o.text.strip() for o in r.outputs]}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(f"[shard {args.shard}] {start + len(chunk)}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
