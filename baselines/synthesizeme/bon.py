#!/usr/bin/env python3
"""Best-of-N selection on the XRec evaluation subset with SynthesizeMe pairwise judges.

Single-elimination knockout over the first min(64, n) candidates in stored order,
matching the other methods' argmax over the first N. With sequential pairing, the
winner of the subtree covering leaves [0, 2^r) is known after round r, so one
63-match knockout yields the Best-of-N winner at every N in {1, 2, 4, ..., 64}.

Presentation order within a match is randomized deterministically (md5 of
qid|round|match); a Tie (which the package also returns on errors) advances the
candidate with the lower original index.

    --judge synthme   the user's fitted persona judge (falls back to the generic judge,
                      flagged in the output, for users whose fit failed)
    --judge default   the package's judge without a persona (generic-judge control)

Shardable and resumable; appends to ``<prep_dir>/<dataset>/bon_<judge>.jsonl``.

    python baselines/synthesizeme/bon.py --dataset XRec_amazon --judge synthme --shard 0 --num_shards 12 --port 8001
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from baselines.synthesizeme.common import apply_patches, default_judge_program, load_judge_program, make_lm
from rethinking import datasets as D
from rethinking.datasets import KS


def match_flip(qid, rnd, i):
    return hashlib.md5(f"{qid}|{rnd}|{i}".encode()).digest()[0] % 2 == 1


def run_tournament(program, conv, cands, qid):
    """Returns ({N: winner index among the first N}, n_ties, n_matches)."""
    from synthesizeme.utils.format_conv import format_conversation
    conv_str = format_conversation(conv)
    m = len(cands)
    entrants = list(range(m))
    prefix = {1: 0}
    ties = matches = rnd = 0
    while len(entrants) > 1:
        rnd += 1
        nxt = []
        for i in range(0, len(entrants) - 1, 2):
            a, b = entrants[i], entrants[i + 1]
            first, second = (b, a) if match_flip(qid, rnd, i) else (a, b)
            pred = program(conversation=conv_str, completion_one=cands[first], completion_two=cands[second])
            matches += 1
            if pred.preference == "First":
                w = first
            elif pred.preference == "Second":
                w = second
            else:
                ties += 1
                w = min(a, b)
            nxt.append(w)
        if len(entrants) % 2 == 1:
            nxt.append(entrants[-1])   # bye
        entrants = nxt
        k = 2 ** rnd
        if k not in prefix and k <= min(m, 64):
            prefix[k] = entrants[0]
    for k in KS:
        if k not in prefix:
            prefix[k] = entrants[0] if k >= m else prefix.get(k // 2, 0)
    return {str(k): prefix[k] for k in KS}, ties, matches


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--judge", choices=["synthme", "default"], default="synthme")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--prep_dir", default="results/baselines/synthesizeme")
    ap.add_argument("--ckpt_root", default="checkpoints/synthesizeme")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()

    ckpt_dir = os.path.join(args.ckpt_root, args.dataset)
    out_path = os.path.join(args.prep_dir, args.dataset, f"bon_{args.judge}.jsonl")
    ev = D.load_json(os.path.join(args.prep_dir, args.dataset, "eval_subset.json"))
    q = {str(x["id"]): x for x in D.load_questions(args.data_root, args.dataset, "test")}
    samples = D.load_samples(args.data_root, args.dataset, "test")
    npz = np.load(D.path(args.data_root, args.dataset, "test", "cand_embeddings.npz"), allow_pickle=True)
    n_of = {str(i): int(n) for i, n in zip(npz["ids"], npz["n_samples"])}

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except json.JSONDecodeError:
                    pass
    todo = [qid for i, qid in enumerate(ev["ids"]) if i % args.num_shards == args.shard and qid not in done]

    make_lm(args.port)
    apply_patches()
    default_prog = default_judge_program()
    cache, cache_lock, write_lock = {}, threading.Lock(), threading.Lock()

    def get_program(uid):
        if args.judge == "default":
            return default_prog, False
        with cache_lock:
            if uid not in cache:
                try:
                    cache[uid] = (load_judge_program(ckpt_dir, uid), False)
                except FileNotFoundError:
                    cache[uid] = (default_prog, True)    # fit failed -> generic fallback
            return cache[uid]

    def handle(qid):
        try:
            uid = str(ev["uid"][qid])
            cands = samples[qid][:min(64, n_of[qid])]
            program, fallback = get_program(uid)
            t0 = time.time()
            conv = [{"role": "system", "content": q[qid]["system"]},
                    {"role": "user", "content": q[qid]["input"]}]
            sel, ties, matches = run_tournament(program, conv, cands, qid)
            rec = {"id": qid, "uid": uid, "judge": args.judge, "sel": sel, "n_cands": len(cands),
                   "ties": ties, "matches": matches, "fallback": fallback,
                   "secs": round(time.time() - t0, 1)}
            with write_lock, open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:
            print(f"[shard {args.shard}] {qid} ERROR: {e!r}", flush=True)

    print(f"[{args.dataset}/{args.judge} shard {args.shard}] {len(todo)} prompts", flush=True)
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        for i, _ in enumerate(ex.map(handle, todo)):
            if (i + 1) % 10 == 0:
                print(f"[shard {args.shard}] {i + 1}/{len(todo)}", flush=True)
    print(f"[shard {args.shard}] done", flush=True)


if __name__ == "__main__":
    main()
