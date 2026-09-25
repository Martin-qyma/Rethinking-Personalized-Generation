#!/usr/bin/env python3
"""Fit a SynthesizeMe persona judge for every user of the evaluation subset.

For each user the package bootstraps judge demos from the user's labeled pairs,
synthesizes a persona from the verified judgements, and saves the persona-prompted
judge program (``<uid>.json`` + ``<uid>_persona.txt``). Shardable and resumable: a
user is skipped when its program already exists. Run several shards in parallel,
each against its own vLLM server port.

    python baselines/synthesizeme/fit.py --dataset XRec_amazon --shard 0 --num_shards 12 --port 8001
"""
import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from baselines.synthesizeme.common import apply_patches, make_lm, nodemos_prompt_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--prep_dir", default="results/baselines/synthesizeme",
                    help="output dir of prep.py")
    ap.add_argument("--ckpt_root", default="checkpoints/synthesizeme")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--num_workers", type=int, default=4, help="threads inside each dspy optimizer")
    ap.add_argument("--num_search_candidates", type=int, default=5,
                    help="persona search budget (package default 10; the paper used 5-10)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ckpt_dir = os.path.join(args.ckpt_root, args.dataset) + "/"   # the package concatenates paths
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(args.prep_dir, args.dataset, "user_prefs.json")) as f:
        prefs = json.load(f)
    with open(os.path.join(args.prep_dir, args.dataset, "eval_subset.json")) as f:
        users = sorted(set(json.load(f)["uid"].values()))
    todo = [u for i, u in enumerate(users) if i % args.num_shards == args.shard]

    lm = make_lm(args.port)
    apply_patches()
    from synthesizeme.personalrm.synthesizeme import SynthesizeMe
    program_path = nodemos_prompt_path()

    fail_log = os.path.join(ckpt_dir, f"failures_shard{args.shard}.jsonl")
    done = skipped = failed = 0
    for u in todo:
        uid = str(u)
        if os.path.exists(os.path.join(ckpt_dir, f"{uid}.json")):
            skipped += 1
            continue
        rows = prefs.get(uid) or []
        if len(rows) < 2:           # bon.py falls back to the generic judge for these users
            with open(fail_log, "a") as f:
                f.write(json.dumps({"uid": uid, "err": "too_few_pairs", "n": len(rows)}) + "\n")
            failed += 1
            continue
        t0 = time.time()
        try:
            rm = SynthesizeMe(user_id=uid, model_id=None, model_url=None, lm=lm,
                              persona_synthesis_program_path=program_path, output_dir=ckpt_dir,
                              num_workers=args.num_workers,
                              num_search_candidates=args.num_search_candidates, seed=args.seed)
            rm.fit(rows)
            rm.save(ckpt_dir)
            done += 1
            print(f"[shard {args.shard}] uid={uid} fitted in {time.time() - t0:.0f}s ({done} done)", flush=True)
        except Exception as e:
            failed += 1
            with open(fail_log, "a") as f:
                f.write(json.dumps({"uid": uid, "err": repr(e), "tb": traceback.format_exc()[-2000:]}) + "\n")
            print(f"[shard {args.shard}] uid={uid} FAILED: {e!r}", flush=True)
    print(f"[shard {args.shard}] finished: {done} fitted, {skipped} skipped, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
