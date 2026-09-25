#!/usr/bin/env python3
"""Inference cost of Best-of-N selection (inference cost table).

Measures wall-clock time and peak GPU memory of each stage for queries with N
candidates on the local GPU:

    ranker    scoring the pool with our ranker on recycled embeddings
    rm        scoring the pool with each generalist reward model (HF transformers)
    embed     the separate vLLM last-token pooling pass over the candidates
              (only paid when the states are not read off the generation pass)
    profile   standalone embedding of the user profile text behind h_u, one per
              user, for one dataset of each family (``--profile_datasets``)
    generate  sampling the pool itself with the generator (vLLM)

The paper numbers were measured on one RTX A6000 with LaMP_5 (Scholarly) prompts;
timings depend on hardware, software versions and batch settings. Ranker and
reward-model memory is the peak allocated by PyTorch (weights included); for the
vLLM stages it is the device memory in use, which is dominated by vLLM's
pre-allocated pool (``--gpu_memory_utilization``).

``embed``/``profile`` and ``generate`` need two different vLLM engines; run
``generate`` in its own invocation.

    CUDA_VISIBLE_DEVICES=0 python analysis/compute_cost.py --dataset LaMP_5 --stages ranker rm
    CUDA_VISIBLE_DEVICES=0 python analysis/compute_cost.py --dataset LaMP_5 --stages embed profile
    CUDA_VISIBLE_DEVICES=0 python analysis/compute_cost.py --dataset LaMP_5 --stages generate
"""
import argparse
import gc
import os
import time

import numpy as np
import torch

from common import D, markdown_table

GiB = 2 ** 30
PROFILE_DATASETS = ["LaMP_5", "XRec_amazon", "LaMPQA_Art_and_Entertainment"]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def device_used_gib():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / GiB


def eval_prompts(args, name=None):
    """The first ``--n_prompts`` evaluation questions of a dataset that have a pool."""
    name = name or args.dataset
    split = D.get(name).eval_split
    pools = D.load_samples(args.data_root, name, split)
    qs = [q for q in D.load_questions(args.data_root, name, split) if str(q["id"]) in pools]
    qs = qs[: args.n_prompts]
    return qs, [pools[str(q["id"])][: args.n_cand] for q in qs]


def cost_row(stage, seconds, n_queries, n_cands, mem_gib, **extra):
    return dict(stage=stage, ms_per_candidate=1000 * seconds / n_cands,
                ms_per_query=1000 * seconds / n_queries, peak_mem_gib=mem_gib,
                n_queries=n_queries, n_candidates=n_cands, **extra)


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def time_ranker(args, device):
    from rethinking.pools import load_pools
    from rethinking.ranker import SIZES, PersonalizedRanker
    torch.cuda.reset_peak_memory_stats()
    pools = load_pools(args.data_root, args.dataset, D.get(args.dataset).eval_split, max_cand=args.n_cand)
    if args.ranker_ckpt:
        model = PersonalizedRanker.load(args.ranker_ckpt, device)
    else:   # timing does not depend on the weights
        model = PersonalizedRanker(pools["dim"], **SIZES["3M"]).to(device).eval()
    P = min(args.n_prompts, len(pools["ids"]))
    hx, hu, hy = (pools[k][:P].to(device) for k in ("hx", "hu", "hy"))
    with torch.no_grad():
        for i in range(min(P, 8)):                       # warm-up
            model(hx[i:i + 1], hu[i:i + 1], hy[i:i + 1])
        sync()
        t0 = time.perf_counter()
        for _ in range(args.repeats):
            for i in range(P):                           # one query at a time
                model(hx[i:i + 1], hu[i:i + 1], hy[i:i + 1])
        sync()
        dt = time.perf_counter() - t0
    n_q = P * args.repeats
    row = cost_row("Ranker scoring (ours)", dt, n_q, n_q * hy.shape[1],
                   torch.cuda.max_memory_allocated() / GiB, params=model.num_parameters())
    del model, hx, hu, hy, pools
    gc.collect(); torch.cuda.empty_cache()
    return row


def time_reward_models(args, device):
    from rethinking.reward_models import RewardModel
    qs, cands = eval_prompts(args)
    builder = D.PromptBuilder(args.dataset)
    prompts = [builder.user_text(q) for q in qs]
    rows = []
    for tag in args.rms:
        torch.cuda.reset_peak_memory_stats()
        rm = RewardModel(tag, device=device, batch_size=args.rm_batch_size)
        rm.score(prompts[0], cands[0])                   # warm-up
        sync()
        t0 = time.perf_counter()
        for p, c in zip(prompts, cands):
            rm.score(p, c)
        sync()
        dt = time.perf_counter() - t0
        rows.append(cost_row(f"Reward model: {tag}", dt, len(qs), sum(map(len, cands)),
                             torch.cuda.max_memory_allocated() / GiB,
                             params=sum(p.numel() for p in rm.model.parameters())))
        print(f"  {tag}: {rows[-1]['ms_per_query']:.0f} ms/query", flush=True)
        del rm
        gc.collect(); torch.cuda.empty_cache()
    return rows


def pooling_engine(args):
    from vllm import LLM
    return LLM(model=args.model, runner="pooling", convert="embed", dtype="bfloat16",
               tensor_parallel_size=1, trust_remote_code=True,
               gpu_memory_utilization=args.gpu_memory_utilization)


def time_embedding(args, llm):
    qs, cands = eval_prompts(args)
    flat = [c if c.strip() else " " for cs in cands for c in cs]
    llm.embed(flat[: min(len(flat), 64)])                # warm-up
    t0 = time.perf_counter()
    llm.embed(flat)
    dt = time.perf_counter() - t0
    return cost_row("Embedding pass (candidates)", dt, len(qs), len(flat), device_used_gib())


def time_profiles(args, llm):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = []
    for name in args.profile_datasets:
        qs, _ = eval_prompts(args, name)
        texts = [D.embedding_texts(name, q)[1] or " " for q in qs]
        n_tok = [len(tok(t).input_ids) for t in texts]
        llm.embed(texts[:8])                             # warm-up
        t0 = time.perf_counter()
        llm.embed(texts)
        dt = time.perf_counter() - t0
        rows.append(dict(dataset=name, n=len(texts), ms_per_profile=1000 * dt / len(texts),
                         tokens_median=float(np.median(n_tok)), tokens_p90=float(np.percentile(n_tok, 90)),
                         device_mem_gib=device_used_gib()))
    return rows


def time_generation(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.model)
    builder = D.PromptBuilder(args.dataset, tok)
    qs, _ = eval_prompts(args)
    texts = [tok.apply_chat_template(builder.messages(q), tokenize=False, add_generation_prompt=True)
             for q in qs]
    llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=1, trust_remote_code=True,
              gpu_memory_utilization=args.gpu_memory_utilization)
    params = SamplingParams(n=args.n_cand, temperature=1.0, max_tokens=D.get(args.dataset).max_new_tokens)
    llm.generate(texts[:1], params)                      # warm-up
    t0 = time.perf_counter()
    llm.generate(texts, params)
    dt = time.perf_counter() - t0
    return cost_row("Candidate generation", dt, len(qs), len(qs) * args.n_cand, device_used_gib())


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--dataset", default="LaMP_5", choices=list(D.DATASETS))
    ap.add_argument("--stages", nargs="+", default=["ranker", "rm"],
                    choices=["ranker", "rm", "embed", "profile", "generate"])
    ap.add_argument("--n_prompts", type=int, default=100, help="queries timed per stage")
    ap.add_argument("--n_cand", type=int, default=64, help="candidates per query (N)")
    ap.add_argument("--repeats", type=int, default=10, help="passes over the queries for the ranker")
    ap.add_argument("--ranker_ckpt", default=None, help="checkpoint dir (default: randomly initialized 3M ranker)")
    ap.add_argument("--rms", nargs="+", default=["skywork", "internlm", "urm", "armorm"])
    ap.add_argument("--rm_batch_size", type=int, default=32)
    ap.add_argument("--profile_datasets", nargs="+", default=PROFILE_DATASETS, choices=list(D.DATASETS))
    ap.add_argument("--model", default=D.GENERATOR)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--out", default=None, help="JSON output (default: <results_dir>/analysis/compute_cost_<stages>.json)")
    args = ap.parse_args()
    if "generate" in args.stages and {"embed", "profile"} & set(args.stages):
        ap.error("run --stages generate in its own invocation (it needs a separate vLLM engine)")
    if not torch.cuda.is_available():
        ap.error("a GPU is required")
    device = torch.cuda.current_device()

    rows, profiles = [], []
    if "ranker" in args.stages:
        rows.append(time_ranker(args, device))
    if "rm" in args.stages:
        rm_rows = time_reward_models(args, device)
        rows += rm_rows
        rows.append(dict(stage=f"Reward models (mean of {len(rm_rows)})",
                         **{k: float(np.mean([r[k] for r in rm_rows]))
                            for k in ("ms_per_candidate", "ms_per_query", "peak_mem_gib")}))
    if {"embed", "profile"} & set(args.stages):
        llm = pooling_engine(args)
        if "embed" in args.stages:
            rows.append(time_embedding(args, llm))
        if "profile" in args.stages:
            profiles = time_profiles(args, llm)
    if "generate" in args.stages:
        rows.append(time_generation(args))

    gpu = torch.cuda.get_device_name(device)
    print(f"## Inference cost on {gpu} ({args.dataset}, N={args.n_cand})\n")
    if rows:
        print(markdown_table(["Stage", "Per candidate (ms)", "Per query (ms)", "Peak GPU memory (GiB)"],
                             [[r["stage"], f"{r['ms_per_candidate']:.4g}", f"{r['ms_per_query']:.4g}",
                               f"{r['peak_mem_gib']:.2f}"] for r in rows]))
    if profiles:
        print("\n### Standalone profile embedding (h_u), one per user\n")
        print(markdown_table(["Dataset", "Profiles", "ms per profile", "Tokens (median / p90)"],
                             [[D.get(p["dataset"]).label, p["n"], f"{p['ms_per_profile']:.1f}",
                               f"{p['tokens_median']:.0f} / {p['tokens_p90']:.0f}"] for p in profiles]))

    out = args.out or os.path.join(args.results_dir, "analysis", f"compute_cost_{'_'.join(args.stages)}.json")
    D.save_json(dict(gpu=gpu, dataset=args.dataset, n_cand=args.n_cand, stages=rows, profiles=profiles),
                out, indent=1)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
