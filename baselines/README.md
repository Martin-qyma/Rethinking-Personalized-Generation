# Personalized reward-model baselines (Appendix B.3, Table B.3)

Code for the comparison of our ranker with reward models that personalize from the
target user's own labeled preferences: PAL, VPL, PReF, LoRe, GPO and SynthesizeMe.
The comparison runs on XRec only, the one family whose users recur across
interactions so that per-user preference labels can be built.

| file | what it does |
|---|---|
| `models.py` | PAL-B, VPL, PReF, LoRe ports and our ranker (`rethinking.ranker.PersonalizedRanker`) under one interface |
| `run_personalized_rms.py` | seen-user protocol: mine context pairs, train, Best-of-N on held-out test interactions (Ours, VPL, PAL, PReF, LoRe) |
| `embed_meanpool.py` | mean-pooled (prompt, candidate) embeddings for GPO |
| `gpo.py` | GPO port (official transformer, meta-trained over users, in-context at test time) |
| `synthesizeme/` | SynthesizeMe with a Llama-3.1-8B-Instruct judge: `prep.py`, `fit.py`, `bon.py`, `collect.py`, `common.py` (package workarounds), `llama8b_nodemos.json`, `run.sh` |
| `collect_table.py` | prints Table B.3 from the result files |

## Protocol

* **Context split.** `ctx` holds up to eight training interactions of every XRec test
  user (the first eight of each user in `trn.pkl` order, written by
  `scripts/prepare_xrec.py`). Each ctx interaction gets **16** on-policy candidates
  (same generator and temperature as the main pools, 128 new tokens), scored against
  the user's reference.
* **Pairs.** From each ctx interaction, up to two pairs by ROUGE-L: (best, worst) and
  (2nd best, 2nd worst), kept if the gap exceeds 0.02; at most 16 pairs per user.
  Every pair-based method (PAL, VPL, PReF, LoRe, SynthesizeMe) sees exactly these
  pairs.
* **Training.** PAL / PReF / LoRe fit the free per-user vector jointly with their
  shared parameters (Bradley-Terry, full-batch Adam, lr 1e-3, 200 epochs; LoRe by
  alternating minimization; PReF with L2 on the user vectors). VPL infers its latent
  from the pairs with its set encoder (VAE objective). GPO is meta-trained over
  users and receives a user's labeled ctx pools in context at test time. Our ranker
  is retrained on the same ctx pools with its own objective (pointwise MSE on
  within-pool z-scored ROUGE-L; no pairs, no user ids) and conditions only on the
  profile embedding, which is why its numbers differ slightly from the main tables.
* **Evaluation.** Best-of-N over the first N of the 64 test candidates of each user's
  held-out test interaction; the table reports ROUGE-L at N = 64. PAL, VPL, PReF,
  LoRe and ours are evaluated on all test interactions of users with at least one
  context pair; GPO on those of users with at least two ctx interactions (a slightly
  smaller population, since it needs no mined pairs); SynthesizeMe on a
  1,000-prompt random subset (seed 0) of the first population, because its
  knockout tournament costs 63 LLM judge calls per prompt.
* **Swap diagnostic** (reported by `collect_table.py --detail`): the change in
  ROUGE-L@64 when every user is given another user's fitted parameters / context set.

## Reproducing Table B.3

All commands run from the repository root; `--data_root data` is the default.

**1. Data** (per dataset `XRec_amazon`, `XRec_yelp`, `XRec_google`; shown for Amazon).

```bash
python scripts/prepare_xrec.py --src_dir /path/to/XRec/data         # test, train, ctx splits
GPUS=0,1,2,3 bash scripts/run_sampling.sh XRec_amazon test           # 256 candidates (first 64 used)
GPUS=0,1,2,3 N=16 bash scripts/run_sampling.sh XRec_amazon ctx       # 16 candidates per ctx interaction
python scripts/score_candidates.py --dataset XRec_amazon --split test
python scripts/score_candidates.py --dataset XRec_amazon --split ctx
python scripts/embed.py --dataset XRec_amazon --split test           # h_x, h_u, h_y
python scripts/embed.py --dataset XRec_amazon --split ctx
python baselines/embed_meanpool.py --dataset XRec_amazon --split test --max_cand 64   # GPO only
python baselines/embed_meanpool.py --dataset XRec_amazon --split ctx  --max_cand 16
```

Files used: `{ctx,test}_{questions.json, samples_scores.json, cand_embeddings.npz,
meanpool_embeddings.npz}`, plus `{ctx,test}_samples.json` for SynthesizeMe.

**2. Trained reward models** (one GPU; a few minutes per dataset):

```bash
for ds in XRec_amazon XRec_yelp XRec_google; do
    python baselines/run_personalized_rms.py --dataset $ds    # -> results/baselines/personalized_rms/$ds.json
    python baselines/gpo.py --dataset $ds                     # -> results/baselines/gpo/$ds.json
done
```

**3. SynthesizeMe** (needs the `SynthesizeMe` package and vLLM servers of
Llama-3.1-8B-Instruct; see the header of `synthesizeme/run.sh`):

```bash
bash baselines/synthesizeme/run.sh prep
bash baselines/synthesizeme/run.sh fit              # persona per eval user (resumable)
bash baselines/synthesizeme/run.sh bon synthme      # persona judge
bash baselines/synthesizeme/run.sh bon default      # generic judge without persona (control)
bash baselines/synthesizeme/run.sh collect          # -> results/baselines/synthesizeme/results.json
```

**4. Table**

```bash
python baselines/collect_table.py [--detail]
```

Expected output (seed 0):

| Dataset | Ours | VPL | PAL | PReF | LoRe | GPO | SynthesizeMe |
|---|---|---|---|---|---|---|---|
| Amazon | 0.2641 | 0.2628 | 0.2596 | 0.2598 | 0.2503 | 0.2690 | 0.1962 |
| Yelp | 0.1906 | 0.1916 | 0.1902 | 0.1791 | 0.1798 | 0.1957 | 0.1687 |
| Google | 0.1685 | 0.1697 | 0.1614 | 0.1526 | 0.1544 | 0.1743 | 0.1381 |

The generic judge without a persona scores 0.2033 / 0.1677 / 0.1378.

## Faithfulness notes

Each port keeps the mechanism that defines the method, i.e. how the per-user
parameters are obtained; all methods except GPO read the same frozen generator
embeddings as our ranker and are sized near its 2.8M parameters (hidden width 256).

* **PAL-B** (official `main_pal_b_unseen.py`, config `b-dim1536-k2-...-mlp2`): K = 2
  prompt-conditioned prototypes, softmax mixture weights per user, cosine ("angle")
  score with a learnable logit scale; user weights fit by gradient descent.
* **VPL**: pair encoder + self-attention set encoder (batched version of
  `vae_utils.SequenceEncoder`) -> (mu, logvar) clamped as in the official code; the
  latent is inferred, never fit per user; loss = BT + 0.01 KL; test uses the mean.
* **PReF**: J = 16 basis reward features phi(x, y); preference logit on the pair
  difference (their `Model.forward`); users by L2-regularized logistic regression.
* **LoRe**: linear reward on the response embedding with a rank-8 shared basis and
  simplex user weights, temperature 100, alternating minimization. The prompt
  embedding cancels in LoRe's chosen-minus-rejected difference, which does not affect
  within-prompt ranking. Its cosine regularizer toward a base-RM direction is kept
  but inactive (no base RM).
* **GPO** (official `models/tnp.py`, `models/gpo.py`, `configs/gpo.yaml`): mean-pooled
  (prompt, candidate) embeddings as in their Appendix D; within-question ROUGE-L
  normalized to a distribution as the target; transformer without positional
  encodings where every position attends only to context; per-question softmax over
  predicted means with a Gaussian NLL. Episodes: one user, a random split of their
  ctx interactions into up to 6 context and 4 target questions; 20,000 steps, Adam
  1e-4. At test time the user's first 6 ctx interactions are the context.
* **SynthesizeMe**: the released package with three workarounds documented in
  `synthesizeme/common.py` (explicit temperature 0 under dspy 3.x, the empty-demo
  fallback bug, and demo-free persona synthesis prompt for the 8B judge). Persona
  search budget 5 (package default 10; part of the users were fit with 10). Best-of-N
  by a single-elimination tournament whose prefix winners give every N at once.

## Reproducibility

With seed 0 the scripts reproduce the paper values exactly: the RM runner keeps the
operation order of the original runs (module construction order, the random stream
of pair orientations including one unused initial draw, population-std z-scoring of
our ranker's targets), and GPO consumes its NumPy episode stream in the original
order. Results were checked on all three datasets against the numbers above.
