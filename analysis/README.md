# Appendix analyses

Scripts that regenerate the analysis tables of the appendix from the prepared
data (`data/`) and the ranker results (`results/ranker/`). Run them from the
repository root; each prints markdown tables and writes a JSON file to
`results/analysis/`.

| Script | Paper table | Inputs |
|---|---|---|
| `bleu_transfer.py` | App. B.2, `tab:bleu_transfer` (BLEU of the Best-of-64 selection, ours vs. four reward models) | `results/ranker/<ds>/3M_mse.json`, `<split>_samples_scores.json`, `<split>_rmscores_{skywork,internlm,urm,armorm}.jsonl` |
| `metric_agreement.py` | App. B.2, `tab:metric_agreement` (ROUGE-L vs. BLEU agreement within and across pools) | `<split>_samples_scores.json` |
| `history_stats.py` | App. B.4, `tab:history_stats` (history size per user) | LaMP: `dev_questions.json`; LaMP-QA: `{train,dev}_questions.json` + Personalized-RewardBench profiles (Hugging Face Hub); XRec: `{ctx,test}_questions.json` |
| `history_stratified.py` | App. B.4, `tab:history_stratified` (Best-of-64 ROUGE-L by history bucket) | as `history_stats.py`, plus `results/ranker/<ds>/3M_mse.json` and `<split>_samples_scores.json` |
| `compute_cost.py` | App. C.4, `tab:compute_cost` and the per-user profile-embedding cost | eval split `questions`, `samples`, `cand_embeddings.npz`; the reward models and the generator (GPU) |

`common.py` holds the shared helpers (ranker selections, history-size
definitions, table formatting).

## Usage

```bash
python analysis/bleu_transfer.py      --data_root data --results_dir results
python analysis/metric_agreement.py   --data_root data --results_dir results
python analysis/history_stats.py      --data_root data --results_dir results
python analysis/history_stratified.py --data_root data --results_dir results

# timings depend on the hardware; the paper used one RTX A6000 and LaMP_5 prompts
CUDA_VISIBLE_DEVICES=0 python analysis/compute_cost.py --dataset LaMP_5 --stages ranker rm
CUDA_VISIBLE_DEVICES=0 python analysis/compute_cost.py --dataset LaMP_5 --stages embed profile
CUDA_VISIBLE_DEVICES=0 python analysis/compute_cost.py --dataset LaMP_5 --stages generate
```

## Notes on populations

- **`bleu_transfer.py`**: in the paper table each selector is averaged over the
  prompts it was evaluated on. The reward models scored the first 1,000
  evaluation prompts (all prompts for LaMP-QA); our ranker is evaluated on the
  full evaluation split. `--population common` restricts every row to the
  prompts shared by all selectors (ours then reads 0.0383 / 0.0691 / 0.1149 /
  0.0259 / 0.0217 / 0.0277 / 0.0804 / 0.0415 / 0.0328, and it stays the best
  selector on every dataset). `--metric rougeL` gives the same comparison under
  the training metric.
- **`metric_agreement.py`**: the first 64 candidates of every evaluation pool
  that is non-constant under both metrics.
- **`history_stats.py`**: one value per benchmark user: the evaluation questions
  for LaMP, all questions of the category for LaMP-QA, and the distinct test
  users for XRec (interactions in our `ctx` + `test` splits, a lower bound).
  `--population eval` gives one value per evaluation prompt instead.
- **`history_stratified.py`**: the evaluation prompts scored by the ranker,
  bucketed by `<10, 10-24, 25-49, 50-99, 100+`. XRec users have at most 22
  interactions, so they fall into the two lowest buckets. A dagger marks buckets
  with fewer than 10 prompts.
