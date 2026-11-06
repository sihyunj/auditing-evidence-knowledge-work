# Auditing Evidence in AI-Assisted Knowledge Work: Detector Choice, Record Coverage, and Combined Attacks

Core implementation of the paper's two studies.

## What the code reproduces

**Study 1: text detectors and generation records** (`text_study.py`).
120 arXiv abstracts and 71 Wikipedia introductions, each paired with a generated
version and one rewrite of that generation. AUROC, original generation / rewrite:

| Method | Original | Rewritten |
|---|---|---|
| Embedding classifier (all-MiniLM-L6-v2 + logistic regression, source-grouped CV) | 0.834 | 0.733 |
| Qwen logit judge (independent family) | 0.783 | 0.765 |
| DeepSeek numerical judge (same model as generator; 111 complete triples) | 0.540 | 0.422 |
| GPT-2 perplexity | 0.717 | 0.684 |
| RoBERTa detector | 0.829 | 0.564 |
| Word count | 0.576 | 0.631 |

Generation-log retrieval (max cosine similarity to logged generations) separates
rewrites from references with AUROC 0.9995 using the full log, falling to 0.825,
0.690, 0.597 and 0.575 at 75%, 50%, 25% and 10% log coverage (means over 25
subsamples).

**Study 2: graph fraud detectors under combined attacks** (`graph_study.py`).
YelpChi (fraud rate 14.5%) and Amazon (9.5%). BWGNN and CARE-style (CARE-GNN with
hard neighbour selection and RL thresholds replaced by learned soft similarity
weights) are trained on clean graphs and evaluated, fixed, under content forgery,
relation camouflage with edge multiplier β, and both combined (5 seeds). At β = 2,
BWGNN keeps AUROC 0.59 / 0.89 (YelpChi / Amazon) under relation camouflage alone,
but under the combined attack all four detector–dataset mean AUPRC values fall
below the fraud rate: BWGNN 0.111 / 0.086, CARE-style 0.096 / 0.085.

## Requirements

Python 3.10 or later. The default check needs only `numpy`. Full reruns:

```bash
pip install -r requirements.txt
```

Data (not included):

- **Text corpora** — the full evaluated corpora are `abstracts_texts.json`
  (120 arXiv abstracts) and `wiki_texts.json` (71 Wikipedia introductions),
  aligned lists of `human`, `ai`, `humanized_ai`. They are not distributed
  with this repository. A small sample of the evaluated corpora
  (`sample_corpus/`: the first 10 abstracts — 8% of 120 — and the first 8
  Wikipedia intros — 11% of 71 —, same format; 18 of 191 items, about 9%) is
  included so the text rerun can be tried; see "How to run" below. The full
  corpora can be rebuilt from the public sources with `run_audit.py corpus`
  (reference texts from the public arXiv Atom API, cs.CL, cs.LG, cs.IR, cs.CY,
  and the Wikipedia REST API, 140 fixed topics; generations and rewrites by
  DeepSeek V4 Flash at temperature 0.8), but generation is not seeded, so a
  rebuilt corpus will not match the evaluated one.
- **Graph benchmarks** `YelpChi.mat` and `Amazon.mat` from GADBench
  (<https://github.com/squareRoot3/GADBench>), which redistributes the CARE-GNN
  releases.
- **Models** are fetched from Hugging Face on first use:
  `sentence-transformers/all-MiniLM-L6-v2` (revision `1110a243`), `gpt2`,
  `Hello-SimpleAI/chatgpt-detector-roberta`. The DeepSeek judges need an
  OpenAI-compatible endpoint serving the generator; the Qwen judge needs `mlx-lm`
  and Qwen3.6-35B-A3B-4bit-DWQ weights.

## How to run

```bash
python run_audit.py
```

Reads the shipped results in `results/`, recomputes the DeepSeek numerical-judge
AUROCs from its per-item scores, and prints 129 headline values next to the values
reported in the paper, marking any that differ at the paper's precision (exit code
0 if all agree). Runs in under half a second (measured 0.2–0.4 s on a 4-CPU VM).

Full reruns write JSON to `outputs/` and print the same comparison:

```bash
python run_audit.py text  --corpus DIR [--detectors] [--judges numerical binary qwen_logit]
python run_audit.py graph --data DIR [--datasets yelp amazon] [--seeds 5]
python run_audit.py corpus --out DIR
```

- `text`: grouped and item-level embedding classifiers, word count, retrieval with
  the coverage sweep and source-held-out logs, and source-cluster bootstrap
  intervals; the DeepSeek numerical judge uses its shipped per-item scores.
  `--detectors` adds GPT-2 and RoBERTa; `--judges` re-scores the abstracts
  (`$JUDGE_EP`, `$LLM_KEY`, `$LLM_MODEL` for DeepSeek; `$JUDGE_MODEL_PATH` for
  Qwen). About 2 minutes on CPU without detectors or judges (measured 106–114 s
  on a 4-CPU VM; the MiniLM model downloads once, about 15 s, on first use).

Try the text rerun on the included sample first:

```bash
python run_audit.py text --corpus sample_corpus/
```

This is a smoke test: it runs the whole text pipeline in about 30 s (measured
31.8 s on a 4-CPU VM), but the numbers will differ from the paper because the
sample is small (10 abstracts, 8 Wikipedia intros — about 9% of the full
corpora) — a few values are
degenerate at this size, e.g. the item-level classifier on Wikipedia. The
DeepSeek numerical-judge rows show "(not run)" on the sample because the
shipped per-item judge scores belong to the full corpus. To verify the
paper's numbers, run the default check (`python run_audit.py`), which
recomputes the 129 headline values from the shipped results.
- `graph`: own- and neighbour-feature classifiers under content forgery and
  relation camouflage, then the β sweeps for BWGNN (relations; combined) and
  CARE-style (combined).
- `corpus`: rebuilds both text corpora (`$GEN_EP`, `$LLM_KEY`).

Files: `run_audit.py` (entry point), `text_study.py`, `graph_study.py`,
`corpus.py`, `common.py` (AUROC, bootstrap, endpoint call). In
`results/graph/robust_combined_caregnn_5seed.json` the key `CARE-GNN` holds
the CARE-style model.
