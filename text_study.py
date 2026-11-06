"""Study 1: auditing text detectors and generation records.

Inputs are a corpus of aligned triples (reference, original generation, rewrite).
Implemented here:
  * source-grouped embedding classifier (all-MiniLM-L6-v2 + logistic regression,
    10 x grouped 5-fold CV; trained on reference vs. original generation, then the
    held-out rewrites are scored by the same fitted model), plus the item-level
    split it is compared against;
  * word-count baseline;
  * generation-log retrieval (max cosine similarity to logged generations): full log,
    symmetric source-held-out log, and the record-coverage sweep;
  * GPT-2 perplexity and pretrained RoBERTa detectors;
  * LLM judges: same-model numerical judge, same-model binary judge (both over an
    OpenAI-compatible endpoint) and the independent-family Qwen logit judge (mlx_lm).
"""
from __future__ import annotations

import os
import re

import numpy as np

from common import auroc, chat, classwise_bootstrap, source_cluster_bootstrap

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
COVERAGE_FRACTIONS = (1.0, 0.75, 0.5, 0.25, 0.1)


# --------------------------------------------------------------------------- #
# Embedding classifier                                                        #
# --------------------------------------------------------------------------- #
def source_groups(references):
    """Identical reference texts form one source group (with their counterparts)."""
    return np.unique(np.asarray(references, dtype=object).astype(str), return_inverse=True)[1]


def embed(texts):
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER, revision=ENCODER_REVISION, device="cpu")
    return [enc.encode(t, normalize_embeddings=True, show_progress_bar=False) for t in texts]


def cross_validated_scores(eH, eA, eZ, groups, grouped=True, repeats=10, folds=5):
    """Held-out P(generated) for every reference, generation and rewrite.

    grouped=True keeps all versions of a source on one side of every split
    (StratifiedGroupKFold); grouped=False is the item-level split. The classifier
    is fit on reference + original-generation embeddings only. Each item receives
    one held-out prediction per repetition; the mean over repetitions is returned.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    n = len(eH)
    X, y, gg = np.vstack([eH, eA]), np.r_[np.zeros(n), np.ones(n)], np.r_[groups, groups]
    runs = []
    for seed in range(repeats):
        cv = (StratifiedGroupKFold if grouped else StratifiedKFold)(folds, shuffle=True, random_state=seed)
        scores, seen = np.zeros((3, n)), np.zeros((3, n), int)
        for tr, te in (cv.split(X, y, gg) if grouped else cv.split(X, y)):
            if grouped:
                assert not set(gg[tr]) & set(gg[te]), "source group crosses the split"
            clf = LogisticRegression(max_iter=500).fit(X[tr], y[tr])
            hi, ai = te[te < n], te[te >= n] - n
            scores[0, hi] = clf.predict_proba(eH[hi])[:, 1]
            scores[1, ai] = clf.predict_proba(eA[ai])[:, 1]
            scores[2, ai] = clf.predict_proba(eZ[ai])[:, 1]  # rewrite of a held-out generation
            seen[0, hi] += 1
            seen[1:, ai] += 1
        assert np.all(seen == 1), "each item needs exactly one held-out prediction per repetition"
        runs.append(scores)
    return np.mean(runs, axis=0)


# --------------------------------------------------------------------------- #
# Generation-log retrieval                                                    #
# --------------------------------------------------------------------------- #
def retrieval(eH, eA, eZ, groups, seeds=25):
    """Score a query by its highest cosine similarity to the logged generations eA."""
    simH, simZ = eH @ eA.T, eZ @ eA.T
    same_source = groups[:, None] == groups[None, :]
    heldH, heldZ = np.where(same_source, -np.inf, simH), np.where(same_source, -np.inf, simZ)
    asym = simZ.copy()
    np.fill_diagonal(asym, -np.inf)  # own generation removed for rewrites only
    n = len(eA)
    coverage = {}
    for frac in COVERAGE_FRACTIONS:
        vals = []
        for seed in range(seeds):  # same retained log for both query classes
            kept = np.random.default_rng(seed).choice(n, max(1, round(frac * n)), replace=False)
            vals.append(auroc(simH[:, kept].max(1), simZ[:, kept].max(1)))
        coverage[str(frac)] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals))}
    held = source_cluster_bootstrap(heldH.max(1), heldZ.max(1), heldZ.max(1), groups)
    return {"full": auroc(simH.max(1), simZ.max(1)),
            "symmetric_source_heldout": held["clean"],
            "symmetric_source_heldout_ci": held["clean_ci"],
            "asymmetric_heldout": auroc(simH.max(1), asym.max(1)),
            "mean_max_similarity": {"reference": float(simH.max(1).mean()),
                                    "rewritten": float(simZ.max(1).mean())},
            "coverage": coverage}


# --------------------------------------------------------------------------- #
# Pretrained detectors                                                        #
# --------------------------------------------------------------------------- #
def gpt2_scores(texts):
    """Negative mean token NLL under GPT-2 (higher = lower perplexity)."""
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok, mdl = GPT2TokenizerFast.from_pretrained("gpt2"), GPT2LMHeadModel.from_pretrained("gpt2").eval()
    out = []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=512).input_ids
            out.append(-float(mdl(ids, labels=ids).loss) if ids.size(1) > 1 else 0.0)
    return np.array(out)


def roberta_scores(texts, model_name="Hello-SimpleAI/chatgpt-detector-roberta"):
    """Probability of the detector's last class (direction is fixed on clean data)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    out = []
    with torch.no_grad():
        for t in texts:
            x = tok(t, truncation=True, max_length=512, return_tensors="pt")
            out.append(float(torch.softmax(mdl(**x).logits[0], -1)[-1]))
    return np.array(out)


def oriented_bootstrap(sH, sA, sZ, n_boot=1000):
    """Choose score direction on reference vs. original generation, keep it for rewrites."""
    if auroc(sH, sA) < 0.5:
        sH, sA, sZ = -sH, -sA, -sZ
    return {"clean": classwise_bootstrap(sH, sA, n_boot), "post": classwise_bootstrap(sH, sZ, n_boot)}


# --------------------------------------------------------------------------- #
# LLM judges                                                                  #
# --------------------------------------------------------------------------- #
def numerical_judge(text, kind="scientific paper abstract"):
    """Same-model judge: integer 0-100 at temperature 0, scaled to [0, 1]; None if unparseable."""
    prompt = (f"You are an expert detector of AI-generated text. Below is a {kind}. "
              f"Estimate the probability that it was written by an AI language model rather "
              f"than by a human. Consider style, specificity, and signs of genuine effort. "
              f"Answer with ONLY an integer from 0 to 100 "
              f"(0 = certainly human-written, 100 = certainly AI-generated).\n\n"
              f"Text:\n\"\"\"\n{text}\n\"\"\"\n\nProbability (0-100):")
    reply = chat(prompt, 0.0, 8, endpoint_env="JUDGE_EP")
    m = re.search(r"\d{1,3}", reply or "")
    return min(100, max(0, int(m.group()))) / 100.0 if m else None


def binary_prompt(text, kind="scientific paper abstract"):
    return (f"Is the following {kind} AI-generated rather than human-written? "
            f"Answer with ONLY one word: Yes or No.\n\nText:\n{text[:1400]}")


def binary_frequency_judge(text, samples=4):
    """Same-model binary judge: share of 'Yes' over `samples` replies at temperature 0.8."""
    votes = []
    for _ in range(samples):
        r = (chat(binary_prompt(text), 0.8, 2, endpoint_env="JUDGE_EP") or "").strip().lower()
        if r.startswith("yes") or r.startswith("no"):
            votes.append(float(r.startswith("yes")))
    return float(np.mean(votes)) if votes else None


class QwenLogitJudge:
    """Independent-family judge: p(Yes) / (p(Yes) + p(No)) from next-token logits.

    Runs in-process with mlx_lm (Apple silicon); $JUDGE_MODEL_PATH points at the
    Qwen3.6-35B-A3B-4bit-DWQ weights. Chat template applied with thinking disabled.
    """

    def __init__(self, model_path=None):
        from mlx_lm import load
        self.model, self.tok = load(model_path or os.environ["JUDGE_MODEL_PATH"])
        first = lambda s: (self.tok.encode(s, add_special_tokens=False) or [None])[0]
        variants = lambda w: {w, " " + w, w.lower(), " " + w.lower(), w.upper()}
        self.yes = {t for t in map(first, variants("Yes")) if t is not None}
        self.no = {t for t in map(first, variants("No")) if t is not None}

    def __call__(self, text):
        import mlx.core as mx
        ids = self.tok.apply_chat_template([{"role": "user", "content": binary_prompt(text)}],
                                           add_generation_prompt=True, enable_thinking=False)
        probs = mx.softmax(self.model(mx.array(ids)[None])[0, -1].astype(mx.float32))
        py, pn = float(sum(probs[i] for i in self.yes)), float(sum(probs[i] for i in self.no))
        return py / (py + pn) if py + pn > 0 else None


JUDGES = {"numerical": lambda: numerical_judge, "binary": lambda: binary_frequency_judge,
          "qwen_logit": QwenLogitJudge}


def judge_corpus(corpus, judge):
    to_float = lambda s: np.nan if s is None else s
    return {k: np.array([to_float(judge(t)) for t in corpus[k]], float)
            for k in ("human", "ai", "humanized_ai")}


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
def run_corpus(corpus, cached_judge=None, detectors=False, judges=()):
    """All text-study measurements on one corpus dict {human, ai, humanized_ai}."""
    H, A, Z = corpus["human"], corpus["ai"], corpus["humanized_ai"]
    groups = source_groups(H)
    eH, eA, eZ = embed([H, A, Z])
    grouped = cross_validated_scores(eH, eA, eZ, groups, grouped=True)
    item = cross_validated_scores(eH, eA, eZ, groups, grouped=False)
    words = [np.array([len(t.split()) for t in x]) for x in (H, A, Z)]
    res = {"n": len(H), "source_groups": int(groups.max() + 1),
           "grouped_classifier": source_cluster_bootstrap(*grouped, groups),
           "item_level_classifier": {"clean": auroc(item[0], item[1]), "post": auroc(item[0], item[2])},
           "word_count": {"clean": auroc(words[0], words[1]), "post": auroc(words[0], words[2])},
           "retrieval": retrieval(eH, eA, eZ, groups)}
    if cached_judge is not None:  # shipped same-model numerical judge scores, complete triples
        res["numerical_judge"] = source_cluster_bootstrap(
            cached_judge["human"], cached_judge["ai"], cached_judge["humanized"], groups)
    if detectors:
        n = len(H)
        for name, fn in (("gpt2_perplexity", gpt2_scores), ("roberta_detector", roberta_scores)):
            s = fn(H + A + Z)
            res[name] = oriented_bootstrap(s[:n], s[n:2 * n], s[2 * n:])
    for name in judges:
        s = judge_corpus(corpus, JUDGES[name]())
        if name == "numerical":
            res["numerical_judge"] = source_cluster_bootstrap(s["human"], s["ai"], s["humanized_ai"], groups)
        else:
            res[name + "_judge"] = {"clean": classwise_bootstrap(s["human"], s["ai"], 2000),
                                    "post": classwise_bootstrap(s["human"], s["humanized_ai"], 2000)}
    return res
