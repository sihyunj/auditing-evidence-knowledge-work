"""Shared helpers: AUROC, bootstrap procedures, JSON I/O, LLM endpoint call."""
from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load_json(path):
    # The shipped judge scores encode missing values as bare NaN, which json accepts.
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def auroc(neg, pos):
    """AUROC of scores `pos` (positive class) against `neg`, ties counted as 1/2.

    Mann-Whitney rank formula; identical to sklearn's roc_auc_score.
    """
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    _, inv, cnt = np.unique(np.r_[neg, pos], return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    n, p = len(neg), len(pos)
    return float((ranks[n:].sum() - p * (p + 1) / 2.0) / (n * p))


def mean_sd(values, digits=4):
    return [round(float(np.mean(values)), digits), round(float(np.std(values)), digits)]


def classwise_bootstrap(neg, pos, n_boot, seed=0):
    """AUROC bootstrap resampling each class separately (NaN scores dropped).

    Returns [mean, sd, 2.5th percentile, 97.5th percentile].
    """
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    neg, pos = neg[~np.isnan(neg)], pos[~np.isnan(pos)]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(neg), len(neg))
        j = rng.integers(0, len(pos), len(pos))
        vals.append(auroc(neg[i], pos[j]))
    return [round(float(np.mean(vals)), 3), round(float(np.std(vals)), 3),
            round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def source_cluster_bootstrap(h, a, z, groups, n_boot=4000, seed=20261106):
    """Paired source-cluster bootstrap for AUROC(h vs a), AUROC(h vs z) and their drop.

    h, a, z are aligned per source entry (reference, original generation, rewrite).
    Entries with any missing score are dropped. Source groups are resampled with
    replacement and every version of a group is kept together. Predictions are
    held fixed (no refitting), so intervals describe source-sampling uncertainty.
    """
    h, a, z, groups = (np.asarray(v) for v in (h, a, z, groups))
    h, a, z = h.astype(float), a.astype(float), z.astype(float)
    keep = np.isfinite(h) & np.isfinite(a) & np.isfinite(z)
    h, a, z, groups = h[keep], a[keep], z[keep], groups[keep]
    _, g = np.unique(groups, return_inverse=True)
    n_groups = int(g.max() + 1)
    rng = np.random.default_rng(seed)
    # weight of each entry in each resample = multiplicity of its source group
    w = rng.multinomial(n_groups, np.ones(n_groups) / n_groups, size=n_boot)[:, g].astype(float)
    den = w.sum(1) ** 2

    def pairwise(pos):  # [i, j] = 1 if pos_i > h_j, 1/2 if tied
        return (pos[:, None] > h[None, :]).astype(float) + 0.5 * (pos[:, None] == h[None, :])

    clean = np.sum((w @ pairwise(a)) * w, axis=1) / den
    post = np.sum((w @ pairwise(z)) * w, axis=1) / den
    ci = lambda v: [float(x) for x in np.percentile(v, [2.5, 97.5])]
    return {"n_complete": int(len(h)), "n_source_groups": n_groups, "bootstrap_resamples": n_boot,
            "clean": auroc(h, a), "post": auroc(h, z), "drop": auroc(h, a) - auroc(h, z),
            "clean_ci": ci(clean), "post_ci": ci(post), "drop_ci": ci(clean - post)}


def chat(prompt, temperature, max_tokens, endpoint_env="GEN_EP", tries=3):
    """One chat completion from an OpenAI-compatible endpoint.

    Endpoint URL from $GEN_EP / $JUDGE_EP, served model name from $LLM_MODEL
    (default deepseek-v4-flash), API key from $LLM_KEY. Returns None on failure.
    """
    endpoint = os.environ[endpoint_env]
    body = {"model": os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
            "temperature": temperature, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    headers = {"Authorization": f"Bearer {os.environ['LLM_KEY']}",
               "Content-Type": "application/json"}
    for _ in range(tries):
        try:
            req = urllib.request.Request(endpoint, method="POST", headers=headers,
                                         data=json.dumps(body).encode())
            resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
            return resp["choices"][0]["message"]["content"].strip()
        except Exception:
            time.sleep(1)
    return None
