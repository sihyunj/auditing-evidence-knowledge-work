"""Study 2: graph-fraud detectors when content and relations change together.

Data: YelpChi.mat / Amazon.mat (GADBench release). Labelled nodes: all of YelpChi;
Amazon nodes from index 3305 on (standard CARE-GNN subset).

Attacks (applied after clean training, to every labelled fraud node):
  * content forgery   - each fraud node's features <- features of a random benign node;
  * relation camouflage - each fraud node adds round(beta * union-graph degree) edges
    to random benign nodes, symmetrically, in every relation (repeats allowed);
  * combined           - both, with full content forgery at every beta > 0.

Detectors:
  * own-feature and neighbour-feature gradient-boosted classifiers
    (neighbour features = per-relation neighbour means + log degree);
  * BWGNN (Beta-wavelet spectral GNN) on the union graph;
  * CARE-style: CARE-GNN with its hard neighbour selection and RL threshold
    replaced by learned soft similarity weights, per relation.

Splits: stratified 70/30 train/test; GNNs carve validation out of the training block
(50/20/30). Metrics on the test split: AUPRC (average precision) and AUROC.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

BETAS = [0.0, 0.25, 0.5, 1.0, 2.0]
RELATIONS = {"yelp": ["net_rur", "net_rtr", "net_rsr"], "amazon": ["net_upu", "net_usu", "net_uvu"]}
MAT_FILES = {"yelp": "YelpChi.mat", "amazon": "Amazon.mat"}
AMAZON_LABELLED_START = 3305
DEGREE_CAP = 64


# --------------------------------------------------------------------------- #
# Data and graph features                                                     #
# --------------------------------------------------------------------------- #
def load_dataset(name, data_dir):
    """Return features X (N, F), labels y, edges {'homo', relations...} as (2, E), labelled idx."""
    mat = sio.loadmat(str(Path(data_dir) / MAT_FILES[name]))
    X = np.asarray(mat["features"].todense(), dtype=np.float32)
    y = np.asarray(mat["label"]).reshape(-1).astype(int)
    edges = {}
    for key in ["homo"] + RELATIONS[name]:
        coo = mat[key].tocoo()
        edges[key] = np.vstack([coo.row, coo.col]).astype(np.int64)
    start = AMAZON_LABELLED_START if name == "amazon" else 0
    return X, y, edges, np.arange(start, len(y))


def adjacency(ei, N):
    return sp.csr_matrix((np.ones(ei.shape[1], np.float32), (ei[0], ei[1])), shape=(N, N))


def neighbour_mean(X, ei, N):
    A = adjacency(ei, N)
    deg = np.asarray(A.sum(1)).ravel()
    return (np.asarray(A @ X) / np.where(deg > 0, deg, 1.0)[:, None]).astype(np.float32), deg


def neighbour_features(X, edges, N):
    """Per-relation neighbour means + log union-graph degree (no own-node features)."""
    parts = [neighbour_mean(X, ei, N)[0] for ei in edges.values()]
    parts.append(np.log1p(neighbour_mean(X[:, :1], edges["homo"], N)[1])[:, None])
    return np.hstack(parts).astype(np.float32)


# --------------------------------------------------------------------------- #
# Attacks                                                                     #
# --------------------------------------------------------------------------- #
def content_forgery(X, fraud, benign, rng):
    Xa = X.copy()
    Xa[fraud] = X[rng.choice(benign, len(fraud), replace=True)]
    return Xa


def relation_camouflage(edges, fraud, benign, homo_deg, beta, rng):
    if beta <= 0:
        return edges
    src, dst = [], []
    for i in fraud:
        k = int(round(beta * homo_deg[i]))
        if k <= 0:
            continue
        tgt = rng.choice(benign, k, replace=True).tolist()
        src += [i] * k + tgt
        dst += tgt + [i] * k
    add = np.array([src, dst], dtype=np.int64)
    return {rel: np.hstack([ei, add]) for rel, ei in edges.items()}


def camouflage_rng(seed, beta):
    return np.random.default_rng(1000 * seed + int(beta * 10))


def forgery_rng(seed, beta):
    return np.random.default_rng(10000 * seed + int(beta * 100))


def cap_degree(ei, N, k=DEGREE_CAP, seed=0):
    """Keep <= k random out-edges per node, then symmetrise and deduplicate."""
    order = np.argsort(ei[0], kind="stable")
    src, dst = ei[0][order], ei[1][order]
    _, starts = np.unique(src, return_index=True)
    ends = np.append(starts[1:], len(src))
    rng = np.random.default_rng(seed)
    keep = [np.arange(s, e) if e - s <= k else rng.choice(e - s, size=k, replace=False) + s
            for s, e in zip(starts, ends)]
    keep = np.concatenate(keep)
    ks, kd = src[keep], dst[keep]
    pairs = np.unique(np.stack([np.r_[ks, kd], np.r_[kd, ks]], axis=1), axis=0)
    return pairs.T.astype(np.int64)


# --------------------------------------------------------------------------- #
# Own-feature vs neighbour-feature classifiers                                #
# --------------------------------------------------------------------------- #
def feature_classifier_study(X, y, edges, lab, seeds=5):
    """Content forgery on own vs neighbour features; relation camouflage on neighbour features."""
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    from sklearn.metrics import average_precision_score as ap, roc_auc_score as roc

    N = len(y)
    benign, fraud = lab[y[lab] == 0], lab[y[lab] == 1]
    homo_deg = neighbour_mean(X[:, :1], edges["homo"], N)[1]
    nb_clean = neighbour_features(X, edges, N)
    forgery = {t: {c: {"auprc": [], "auroc": []} for c in ("clean", "attack")} for t in ("own", "neighbour")}
    camo = {f"{b:.2f}": {"auprc": [], "auroc": []} for b in BETAS}
    for s in range(seeds):
        tr, te = train_test(lab, y, s)
        Xa = content_forgery(X, fraud, benign, np.random.default_rng(s))
        for tier, (clean, attacked) in (("own", (X, Xa)),
                                        ("neighbour", (nb_clean, neighbour_features(Xa, edges, N)))):
            clf = HGB(random_state=s).fit(clean[tr], y[tr])
            for cond, F in (("clean", clean), ("attack", attacked)):
                p = clf.predict_proba(F[te])[:, 1]
                forgery[tier][cond]["auprc"].append(ap(y[te], p))
                forgery[tier][cond]["auroc"].append(roc(y[te], p))
        clf = HGB(random_state=s).fit(nb_clean[tr], y[tr])
        for b in BETAS:
            F = nb_clean if b == 0 else neighbour_features(
                X, relation_camouflage(edges, fraud, benign, homo_deg, b, camouflage_rng(s, b)), N)
            p = clf.predict_proba(F[te])[:, 1]
            camo[f"{b:.2f}"]["auprc"].append(ap(y[te], p))
            camo[f"{b:.2f}"]["auroc"].append(roc(y[te], p))
    ms = lambda v: [round(float(np.mean(v)), 3), round(float(np.std(v)), 3)]
    return {"content_forgery": {t: {c: {m: ms(v) for m, v in d.items()} for c, d in tiers.items()}
                                for t, tiers in forgery.items()},
            "neighbour, relations": {m: {b: ms(camo[b][m]) for b in camo} for m in ("auprc", "auroc")}}


def train_test(lab, y, seed):
    from sklearn.model_selection import train_test_split
    return train_test_split(lab, test_size=0.3, random_state=seed, stratify=y[lab])


# --------------------------------------------------------------------------- #
# GNNs                                                                        #
# --------------------------------------------------------------------------- #
def build_models():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class BWGNN(nn.Module):
        """Beta-wavelet band-pass filter bank on the symmetric-normalised union graph."""

        def __init__(self, in_dim, hidden, d=3, dropout=0.5):
            super().__init__()
            self.d, self.dropout = d, dropout
            self.lin_in = nn.Linear(in_dim, hidden)
            # filter p ~ (L/2)^p (1 - L/2)^(d-p), expanded as coefficients of powers of L/2
            self.filters = []
            for p in range(d + 1):
                c = [0.0] * (d + 1)
                for k in range(d - p + 1):
                    c[p + k] += math.comb(d - p, k) * (-1) ** k
                self.filters.append([math.comb(d, p) / 2 ** d * v for v in c])
            self.filter_mlps = nn.ModuleList(nn.Linear(hidden, hidden) for _ in self.filters)
            self.lin_out = nn.Sequential(nn.Linear(hidden * len(self.filters), hidden), nn.ReLU(),
                                         nn.Dropout(dropout), nn.Linear(hidden, 2))

        def set_graph(self, graph, N):
            row, col = graph
            dinv = torch.bincount(row, minlength=N).float().clamp(min=1).pow(-0.5)
            self.A = torch.sparse_coo_tensor(graph, dinv[row] * dinv[col], (N, N)).coalesce()

        def forward(self, x):
            h = F.dropout(F.relu(self.lin_in(x)), self.dropout, self.training)
            powers = [h]
            for _ in range(self.d):  # (L/2) h = (h - A h) / 2
                powers.append((powers[-1] - torch.sparse.mm(self.A, powers[-1])) * 0.5)
            outs = [F.relu(mlp(sum(c * powers[i] for i, c in enumerate(coeffs) if c != 0.0)))
                    for coeffs, mlp in zip(self.filters, self.filter_mlps)]
            return self.lin_out(torch.cat(outs, dim=1))

    def weighted_mean(x, ei, w, N):
        """out_j = (x_j + sum_{i->j} w_ij x_i) / (1 + sum_{i->j} w_ij)."""
        row, col = ei
        out = torch.zeros(N, x.size(1), dtype=x.dtype).index_add_(0, col, x[row] * w.unsqueeze(-1))
        den = torch.zeros(N, dtype=x.dtype).index_add_(0, col, w)
        return (out + x) / (den + 1.0).clamp(min=1e-8).unsqueeze(-1)

    class CAREStyle(nn.Module):
        """CARE-GNN adaptation: label-aware similarity ell(x) = sigmoid(a.x) gives soft edge
        weights 1 - |ell_i - ell_j| (in place of hard top-p selection with RL thresholds);
        per-relation two-layer aggregation combined by learned softmax relation weights."""

        def __init__(self, in_dim, hidden, relations, dropout=0.5):
            super().__init__()
            self.relations, self.dropout = relations, dropout
            self.simhead = nn.Linear(in_dim, 1)
            self.lin1 = nn.ModuleDict({r: nn.Linear(in_dim, hidden) for r in relations})
            self.lin2 = nn.ModuleDict({r: nn.Linear(hidden, hidden) for r in relations})
            self.relw = nn.Parameter(torch.zeros(len(relations)))
            self.out = nn.Linear(hidden, 2)

        def set_graph(self, graph, N):
            self.edges, self.N = graph, N

        def forward(self, x):
            ell = torch.sigmoid(self.simhead(x)).squeeze(-1)
            rw = torch.softmax(self.relw, 0)
            weight = lambda ei: (1.0 - (ell[ei[0]] - ell[ei[1]]).abs()).clamp(min=0.0)
            h = sum(rw[k] * F.relu(self.lin1[r](weighted_mean(x, self.edges[r], weight(self.edges[r]), self.N)))
                    for k, r in enumerate(self.relations))
            h = F.dropout(h, self.dropout, self.training)
            h2 = sum(rw[k] * self.lin2[r](weighted_mean(h, self.edges[r], weight(self.edges[r]), self.N))
                     for k, r in enumerate(self.relations))
            return self.out(F.relu(h2))

    return BWGNN, CAREStyle


def gnn_attack_sweep(kind, X, y, edges, lab, seed, combined, betas=BETAS,
                     hidden=64, epochs=150, eval_every=5, patience=8):
    """Train one GNN on the clean graph; evaluate it, fixed, on attacked graphs at each beta."""
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import average_precision_score as ap, roc_auc_score as roc
    from sklearn.model_selection import train_test_split

    BWGNN, CAREStyle = build_models()
    N = len(y)
    benign, fraud = lab[y[lab] == 0], lab[y[lab] == 1]
    homo_deg = neighbour_mean(X[:, :1], edges["homo"], N)[1]
    tr_all, te = train_test(lab, y, seed)
    tr, va = train_test_split(tr_all, test_size=2.0 / 7.0, random_state=seed, stratify=y[tr_all])
    mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True)
    sd = np.where(sd > 1e-8, sd, 1.0)
    features = lambda M: torch.from_numpy(((M - mu) / sd).astype(np.float32))

    relations = [r for r in edges if r != "homo"]
    if kind == "bwgnn":  # message passing on the degree-capped union graph
        to_graph = lambda ed: torch.from_numpy(cap_degree(ed["homo"], N))
        model = BWGNN(X.shape[1], hidden)
    else:                # CARE-style: degree-capped per-relation graphs
        to_graph = lambda ed: {r: torch.from_numpy(cap_degree(ed[r], N)) for r in relations}
        model = CAREStyle(X.shape[1], hidden, relations)

    # Clean training with class-weighted loss; early stopping on validation AUPRC.
    # (Model is initialised before seeding, as in the runs behind the shipped results.)
    torch.manual_seed(seed)
    np.random.seed(seed)
    x, y_t, tr_t = features(X), torch.from_numpy(y.astype(np.int64)), torch.from_numpy(tr)
    clean_graph = to_graph(edges)
    model.set_graph(clean_graph, N)
    w = torch.tensor([1.0, (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)])
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    best, state, wait = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(x)[tr_t], y_t[tr_t], weight=w).backward()
        opt.step()
        if (ep + 1) % eval_every and ep != epochs - 1:
            continue
        model.eval()
        with torch.no_grad():
            v = ap(y[va], F.softmax(model(x), 1)[va, 1].numpy())
        if v > best:
            best, wait = v, 0
            state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state)
    model.eval()

    auprc, auroc = {}, {}
    for b in betas:
        if b == 0:
            graph, xb = clean_graph, x
        else:
            graph = to_graph(relation_camouflage(edges, fraud, benign, homo_deg, b, camouflage_rng(seed, b)))
            xb = features(content_forgery(X, fraud, benign, forgery_rng(seed, b))) if combined else x
        model.set_graph(graph, N)
        with torch.no_grad():
            p = F.softmax(model(xb), 1)[:, 1].numpy()
        auprc[f"{b:.2f}"] = float(ap(y[te], p[te]))
        auroc[f"{b:.2f}"] = float(roc(y[te], p[te]))
    return auprc, auroc


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
MODEL_LABELS = {"bwgnn": "BWGNN", "care": "CARE-style"}
# (model, attack) pairs reported in the paper
SWEEPS = [("bwgnn", "relations"), ("bwgnn", "combined"), ("care", "combined")]


def run(data_dir, datasets=("yelp", "amazon"), sweeps=SWEEPS, seeds=5, feature_classifiers=True):
    out = {"betas": BETAS, "seeds": seeds}
    for name in datasets:
        X, y, edges, lab = load_dataset(name, data_dir)
        res = {"fraud_rate": round(float(y[lab].mean()), 3), "n_labelled": int(len(lab))}
        if feature_classifiers:
            res.update(feature_classifier_study(X, y, edges, lab, seeds))
            print(f"[{name}] feature classifiers done", flush=True)
        for kind, attack in sweeps:
            runs = []
            for s in range(seeds):
                t0 = time.perf_counter()
                runs.append(gnn_attack_sweep(kind, X, y, edges, lab, s, combined=(attack == "combined")))
                print(f"[{name}/{kind}/{attack} seed {s}] {time.perf_counter() - t0:.0f}s", flush=True)
            ms = lambda m, b: [round(float(np.mean([r[m][b] for r in runs])), 4),
                               round(float(np.std([r[m][b] for r in runs])), 4)]
            res[f"{MODEL_LABELS[kind]}, {attack}"] = {
                "auprc": {f"{b:.2f}": ms(0, f"{b:.2f}") for b in BETAS},
                "auroc": {f"{b:.2f}": ms(1, f"{b:.2f}") for b in BETAS}}
        out[name] = res
    return out
