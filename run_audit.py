"""Single entry point.

    python run_audit.py [check]                 headline numbers from results/ (numpy only)
    python run_audit.py text   --corpus DIR     rerun Study 1 on the text corpora
    python run_audit.py graph  --data DIR       rerun Study 2 on YelpChi.mat / Amazon.mat
    python run_audit.py corpus --out DIR        rebuild the text corpora (needs an LLM endpoint)

Every mode prints each headline quantity next to the value reported in the paper.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from common import RESULTS, auroc, load_json, save_json

# --------------------------------------------------------------------------- #
# Paper values: (label, results dict, key path, reported value, decimals)     #
# --------------------------------------------------------------------------- #
BETA_KEYS = ["0.00", "0.25", "0.50", "1.00", "2.00"]

ABSTRACT_ROWS = [
    ("Embedding classifier, grouped: original", ("grouped_classifier", "clean"), 0.834, 3),
    ("Embedding classifier, grouped: rewritten", ("grouped_classifier", "post"), 0.733, 3),
    ("Embedding classifier, grouped: paired drop", ("grouped_classifier", "drop"), 0.101, 3),
    ("  drop 95% CI lower", ("grouped_classifier", "drop_ci", 0), 0.079, 3),
    ("  drop 95% CI upper", ("grouped_classifier", "drop_ci", 1), 0.126, 3),
    ("Embedding classifier, item-level split: original", ("item_level_classifier", "clean"), 0.741, 3),
    ("Embedding classifier, item-level split: rewritten", ("item_level_classifier", "post"), 0.625, 3),
    ("Word count: original", ("word_count", "clean"), 0.576, 3),
    ("Word count: rewritten", ("word_count", "post"), 0.631, 3),
    ("GPT-2 perplexity: original", ("gpt2_perplexity", "clean", 0), 0.717, 3),
    ("GPT-2 perplexity: rewritten", ("gpt2_perplexity", "post", 0), 0.684, 3),
    ("RoBERTa detector: original", ("roberta_detector", "clean", 0), 0.829, 3),
    ("RoBERTa detector: rewritten", ("roberta_detector", "post", 0), 0.564, 3),
    ("DeepSeek numerical judge: complete triples", ("numerical_judge", "n_complete"), 111, 0),
    ("DeepSeek numerical judge: original", ("numerical_judge", "clean"), 0.540, 3),
    ("DeepSeek numerical judge: rewritten", ("numerical_judge", "post"), 0.422, 3),
    ("DeepSeek numerical judge: paired drop", ("numerical_judge", "drop"), 0.118, 3),
    ("  drop 95% CI lower", ("numerical_judge", "drop_ci", 0), 0.049, 3),
    ("  drop 95% CI upper", ("numerical_judge", "drop_ci", 1), 0.184, 3),
    ("DeepSeek binary judge: original", ("binary_judge", "clean", 0), 0.469, 3),
    ("DeepSeek binary judge: rewritten", ("binary_judge", "post", 0), 0.498, 3),
    ("Qwen logit judge: original", ("qwen_logit_judge", "clean", 0), 0.783, 3),
    ("Qwen logit judge: rewritten", ("qwen_logit_judge", "post", 0), 0.765, 3),
    ("  rewritten 95% CI lower", ("qwen_logit_judge", "post", 2), 0.702, 3),
    ("  rewritten 95% CI upper", ("qwen_logit_judge", "post", 3), 0.821, 3),
    ("Retrieval, full log", ("retrieval", "full"), 0.9995, 4),
    ("  mean max similarity, rewritten", ("retrieval", "mean_max_similarity", "rewritten"), 0.959, 3),
    ("  mean max similarity, reference", ("retrieval", "mean_max_similarity", "reference"), 0.734, 3),
    ("Retrieval, 75% log coverage: mean", ("retrieval", "coverage", "0.75", "mean"), 0.825, 3),
    ("Retrieval, 75% log coverage: SD over 25 subsamples", ("retrieval", "coverage", "0.75", "sd"), 0.011, 3),
    ("Retrieval, 50% log coverage: mean", ("retrieval", "coverage", "0.5", "mean"), 0.690, 3),
    ("Retrieval, 50% log coverage: SD over 25 subsamples", ("retrieval", "coverage", "0.5", "sd"), 0.009, 3),
    ("Retrieval, 25% log coverage: mean", ("retrieval", "coverage", "0.25", "mean"), 0.597, 3),
    ("Retrieval, 25% log coverage: SD over 25 subsamples", ("retrieval", "coverage", "0.25", "sd"), 0.015, 3),
    ("Retrieval, 10% log coverage: mean", ("retrieval", "coverage", "0.1", "mean"), 0.575, 3),
    ("Retrieval, 10% log coverage: SD over 25 subsamples", ("retrieval", "coverage", "0.1", "sd"), 0.017, 3),
    ("Retrieval, symmetric source held out", ("retrieval", "symmetric_source_heldout"), 0.583, 3),
    ("Retrieval, asymmetric removal", ("retrieval", "asymmetric_heldout"), 0.253, 3),
]
WIKI_ROWS = [
    ("Embedding classifier, grouped: original", ("grouped_classifier", "clean"), 0.843, 3),
    ("Embedding classifier, grouped: rewritten", ("grouped_classifier", "post"), 0.865, 3),
    ("Embedding classifier, item-level split: original", ("item_level_classifier", "clean"), 0.513, 3),
    ("Embedding classifier, item-level split: rewritten", ("item_level_classifier", "post"), 0.545, 3),
    ("Word count: original", ("word_count", "clean"), 0.873, 3),
    ("Word count: rewritten", ("word_count", "post"), 0.942, 3),
    ("Retrieval, full log", ("retrieval", "full"), 0.994, 3),
    ("Retrieval, symmetric source held out", ("retrieval", "symmetric_source_heldout"), 0.581, 3),
    ("Retrieval, 50% log coverage: mean", ("retrieval", "coverage", "0.5", "mean"), 0.646, 3),
    ("Retrieval, 50% log coverage: SD over 25 subsamples", ("retrieval", "coverage", "0.5", "sd"), 0.006, 3),
]

CAMO_PAPER = {  # AUROC by beta, 2 decimals
    "yelp": {"neighbour, relations": [0.84, 0.09, 0.04, 0.01, 0.00],
             "BWGNN, relations": [0.86, 0.80, 0.75, 0.68, 0.59],
             "BWGNN, combined": [0.86, 0.53, 0.49, 0.44, 0.38],
             "CARE-style, combined": [0.93, 0.36, 0.25, 0.16, 0.22]},
    "amazon": {"neighbour, relations": [0.93, 0.76, 0.61, 0.39, 0.20],
               "BWGNN, relations": [0.98, 0.97, 0.96, 0.93, 0.89],
               "BWGNN, combined": [0.98, 0.57, 0.53, 0.48, 0.43],
               "CARE-style, combined": [0.97, 0.55, 0.48, 0.45, 0.41]},
}
FORGERY_PAPER = {  # (features, condition): (AUPRC, AUROC), 2 decimals
    "yelp": {("own", "clean"): (0.82, 0.94), ("own", "attack"): (0.14, 0.50),
             ("neighbour", "clean"): (0.50, 0.84), ("neighbour", "attack"): (0.25, 0.73)},
    "amazon": {("own", "clean"): (0.92, 0.98), ("own", "attack"): (0.09, 0.50),
               ("neighbour", "clean"): (0.71, 0.93), ("neighbour", "attack"): (0.59, 0.89)},
}
DETAIL_PAPER = {  # (detector, metric, beta): (mean, SD over 5 runs), 3 decimals
    "yelp": {("BWGNN, combined", "auroc", "0.00"): (0.859, 0.005), ("BWGNN, combined", "auroc", "2.00"): (0.383, 0.028),
             ("BWGNN, combined", "auprc", "0.00"): (0.593, 0.017), ("BWGNN, combined", "auprc", "2.00"): (0.111, 0.005),
             ("CARE-style, combined", "auprc", "2.00"): (0.096, 0.018), ("CARE-style, combined", "auroc", "2.00"): (0.218, 0.162)},
    "amazon": {("BWGNN, combined", "auroc", "0.00"): (0.981, 0.003), ("BWGNN, combined", "auroc", "2.00"): (0.428, 0.027),
               ("BWGNN, combined", "auprc", "0.00"): (0.912, 0.005), ("BWGNN, combined", "auprc", "2.00"): (0.086, 0.007),
               ("CARE-style, combined", "auprc", "2.00"): (0.085, 0.016), ("CARE-style, combined", "auroc", "2.00"): (0.406, 0.079)},
}


def text_rows(abstracts, wiki):
    return ([("Abstracts", None, None, None, None)] + [(l, abstracts, k, v, d) for l, k, v, d in ABSTRACT_ROWS]
            + [("Wikipedia", None, None, None, None)] + [(l, wiki, k, v, d) for l, k, v, d in WIKI_ROWS])


def graph_rows(res):
    rows = []
    for ds, title in (("yelp", "YelpChi"), ("amazon", "Amazon")):
        d = res[ds]
        rows.append((f"{title} (fraud rate {d['fraud_rate']})", None, None, None, None))
        for (tier, cond), (prc, roc) in FORGERY_PAPER[ds].items():
            name = f"{tier} features, {'after content forgery' if cond == 'attack' else 'clean'}"
            rows += [(f"{name}: AUPRC", d, ("content_forgery", tier, cond, "auprc", 0), prc, 2),
                     (f"{name}: AUROC", d, ("content_forgery", tier, cond, "auroc", 0), roc, 2)]
        for det, paper in CAMO_PAPER[ds].items():
            rows += [(f"AUROC {det}, beta={float(b):g}", d, (det, "auroc", b, 0), v, 2)
                     for b, v in zip(BETA_KEYS, paper)]
        for (det, metric, b), (mean, sd) in DETAIL_PAPER[ds].items():
            name = f"{det} {metric.upper()}, beta={float(b):g}"
            rows += [(f"{name}: mean", d, (det, metric, b, 0), mean, 3),
                     (f"{name}: SD over 5 runs", d, (det, metric, b, 1), sd, 3)]
    try:
        below = {"n": sum(res[ds][det]["auprc"]["2.00"][0] < res[ds]["fraud_rate"]
                          for ds in ("yelp", "amazon") for det in ("BWGNN, combined", "CARE-style, combined"))}
    except KeyError:
        below = {}
    rows += [("Headline", None, None, None, None),
             ("Combined attack, beta=2: mean AUPRCs below the fraud rate (of 4)", below, ("n",), 4, 0)]
    return rows


def report(title, rows):
    """Print computed vs. reported values; return the number of mismatches."""
    print(f"\n=== {title} ===")
    bad = 0
    for label, root, path, paper, dec in rows:
        if root is None:
            print(f"-- {label}")
            continue
        try:
            value = root
            for k in path:
                value = value[k]
            value = float(value)
        except KeyError:
            print(f"   {label:<66} {'(not run)':>9}")
            continue
        ok = abs(value - paper) <= 0.5 * 10 ** -dec + 1e-9  # agrees at the paper's precision
        bad += not ok
        print(f"   {label:<66} {value:9.{dec}f}  paper {paper:.{dec}f}  {'ok' if ok else 'DIFFERS'}")
    return bad


# --------------------------------------------------------------------------- #
# Shipped results mapped onto the schema produced by text_study / graph_study #
# --------------------------------------------------------------------------- #
def shipped_text():
    T = RESULTS / "text"
    proto = load_json(T / "protocol_comparison.json")
    det = load_json(T / "detectors.json")

    def corpus(tag):
        audit = load_json(T / f"{tag}_audit.json")
        r = audit["retrieval"]
        p = proto[tag]
        return {"grouped_classifier": audit["grouped_embedding"],
                "item_level_classifier": dict(zip(("clean", "post"), p["original_ungrouped"]["average_oof_probability_auroc"])),
                "word_count": p["length_only_auroc"],
                "retrieval": {"full": r["full"], "symmetric_source_heldout": r["symmetric_source_group_heldout"],
                              "asymmetric_heldout": r["original_asymmetric_loo"],
                              "mean_max_similarity": {"reference": r["mean_similarity"]["reference_full"],
                                                      "rewritten": r["mean_similarity"]["rewritten_full"]},
                              "coverage": audit["coverage"]}}

    abstracts = corpus("abstracts")
    # Same-model numerical judge: point estimates recomputed from the per-item scores
    # on complete triples; source-cluster CIs need the corpus' source groups, so they
    # are read from the shipped paired analysis.
    j = {k: np.array(v, float) for k, v in load_json(T / "llmjudge_abstracts.json").items()}
    full = np.isfinite(j["human"]) & np.isfinite(j["ai"]) & np.isfinite(j["humanized"])
    h, a, z = j["human"][full], j["ai"][full], j["humanized"][full]
    paired = load_json(T / "deepseek_paired_audit.json")
    abstracts["numerical_judge"] = {"n_complete": int(full.sum()), "clean": auroc(h, a), "post": auroc(h, z),
                                    "drop": auroc(h, a) - auroc(h, z), "drop_ci": paired["drop_ci"]}
    for key, src in (("gpt2_perplexity", det["gpt2_perplexity"]), ("roberta_detector", det["roberta_chatgpt_detector"]),
                     ("binary_judge", load_json(T / "xmodel_judge_deepseek_binary.json")),
                     ("qwen_logit_judge", load_json(T / "xmodel_judge_qwen_logprob.json"))):
        abstracts[key] = {"clean": src["clean_auroc"], "post": src["postrewrite_auroc"]}
    return abstracts, corpus("wiki")


def shipped_graph():
    G = RESULTS / "graph"
    structure, camo = load_json(G / "structure.json"), load_json(G / "camouflage.json")
    relations, combined = load_json(G / "robust_camouflage_gnn_5seed.json"), load_json(G / "robust_combined_5seed.json")
    care = load_json(G / "robust_combined_caregnn_5seed.json")  # "CARE-GNN" key = CARE-style model
    return {ds: {"fraud_rate": structure[ds]["fraud_rate"],
                 "content_forgery": {"own": structure[ds]["intrinsic"], "neighbour": structure[ds]["structural"]},
                 "neighbour, relations": camo[ds]["naive"],
                 "BWGNN, relations": relations[ds]["BWGNN"],
                 "BWGNN, combined": combined[ds]["BWGNN"],
                 "CARE-style, combined": care[ds]["CARE-GNN"]}
            for ds in ("yelp", "amazon")}


# --------------------------------------------------------------------------- #
# Modes                                                                       #
# --------------------------------------------------------------------------- #
def cmd_check(_):
    bad = report("Study 1: text detectors and generation records (shipped results)", text_rows(*shipped_text()))
    bad += report("Study 2: graph fraud detectors under combined attacks (shipped results)", graph_rows(shipped_graph()))
    print(f"\n{'All headline values match the paper.' if not bad else f'{bad} value(s) differ from the paper.'}")
    return int(bad > 0)


def cmd_text(args):
    from text_study import run_corpus
    cached = load_json(RESULTS / "text" / "llmjudge_abstracts.json")
    out = {"abstracts": run_corpus(load_json(args.corpus / "abstracts_texts.json"), cached_judge=cached,
                                   detectors=args.detectors, judges=args.judges),
           "wiki": run_corpus(load_json(args.corpus / "wiki_texts.json"))}
    save_json(out, args.out / "text_study.json")
    report("Study 1 (rerun)", text_rows(out["abstracts"], out["wiki"]))
    print(f"\nwrote {args.out / 'text_study.json'}")


def cmd_graph(args):
    from graph_study import run
    out = run(args.data, datasets=args.datasets, seeds=args.seeds)
    save_json(out, args.out / "graph_study.json")
    if set(args.datasets) == {"yelp", "amazon"}:
        report("Study 2 (rerun)", graph_rows(out))
    print(f"\nwrote {args.out / 'graph_study.json'}")


def cmd_corpus(args):
    from corpus import build
    build(args.out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode")
    sub.add_parser("check", help="compare shipped results with the paper (default)")
    t = sub.add_parser("text", help="rerun the text study")
    t.add_argument("--corpus", type=Path, required=True, help="dir with abstracts_texts.json, wiki_texts.json")
    t.add_argument("--detectors", action="store_true", help="also run GPT-2 and RoBERTa detectors")
    t.add_argument("--judges", nargs="*", default=[], choices=["numerical", "binary", "qwen_logit"],
                   help="re-score abstracts with LLM judges (needs $JUDGE_EP/$LLM_KEY or mlx_lm)")
    t.add_argument("--out", type=Path, default=Path("outputs"))
    g = sub.add_parser("graph", help="rerun the graph study")
    g.add_argument("--data", type=Path, required=True, help="dir with YelpChi.mat, Amazon.mat")
    g.add_argument("--datasets", nargs="+", default=["yelp", "amazon"], choices=["yelp", "amazon"])
    g.add_argument("--seeds", type=int, default=5)
    g.add_argument("--out", type=Path, default=Path("outputs"))
    c = sub.add_parser("corpus", help="rebuild text corpora (needs $GEN_EP/$LLM_KEY)")
    c.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    modes = {None: cmd_check, "check": cmd_check, "text": cmd_text, "graph": cmd_graph, "corpus": cmd_corpus}
    return modes[args.mode](args) or 0


if __name__ == "__main__":
    sys.exit(main())
