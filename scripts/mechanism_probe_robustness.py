#!/usr/bin/env python3
"""Mechanism probes (U/K AUROC, Local-R@10) on artifact tiers."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mechanism_analysis_utils import (  # noqa: E402
    MODELS,
    background_probe_splits,
    encode_propositions,
    fit_sr_classifier,
    load_embeddings,
    load_rows,
    pair_matrix,
)
from silentehr import OUT, pair_features, protocol_split  # noqa: E402
from unknown_known_separability import PROBE_DEFS, build_xy, probe_rows, select_c  # noqa: E402

TIERS = [
    ("held_concept", "new_concept", ROOT / "data" / "controlled_full.jsonl"),
    ("indirect_evidence", "new_background", ROOT / "data" / "controlled_indirect.jsonl"),
    ("length_matched", "new_background", ROOT / "data" / "controlled_lengthmatched.jsonl"),
]

C_GRID = [0.01, 0.1, 1.0, 10.0]
K = 10


def l2_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / norms


def local_r_at_k(train_X, train_y_refuted, query_X, k):
    n_neighbors = min(k, len(train_X))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(train_X)
    _, idx = nn.kneighbors(query_X)
    return train_y_refuted[idx].mean(axis=1)


def uk_auroc_probe(model, rows, E, P, protocol):
    train_bg, val_bg, test_bg = background_probe_splits(rows)
    tr_pos, tr_neg = probe_rows(train_bg, {"unmentioned"})
    va_pos, va_neg = probe_rows(val_bg, {"unmentioned"})
    _, scaler, clf = select_c(tr_pos, tr_neg, va_pos, va_neg, E, P)
    if clf is None:
        return None
    Xte, yte, _, _ = build_xy(
        *probe_rows(test_bg, {"unmentioned"}), E, P)
    if Xte is None or len(np.unique(yte)) < 2:
        return None
    proba = clf.predict_proba(scaler.transform(Xte))[
        :, list(clf.classes_).index(1)]
    return float(roc_auc_score(yte, proba))


def local_r10(model, rows, E, P, protocol):
    protocol_train, test = protocol_split(rows, protocol)
    clf, scaler, known_train = fit_sr_classifier(protocol_train, E, P)
    train_X, train_y_sr = pair_matrix(known_train, E, P)
    train_y_ref = (train_y_sr == 0).astype(float)
    train_Xs = scaler.transform(train_X)
    unm_test = [r for r in test if r["state"] == "unmentioned"]
    if not unm_test:
        return None
    query_Xs = scaler.transform(np.asarray([
        pair_features(E[r["id"]], P[r["proposition"]]) for r in unm_test]))
    lr = local_r_at_k(train_Xs, train_y_ref, query_Xs, K)
    return float(np.mean(lr))


def analyze_tier(rows, protocol):
    tier_out = {"protocol": protocol, "models": {}}
    for label, model in MODELS:
        device = None if model == "tfidf" else "cuda"
        emb, E = load_embeddings(model, rows, device=device)
        P = encode_propositions(emb, rows)
        tier_out["models"][model] = {
            "U_vs_K_AUROC": uk_auroc_probe(model, rows, E, P, protocol),
            f"LocalR@{K}": local_r10(model, rows, E, P, protocol),
        }
    return tier_out


def main():
    all_out = {}
    for tier_name, protocol, path in TIERS:
        print(f"\n=== {tier_name} ({protocol}) ===", flush=True)
        rows = load_rows(path)
        tier_out = analyze_tier(rows, protocol)
        all_out[tier_name] = tier_out
        for label, model in MODELS:
            m = tier_out["models"][model]
            print(
                f"  {label:<14} U/K={m['U_vs_K_AUROC']} "
                f"LR@{K}={m[f'LocalR@{K}']}",
                flush=True,
            )

    out_path = OUT / "mechanism_probe_robustness.json"
    out_path.write_text(json.dumps(all_out, indent=2), encoding="utf-8")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
