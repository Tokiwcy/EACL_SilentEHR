#!/usr/bin/env python3
"""Local-neighborhood Refuted composition (Local-R@k) on Unmentioned test cases."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mechanism_analysis_utils import (  # noqa: E402
    FULL_DATA,
    MODELS,
    encode_propositions,
    fit_sr_classifier,
    load_embeddings,
    load_rows,
    pair_matrix,
)
from silentehr import OUT, binary_trap, pair_features, protocol_split  # noqa: E402

K_VALUES = [5, 10, 20]
PRIMARY_K = 10


def l2_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / norms


def local_r_at_k(train_X, train_y_refuted, query_X, k, metric):
    """train_y_refuted: 1 if Refuted, 0 if Supported."""
    n_neighbors = min(k, len(train_X))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nn.fit(train_X)
    _, idx = nn.kneighbors(query_X)
    neighbor_ref = train_y_refuted[idx]
    return neighbor_ref.mean(axis=1)


def analyze_model(model, rows, device=None):
    emb, E = load_embeddings(model, rows, device=device)
    P = encode_propositions(emb, rows)
    trap = binary_trap(rows, E, emb, protocol="new_background")
    artifacts = trap["_artifacts"]

    protocol_train, test = protocol_split(rows, "new_background")
    clf, scaler, known_train = fit_sr_classifier(protocol_train, E, P)

    train_X, train_y_sr = pair_matrix(known_train, E, P)
    train_y_refuted = (train_y_sr == 0).astype(float)
    train_Xs = scaler.transform(train_X)

    unm_test = [r for r in test if r["state"] == "unmentioned"]
    if not unm_test:
        raise RuntimeError("no unmentioned test rows")
    query_X = np.asarray([
        pair_features(E[r["id"]], P[r["proposition"]]) for r in unm_test])
    query_Xs = scaler.transform(query_X)

    # match artifact indices for p_ref
    meta = artifacts["meta"]
    unm_idx = [i for i, m in enumerate(meta) if m["state"] == "unmentioned"]
    p_ref = artifacts["p_ref"][unm_idx]

    unr = trap["UNR_by_state"]["unmentioned"]
    out = {
        "model": model,
        "n_train_known": len(known_train),
        "n_unmentioned_test": len(unm_test),
        "UNR": unr,
        "distances": {},
    }

    for dist_name, metric, transform in (
        ("standardized_euclidean", "euclidean", lambda X: X),
        ("cosine_l2_normalized", "cosine", l2_normalize),
    ):
        tr = transform(train_Xs)
        q = transform(query_Xs)
        dist_out = {"metric": dist_name, "k": {}}
        for k in K_VALUES:
            lr = local_r_at_k(tr, train_y_refuted, q, k, metric)
            dist_out["k"][str(k)] = {
                "LocalR": float(np.mean(lr)),
                "refuted_majority_frac": float(np.mean(lr > 0.5)),
                "spearman_vs_p_ref": float(spearmanr(lr, p_ref).statistic),
            }
        out["distances"][dist_name] = dist_out

    return out


def main():
    rows = load_rows(FULL_DATA)
    all_out = {"protocol": "new_background", "k_values": K_VALUES, "models": {}}

    for label, model in MODELS:
        print(f"{label} ...", flush=True)
        device = None if model == "tfidf" else "cuda"
        res = analyze_model(model, rows, device=device)
        all_out["models"][model] = res
        primary = res["distances"]["standardized_euclidean"]["k"][str(PRIMARY_K)]
        print(
            f"  UNR={res['UNR']:.3f} LocalR@{PRIMARY_K}={primary['LocalR']:.3f} "
            f"Spearman={primary['spearman_vs_p_ref']:.3f}",
            flush=True,
        )

    out_path = OUT / "local_neighborhood_composition.json"
    out_path.write_text(json.dumps(all_out, indent=2), encoding="utf-8")
    print(f"\nsaved {out_path}")

    print(f"\n{'Encoder':<14} {'UNR':>6} {'LR@5':>6} {'LR@10':>6} {'LR@20':>6} "
          f"{'Maj@10':>7} {'Sp@10':>7}")
    for label, model in MODELS:
        d = all_out["models"][model]["distances"]["standardized_euclidean"]["k"]
        unr = all_out["models"][model]["UNR"]
        print(
            f"{label:<14} {unr:6.3f} "
            f"{d['5']['LocalR']:6.3f} {d['10']['LocalR']:6.3f} {d['20']['LocalR']:6.3f} "
            f"{d['10']['refuted_majority_frac']:7.3f} "
            f"{d['10']['spearman_vs_p_ref']:7.3f}"
        )


if __name__ == "__main__":
    main()
