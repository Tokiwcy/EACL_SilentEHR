"""Binary-trap evaluation helpers (no dependency on scripts/)."""
from __future__ import annotations

import random

import numpy as np

from silentehr import _train_lr, pair_features, protocol_split


def compose_pair_features(ex, eh, mode="full"):
    if mode == "ex":
        return ex
    if mode == "eh":
        return eh
    if mode == "concat":
        return np.concatenate([ex, eh], axis=-1)
    if mode == "full":
        return pair_features(ex, eh)
    raise ValueError(f"unknown pair mode: {mode}")


def shuffled_prop_map(test_rows, P, seed=0):
    row_props = [r["proposition"] for r in test_rows]
    permuted = row_props.copy()
    rng = random.Random(seed)
    rng.shuffle(permuted)
    out = dict(P)
    for r, new_prop in zip(test_rows, permuted):
        out[(r["id"], r["proposition"])] = P[new_prop]
    return out


def row_prop_embedding(r, P, prop_map=None):
    if prop_map is not None:
        return prop_map[(r["id"], r["proposition"])]
    return P[r["proposition"]]


def build_features_v2(rows, E, P, feature_mode, shuffled_map=None):
    X, y, meta = [], [], []
    for r in rows:
        ex = E[r["id"]]
        eh = row_prop_embedding(r, P, shuffled_map)
        if feature_mode == "patient":
            feat = ex
        elif feature_mode == "prop":
            feat = eh
        elif feature_mode == "shuffled":
            feat = pair_features(ex, eh)
        elif feature_mode in ("ex", "eh", "concat", "full"):
            feat = compose_pair_features(ex, eh, feature_mode)
        else:
            raise ValueError(feature_mode)
        X.append(feat)
        y.append(1 if r["y3"] == "Supported" else 0)
        meta.append(r)
    return np.asarray(X), np.asarray(y), meta


def uk_auroc(clf, sc, X, meta, tau=0.9):
    from sklearn.metrics import roc_auc_score

    Xs = sc.transform(X)
    proba = clf.predict_proba(Xs)
    maxp = proba.max(axis=1)
    known_mask = np.array([m["y3"] in ("Supported", "Refuted") for m in meta])
    unm_mask = np.array([m["state"] == "unmentioned" for m in meta])
    known_or_unm = known_mask | unm_mask
    if not unm_mask.any() or known_or_unm.sum() == 0:
        return float("nan")
    if len(np.unique(unm_mask[known_or_unm].astype(int))) < 2:
        return float("nan")
    return float(roc_auc_score(
        unm_mask[known_or_unm].astype(int),
        (1 - maxp)[known_or_unm],
    ))


def run_binary_trap(
    rows,
    E,
    P,
    *,
    feature_mode="full",
    protocol="new_background",
    tau=0.9,
    shuffle_seed=0,
):
    protocol_train, test = protocol_split(rows, protocol)
    train = [r for r in protocol_train if r["y3"] in ("Supported", "Refuted")]

    shuffled_map = None
    if feature_mode == "shuffled":
        shuffled_map = shuffled_prop_map(test, P, seed=shuffle_seed)
        Xtr, ytr, _ = build_features_v2(train, E, P, "full")
        Xte, yte, meta = build_features_v2(test, E, P, "shuffled", shuffled_map)
    else:
        Xtr, ytr, _ = build_features_v2(train, E, P, feature_mode)
        Xte, yte, meta = build_features_v2(test, E, P, feature_mode)

    clf, sc = _train_lr(Xtr, ytr)
    Xtes = sc.transform(Xte)
    proba = clf.predict_proba(Xtes)
    classes = list(clf.classes_)
    i_ref, i_sup = classes.index(0), classes.index(1)
    p_ref, p_sup = proba[:, i_ref], proba[:, i_sup]
    pred = np.where(p_sup >= p_ref, "Supported", "Refuted")

    known_mask = np.array([m["y3"] in ("Supported", "Refuted") for m in meta])
    unk_mask = ~known_mask
    known_acc = float(np.mean(
        pred[known_mask] == np.array([m["y3"] for m in meta])[known_mask]
    )) if known_mask.any() else float("nan")

    unm_mask = np.array([m["state"] == "unmentioned" for m in meta])
    unr_all = float(np.mean(pred[unk_mask] == "Refuted")) if unk_mask.any() else float("nan")
    unr_unm = float(np.mean(pred[unm_mask] == "Refuted")) if unm_mask.any() else float("nan")
    hcunr = float(np.mean(
        (pred[unm_mask] == "Refuted") & (p_ref[unm_mask] > tau)
    )) if unm_mask.any() else float("nan")

    return {
        "known_acc": known_acc,
        "UNR": unr_all,
        "UNR_unmentioned": unr_unm,
        f"HCUNR@{tau}": hcunr,
        "unmentioned_auroc": uk_auroc(clf, sc, Xte, meta, tau=tau),
        "n_train": len(train),
        "n_test": len(test),
    }
