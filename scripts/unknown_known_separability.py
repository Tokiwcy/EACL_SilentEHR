#!/usr/bin/env python3
"""Unknown-vs-known linear separability probes on new-background splits."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mechanism_analysis_utils import (  # noqa: E402
    FULL_DATA,
    MODELS,
    background_probe_splits,
    encode_propositions,
    load_embeddings,
    load_rows,
    pair_matrix,
)
from silentehr import OUT, binary_trap  # noqa: E402

C_GRID = [0.01, 0.1, 1.0, 10.0]

PROBE_DEFS = {
    "U_vs_K": {"positive_states": {"unmentioned"}},
    "NA_vs_K": {"positive_states": {"not_assessed"}},
    "Unc_vs_K": {"positive_states": {"uncertain"}},
    "AllU_vs_K": {"positive_states": None},  # any y3 == Unknown
}


def probe_rows(subset, positive_states):
    pos, neg = [], []
    for r in subset:
        if r["y3"] in ("Supported", "Refuted"):
            neg.append(r)
        elif positive_states is None:
            if r["y3"] == "Unknown":
                pos.append(r)
        elif r["state"] in positive_states:
            pos.append(r)
    return pos, neg


def build_xy(pos, neg, E, P):
    rows = pos + neg
    if not rows:
        return None, None, 0, 0.0
    X, _ = pair_matrix(rows, E, P)
    y = np.asarray([1] * len(pos) + [0] * len(neg))
    prevalence = len(pos) / len(rows)
    return X, y, len(rows), prevalence


def eval_probe(clf, scaler, X, y):
    if X is None or len(np.unique(y)) < 2:
        return None
    Xs = scaler.transform(X)
    proba = clf.predict_proba(Xs)[:, list(clf.classes_).index(1)]
    pred = (proba >= 0.5).astype(int)
    return {
        "AUROC": float(roc_auc_score(y, proba)),
        "AUPRC": float(average_precision_score(y, proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def select_c(train_pos, train_neg, val_pos, val_neg, E, P):
    Xtr, ytr, _, _ = build_xy(train_pos, train_neg, E, P)
    if Xtr is None or len(np.unique(ytr)) < 2:
        return None, None, None
    best_c, best_scaler, best_clf, best_val = None, None, None, -1.0
    for C in C_GRID:
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(
            C=C, max_iter=3000, class_weight="balanced", random_state=0)
        clf.fit(scaler.transform(Xtr), ytr)
        Xv, yv, _, _ = build_xy(val_pos, val_neg, E, P)
        if Xv is None or len(np.unique(yv)) < 2:
            continue
        proba = clf.predict_proba(scaler.transform(Xv))[
            :, list(clf.classes_).index(1)]
        score = float(roc_auc_score(yv, proba))
        if score > best_val:
            best_val, best_c, best_scaler, best_clf = score, C, scaler, clf
    return best_c, best_scaler, best_clf


def analyze_model(model, rows, device=None):
    emb, E = load_embeddings(model, rows, device=device)
    P = encode_propositions(emb, rows)
    trap = binary_trap(rows, E, emb, protocol="new_background")
    unr = trap["UNR_by_state"]["unmentioned"]

    train_bg, val_bg, test_bg = background_probe_splits(rows)
    out = {"model": model, "UNR": unr, "probes": {}}

    for name, cfg in PROBE_DEFS.items():
        tr_pos, tr_neg = probe_rows(train_bg, cfg["positive_states"])
        va_pos, va_neg = probe_rows(val_bg, cfg["positive_states"])
        te_pos, te_neg = probe_rows(test_bg, cfg["positive_states"])

        c, scaler, clf = select_c(tr_pos, tr_neg, va_pos, va_neg, E, P)
        if clf is None:
            out["probes"][name] = {"error": "insufficient classes"}
            continue

        Xte, yte, n_test, prev = build_xy(te_pos, te_neg, E, P)
        metrics = eval_probe(clf, scaler, Xte, yte)
        out["probes"][name] = {
            "selected_C": c,
            "n_train_pos": len(tr_pos),
            "n_train_neg": len(tr_neg),
            "n_test": n_test,
            "test_prevalence": prev,
            **(metrics or {}),
        }
        # recompute val AUROC at selected C for logging
        Xv, yv, _, _ = build_xy(va_pos, va_neg, E, P)
        if Xv is not None and len(np.unique(yv)) == 2:
            proba = clf.predict_proba(scaler.transform(Xv))[
                :, list(clf.classes_).index(1)]
            out["probes"][name]["val_AUROC"] = float(roc_auc_score(yv, proba))

    return out


def main():
    rows = load_rows(FULL_DATA)
    all_out = {"protocol": "new_background", "split": "56/12/12 backgrounds",
               "C_grid": C_GRID, "models": {}}

    for label, model in MODELS:
        print(f"{label} ...", flush=True)
        device = None if model == "tfidf" else "cuda"
        res = analyze_model(model, rows, device=device)
        all_out["models"][model] = res
        u = res["probes"].get("U_vs_K", {})
        print(
            f"  UNR={res['UNR']:.3f} U/K AUROC={u.get('AUROC', float('nan')):.3f} "
            f"C={u.get('selected_C')}",
            flush=True,
        )

    out_path = OUT / "unknown_known_separability.json"
    out_path.write_text(json.dumps(all_out, indent=2), encoding="utf-8")
    print(f"\nsaved {out_path}")

    header = f"{'Encoder':<14} {'UNR':>6} {'U/K':>6} {'NA/K':>6} {'Unc/K':>6} {'All/K':>6}"
    print(f"\n{header}")
    for label, model in MODELS:
        m = all_out["models"][model]
        def au(name):
            p = m["probes"].get(name, {})
            return p.get("AUROC", float("nan"))
        print(
            f"{label:<14} {m['UNR']:6.3f} "
            f"{au('U_vs_K'):6.3f} {au('NA_vs_K'):6.3f} "
            f"{au('Unc_vs_K'):6.3f} {au('AllU_vs_K'):6.3f}"
        )


if __name__ == "__main__":
    main()
