#!/usr/bin/env python3
"""Lab-note temporal observability for MIMIC C+O- cases."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mimic_eval_utils import (  # noqa: E402
    encode_propositions,
    load_mimic_embeddings,
    mimic_split,
    notes_dict,
)
from silentehr import OUT, default_mimic_struct_path, load_rows, pair_features  # noqa: E402

MODEL = "qwen"
SEED = 0


def parse_ts(value):
    if value in (None, ""):
        return pd.NaT
    return pd.to_datetime(value)


def observability_flags(row):
    t_lab = parse_ts(row.get("struct_charttime"))
    t_dis = parse_ts(row.get("dischtime"))
    # Note chart/store times are not exported in mimic_struct; use discharge
    # as the conservative upper bound on note availability.
    t_note_proxy = t_dis
    flags = {
        "t_lab": None if pd.isna(t_lab) else str(t_lab),
        "t_discharge": None if pd.isna(t_dis) else str(t_dis),
        "lab_before_discharge": (
            bool(t_lab <= t_dis) if pd.notna(t_lab) and pd.notna(t_dis) else None
        ),
        "lab_before_note_proxy": (
            bool(t_lab <= t_note_proxy)
            if pd.notna(t_lab) and pd.notna(t_note_proxy) else None
        ),
        "conservative_observable": (
            bool(t_lab <= min(t_note_proxy, t_dis))
            if pd.notna(t_lab) and pd.notna(t_dis) else None
        ),
    }
    return flags


def fit_predict(train, test, Ez, P):
    Xtr = np.asarray([
        pair_features(Ez[r["hadm_id"]], P[r["proposition"]]) for r in train])
    Xte = np.asarray([
        pair_features(Ez[r["hadm_id"]], P[r["proposition"]]) for r in test])
    ytr = np.asarray([int(r["structured_positive"]) for r in train])
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0)
    clf.fit(scaler.transform(Xtr), ytr)
    return clf.predict_proba(scaler.transform(Xte))[:, list(clf.classes_).index(1)]


def r_plus_minus(subset, p_pos):
    mask = np.asarray([
        r["structured_positive"] and r["mention_status"] == "unmentioned"
        for r in subset
    ])
    if not mask.any():
        return {"n": 0, "R": None}
    return {
        "n": int(mask.sum()),
        "R": float(np.mean(p_pos[mask] >= 0.5)),
    }


def summarize_subset(name, rows, p_pos):
    flags = [observability_flags(r) for r in rows]
    n = len(rows)
    def rate(key):
        vals = [f[key] for f in flags if f[key] is not None]
        return float(np.mean(vals)) if vals else None
    return {
        "name": name,
        "n_C+O-": n,
        "lab_before_discharge_frac": rate("lab_before_discharge"),
        "lab_before_note_proxy_frac": rate("lab_before_note_proxy"),
        "conservative_observable_frac": rate("conservative_observable"),
        **r_plus_minus(rows, p_pos),
    }


def main():
    struct_path = default_mimic_struct_path()
    struct_rows = load_rows(struct_path)
    train, test, eligible = mimic_split(seed=SEED, cap_per_cell=500)
    cplus_unm = [
        r for r in test
        if r["structured_positive"] and r["mention_status"] == "unmentioned"
    ]

    notes = notes_dict(train, test)
    emb, Ez = load_mimic_embeddings(MODEL, notes, train, device="cuda")
    P = encode_propositions(emb, train + test)
    p_pos = fit_predict(train, test, Ez, P)

    def row_key(r):
        return (r["hadm_id"], r["concept"])

    test_idx = {row_key(r): i for i, r in enumerate(test)}
    cplus_p = np.asarray([p_pos[test_idx[row_key(r)]] for r in cplus_unm])

    conservative = [
        r for r, f in zip(cplus_unm, [observability_flags(r) for r in cplus_unm])
        if f["conservative_observable"]
    ]
    conservative_p = np.asarray([
        p_pos[test_idx[row_key(r)]] for r in conservative
    ]) if conservative else np.array([])

    result = {
        "limitations": (
            "Discharge-note charttime and storetime are not present in "
            "mimic_struct_deepseek.jsonl. We use dischtime as a conservative "
            "proxy for note availability (t_note). Labs are selected from the "
            "final window_h before discharge, so t_lab <= t_discharge holds "
            "by construction."
        ),
        "window_h": struct_rows[0].get("window_h") if struct_rows else None,
        "model_for_R": MODEL,
        "all_C+O-": summarize_subset("all", cplus_unm, cplus_p),
        "conservative_observable_C+O-": summarize_subset(
            "conservative", conservative, conservative_p),
        "by_concept": {},
    }
    for concept in eligible:
        sub = [r for r in cplus_unm if r["concept"] == concept]
        sub_p = np.asarray([p_pos[test_idx[row_key(r)]] for r in sub])
        result["by_concept"][concept] = summarize_subset(concept, sub, sub_p)

    out_path = OUT / "mimic_temporal_observability.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
