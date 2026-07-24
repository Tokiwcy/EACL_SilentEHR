#!/usr/bin/env python3
"""Diagnose real S/R/U UNR direction: Supported vs Refuted on Unknown subtypes."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mimic_eval_utils import (
    clean_observed,
    encode_extra_notes,
    encode_propositions,
    load_mimic_embeddings,
    mimic_split,
    notes_dict,
)
from real_clinical_sru_utils import subject_split
from silentehr import _train_lr, pair_features


def load_consensus():
    rows = [json.loads(line) for line in (ROOT / "data/real_clinical_sru.jsonl").open(encoding="utf-8")]
    return [r for r in rows if r["label_tier"] == "human_consensus"]


def fit_predict(Ez, P, train_rows, test_rows):
    tr = [r for r in train_rows if r["y3"] in ("Supported", "Refuted")]
    Xtr = np.asarray([pair_features(Ez[r["hadm_id"]], P[r["proposition"]]) for r in tr])
    ytr = np.asarray([1 if r["y3"] == "Supported" else 0 for r in tr])
    clf, sc = _train_lr(Xtr, ytr)
    out = []
    for r in test_rows:
        x = pair_features(Ez[r["hadm_id"]], P[r["proposition"]])
        proba = clf.predict_proba(sc.transform(x.reshape(1, -1)))[0]
        i_ref = list(clf.classes_).index(0)
        i_sup = list(clf.classes_).index(1)
        pred = "Supported" if proba[i_sup] >= proba[i_ref] else "Refuted"
        out.append({"row": r, "pred": pred, "p_sup": float(proba[i_sup]), "p_ref": float(proba[i_ref])})
    return out


def summarize(name, preds):
    print(f"\n=== {name} ===")
    unk = [p for p in preds if p["row"]["y3"] == "Unknown"]
    if unk:
        unr = sum(p["pred"] == "Refuted" for p in unk) / len(unk)
        print(f"  ALL Unknown n={len(unk)} UNR={unr:.2f}  (pred Supported={1-unr:.2f})")
    for obs in ("unmentioned", "admission_only", "historical_only", "uncertain"):
        sub = [p for p in preds if p["row"]["y3"] == "Unknown" and p["row"]["observation_status"] == obs]
        if not sub:
            continue
        unr = sum(p["pred"] == "Refuted" for p in sub) / len(sub)
        print(f"  {obs:16s} n={len(sub):2d}  pred_R={unr:.2f}  pred_S={1-unr:.2f}")
    known = [p for p in preds if p["row"]["y3"] in ("Supported", "Refuted")]
    if known:
        acc = sum(p["row"]["y3"] == p["pred"] for p in known) / len(known)
        print(f"  Known acc={acc:.2f}  (n={len(known)})")


def main():
    cons = load_consensus()
    train_mimic, test_mimic, _ = mimic_split()
    notes = notes_dict(train_mimic, test_mimic)
    for r in cons:
        notes.setdefault(r["hadm_id"], r["profile"])

    model = "qwen"
    emb, Ez = load_mimic_embeddings(
        model, notes_dict(train_mimic, test_mimic), train_mimic, device="cuda")
    Ez = encode_extra_notes(emb, Ez, notes, {r["hadm_id"] for r in cons}, device="cuda")
    P = encode_propositions(emb, train_mimic + test_mimic + cons)

    train_audit, test_audit = subject_split(cons, seed=0)
    train_mimic_sr = [
        r for r in train_mimic
        if clean_observed(r) and r["concept"] in ("anemia", "thrombocytopenia")
    ]

    print("Test Unknown observation breakdown:", dict(Counter(
        r["observation_status"] for r in test_audit if r["y3"] == "Unknown")))

    summarize("A: train human audit S/R (current protocol)", fit_predict(Ez, P, train_audit, test_audit))
    summarize("B: train MIMIC struct S/R, test human audit", fit_predict(Ez, P, train_mimic_sr, test_audit))

    # Protocol C: test only human unmentioned (exclude admission_only from Unknown eval)
    test_unm = [r for r in test_audit if r["observation_status"] == "unmentioned" or r["y3"] in ("Supported", "Refuted")]
    summarize("A + eval only unmentioned Unknown (n=8) + Known", fit_predict(Ez, P, train_audit, test_unm))


if __name__ == "__main__":
    main()
