#!/usr/bin/env python3
"""MIMIC cell recalls R^{+,-} and R^{-,-} for all encoders (cached embeddings)."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mimic_eval_utils import encode_propositions, load_mimic_embeddings, mimic_split, notes_dict  # noqa: E402
from silentehr import OUT, pair_features  # noqa: E402

MODELS = [
    ("Qwen3-Emb", "qwen"),
    ("Jina v5", "jina"),
    ("BioLORD", "biolord"),
    ("ClinicalBERT", "clinicalbert"),
    ("MedCPT", "medcpt"),
    ("med-gte", "med-gte"),
    ("TF-IDF", "tfidf"),
]


def cell_recalls(subset, p_pos):
    """R = P(correct | cell); positive prediction = C+."""
    pred_pos = p_pos >= 0.5
    pred_neg = ~pred_pos
    cells = {
        "C+O+": lambda r: r["structured_positive"] and r["mention_status"] == "observed_pos",
        "C-O+": lambda r: (not r["structured_positive"]) and r["mention_status"] == "observed_neg",
        "C+O-": lambda r: r["structured_positive"] and r["mention_status"] == "unmentioned",
        "C-O-": lambda r: (not r["structured_positive"]) and r["mention_status"] == "unmentioned",
    }
    out = {}
    for name, fn in cells.items():
        mask = np.asarray([fn(r) for r in subset])
        if not mask.any():
            out[name] = {"n": 0}
            continue
        want_pos = name.startswith("C+")
        correct = pred_pos[mask] if want_pos else pred_neg[mask]
        out[name] = {
            "n": int(mask.sum()),
            "R": float(np.mean(correct)),
        }
    return out


def fit_predict(train, test, Ez, P, seed=0):
    Xtr = np.asarray([
        pair_features(Ez[r["hadm_id"]], P[r["proposition"]]) for r in train])
    Xte = np.asarray([
        pair_features(Ez[r["hadm_id"]], P[r["proposition"]]) for r in test])
    ytr = np.asarray([int(r["structured_positive"]) for r in train])
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=seed)
    clf.fit(scaler.transform(Xtr), ytr)
    return clf.predict_proba(scaler.transform(Xte))[:, list(clf.classes_).index(1)]


def main():
    seed = 0
    train, test, eligible = mimic_split(seed=seed, cap_per_cell=500)
    notes = notes_dict(train, test)
    all_out = {
        "seed": seed,
        "cap_per_cell": 500,
        "eligible_concepts": eligible,
        "models": {},
    }

    for label, model in MODELS:
        print(f"{label} ...", flush=True)
        device = None if model == "tfidf" else "cuda"
        emb, Ez = load_mimic_embeddings(model, notes, train, device=device)
        P = encode_propositions(emb, train + test, device=device)
        p_pos = fit_predict(train, test, Ez, P, seed=seed)
        cells = cell_recalls(test, p_pos)
        r_pp = cells.get("C+O+", {}).get("R")
        r_pm = cells.get("C+O-", {}).get("R")
        r_mm = cells.get("C-O-", {}).get("R")
        all_out["models"][model] = cells
        print(
            f"  R++={r_pp:.3f} R+-={r_pm:.3f} R--={r_mm:.3f} "
            f"(n C-O-={cells['C-O-']['n']})",
            flush=True,
        )

    out_path = OUT / "mimic_cell_recalls.json"
    out_path.write_text(json.dumps(all_out, indent=2), encoding="utf-8")
    print(f"\nsaved {out_path}")

    print(f"\n{'Encoder':<14} {'R++':>6} {'R+-':>6} {'R--':>6} {'n(C-O-)':>8}")
    for label, model in MODELS:
        c = all_out["models"][model]
        print(
            f"{label:<14} "
            f"{c['C+O+']['R']:6.3f} {c['C+O-']['R']:6.3f} "
            f"{c['C-O-']['R']:6.3f} {c['C-O-']['n']:8d}"
        )


if __name__ == "__main__":
    main()
