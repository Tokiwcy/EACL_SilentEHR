#!/usr/bin/env python3
"""Extended MIMIC control metrics: cell recalls, balanced accuracy, confusion."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mimic_eval_utils import (  # noqa: E402
    load_mimic_embeddings,
    mimic_split,
    notes_dict,
    encode_propositions,
)
from silentehr import pair_features  # noqa: E402


def cell_metrics(subset, p_pos, tau=0.9):
    pred_pos = p_pos >= 0.5
    pred_neg = ~pred_pos
    cells = {
        "C+O+": (np.asarray([
            r["structured_positive"] and r["mention_status"] == "observed_pos"
            for r in subset]), True),
        "C-O+": (np.asarray([
            (not r["structured_positive"])
            and r["mention_status"] == "observed_neg"
            for r in subset]), False),
        "C+O-": (np.asarray([
            r["structured_positive"] and r["mention_status"] == "unmentioned"
            for r in subset]), True),
        "C-O-": (np.asarray([
            (not r["structured_positive"])
            and r["mention_status"] == "unmentioned"
            for r in subset]), False),
    }
    out = {}
    correct_rates = []
    for name, (mask, want_positive) in cells.items():
        if not mask.any():
            out[name] = {"n": 0}
            continue
        if want_positive:
            correct = float(np.mean(pred_pos[mask]))
        else:
            correct = float(np.mean(pred_neg[mask]))
        out[name] = {
            "n": int(mask.sum()),
            "correct_rate": correct,
        }
        correct_rates.append(correct)
    out["macro_cell_accuracy"] = float(np.mean(correct_rates))
    cplus_unm = cells["C+O-"][0]
    if cplus_unm.any():
        out["UPFNR"] = 1 - float(np.mean(pred_pos[cplus_unm]))
        out["HCWE@0.9"] = float(np.mean(p_pos[cplus_unm] < (1 - tau)))
    return out


def main():
    model = "qwen"
    seed = 0
    train, test, eligible = mimic_split(seed=seed, cap_per_cell=500)
    notes = notes_dict(train, test)
    emb, Ez = load_mimic_embeddings(model, notes, train, device="cuda")
    P = encode_propositions(emb, train + test)

    def fit_predict(train_rows, test_rows, proposition_only=False, prior=None):
        if prior is not None:
            return np.asarray(prior)
        if proposition_only:
            Xtr = np.asarray([P[r["proposition"]] for r in train_rows])
            Xte = np.asarray([P[r["proposition"]] for r in test_rows])
        else:
            Xtr = np.asarray([
                pair_features(Ez[r["hadm_id"]], P[r["proposition"]])
                for r in train_rows])
            Xte = np.asarray([
                pair_features(Ez[r["hadm_id"]], P[r["proposition"]])
                for r in test_rows])
        ytr = np.asarray([int(r["structured_positive"]) for r in train_rows])
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=seed)
        clf.fit(scaler.transform(Xtr), ytr)
        return clf.predict_proba(scaler.transform(Xte))[
            :, list(clf.classes_).index(1)]

    prevalence = {}
    for concept in eligible:
        tr = [r for r in train if r["concept"] == concept]
        prevalence[concept] = float(np.mean(
            [r["structured_positive"] for r in tr]))

    joint_p = fit_predict(train, test)
    prop_p = fit_predict(train, test, proposition_only=True)
    prior_p = np.asarray([prevalence[r["concept"]] for r in test])

    y_all = np.asarray([int(r["structured_positive"]) for r in test])
    unm_idx = [i for i, r in enumerate(test)
               if r["mention_status"] == "unmentioned"]
    y_unm = y_all[unm_idx]

    results = {}
    for name, p in (
        ("patient_proposition", joint_p),
        ("proposition_only", prop_p),
        ("concept_prior", prior_p),
    ):
        results[name] = {
            **cell_metrics(test, p),
            "clinical_state_auroc_all": float(roc_auc_score(y_all, p)),
            "clinical_state_auroc_unmentioned": float(roc_auc_score(
                y_unm, p[unm_idx])) if len(np.unique(y_unm)) == 2 else None,
        }

    out = ROOT / "outputs" / "mimic_control_extended_qwen.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
