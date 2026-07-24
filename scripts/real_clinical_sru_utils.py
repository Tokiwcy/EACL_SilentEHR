#!/usr/bin/env python3
"""Shared real-note S/R/U evaluation paths."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from silentehr import _train_lr, pair_features
from sru_binary_core import (
    REFUTED,
    SUPPORTED,
    predict_sr,
    predict_sr_from_proba,
    row_prediction_record,
    validate_direction,
    y_from_label,
)
from stats_utils import wilson_summary

ROOT = Path(__file__).resolve().parents[1]
SPLIT_MANIFEST = ROOT / "outputs" / "real_clinical_sru_split_manifest.json"

OBSERVATION_TO_SRU = {
    "observed_positive": ("Supported", "supported"),
    "observed_negative": ("Refuted", "refuted"),
    "unmentioned": ("Unknown", "unmentioned"),
    "uncertain": ("Unknown", "uncertain"),
    "historical_only": ("Unknown", "not_assessed"),
    "admission_only": ("Unknown", "not_assessed"),
    "temporal_conflict": ("Unknown", "uncertain"),
}

LAB_CONCEPTS = {"anemia", "thrombocytopenia", "hyperkalemia"}

EXPANSION_CONCEPTS = [
    "chest_pain", "nausea", "dyspnea",
    "penicillin_allergy", "contrast_allergy",
    "anticoagulation", "insulin_therapy",
    "renal_dysfunction", "liver_dysfunction",
    "anemia", "thrombocytopenia",
]


def observation_to_sru(status: str) -> tuple[str, str]:
    if status not in OBSERVATION_TO_SRU:
        raise ValueError(f"unknown observation status: {status}")
    return OBSERVATION_TO_SRU[status]


def concept_group(concept_id: str) -> str:
    from silentehr import CONCEPTS
    if concept_id in LAB_CONCEPTS or concept_id == "hyperkalemia":
        return "laboratory"
    for meta in CONCEPTS:
        if meta["id"] == concept_id:
            return meta["group"]
    return "other"


def qwen_status_from_row(row: dict) -> str:
    return row.get("full_note_mention_status") or row.get("mention_status")


def map_qwen_to_sru(row: dict) -> tuple[str, str]:
    status = qwen_status_from_row(row)
    if status == "observed_pos":
        return "Supported", "supported"
    if status == "observed_neg":
        return "Refuted", "refuted"
    if status in ("unmentioned", None):
        return "Unknown", "unmentioned"
    return "Unknown", "uncertain"


def _train_sr_classifier(train_rows, feature_fn):
    train_sr = [r for r in train_rows if r["y3"] in (SUPPORTED, REFUTED)]
    Xtr = np.asarray([feature_fn(r) for r in train_sr])
    ytr = np.asarray([y_from_label(r["y3"]) for r in train_sr])
    clf, sc = _train_lr(Xtr, ytr)
    return clf, sc, train_sr, ytr


def predict_rows(subset, clf, sc, feature_fn):
    if not subset:
        return None
    X = np.asarray([feature_fn(r) for r in subset])
    bundle = predict_sr(clf, sc, X)
    bundle["rows"] = subset
    return bundle


def metrics_from_predictions(pred_bundle, *, tau=0.9, with_ci=False):
    from sklearn.metrics import roc_auc_score

    if not pred_bundle:
        return None
    subset = pred_bundle["rows"]
    pred = pred_bundle["pred"]
    p_ref = pred_bundle["p_ref"]
    proba = pred_bundle["proba"]

    out = {"n": len(subset), "classes": list(pred_bundle["classes"])}

    known_idx = [i for i, r in enumerate(subset) if r["y3"] in (SUPPORTED, REFUTED)]
    unk_idx = [i for i, r in enumerate(subset) if r["y3"] == "Unknown"]
    unm_idx = [i for i, r in enumerate(subset) if r.get("state") == "unmentioned"]

    if known_idx:
        known_correct = sum(1 for i in known_idx if pred[i] == subset[i]["y3"])
        out["known_acc"] = (
            wilson_summary(known_correct, len(known_idx))
            if with_ci else known_correct / len(known_idx)
        )

    if unk_idx:
        unr = sum(1 for i in unk_idx if pred[i] == REFUTED)
        out["UNR"] = wilson_summary(unr, len(unk_idx)) if with_ci else unr / len(unk_idx)
        usr = sum(1 for i in unk_idx if pred[i] == SUPPORTED)
        out["USR"] = wilson_summary(usr, len(unk_idx)) if with_ci else usr / len(unk_idx)

    if unm_idx:
        hc = sum(
            1 for i in unm_idx
            if pred[i] == REFUTED and p_ref[i] >= tau
        )
        out[f"HCUNR@{tau}"] = (
            wilson_summary(hc, len(unm_idx)) if with_ci else hc / len(unm_idx)
        )

    admission_idx = [
        i for i, r in enumerate(subset)
        if r["y3"] == "Unknown" and r.get("observation_status") == "admission_only"
    ]
    if admission_idx:
        k = sum(1 for i in admission_idx if pred[i] == REFUTED)
        key = "UNR_admission_only"
        out[key] = wilson_summary(k, len(admission_idx)) if with_ci else k / len(admission_idx)

    strict_idx = [
        i for i, r in enumerate(subset)
        if r["y3"] == "Unknown" and (
            r.get("state") == "unmentioned"
            or r.get("observation_status") == "unmentioned"
        )
    ]
    if strict_idx:
        k = sum(1 for i in strict_idx if pred[i] == REFUTED)
        key = "UNR_strict_unmentioned"
        out[key] = wilson_summary(k, len(strict_idx)) if with_ci else k / len(strict_idx)

    if known_idx and unk_idx:
        maxp = proba.max(axis=1)
        det_y = np.array([0] * len(known_idx) + [1] * len(unk_idx))
        det_s = np.concatenate([1 - maxp[known_idx], 1 - maxp[unk_idx]])
        if len(set(det_y)) == 2:
            out["unmentioned_auroc"] = float(roc_auc_score(det_y, det_s))

    out["counts"] = {
        "n_known": len(known_idx),
        "n_unknown": len(unk_idx),
        "n_admission_only": len(admission_idx),
        "n_strict_unmentioned": len(strict_idx),
    }
    return out


def subject_split(rows, seed=0, train_frac=0.7):
    subjects = sorted({r["subject_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(subjects)
    cut = int(train_frac * len(subjects))
    train_sub = set(subjects[:cut])
    train = [r for r in rows if r["subject_id"] in train_sub]
    test = [r for r in rows if r["subject_id"] not in train_sub]
    return train, test


def save_split_manifest(rows, *, seed=0, train_frac=0.7, dataset="human_consensus"):
    train, test = subject_split(rows, seed=seed, train_frac=train_frac)
    manifest = {
        "dataset": dataset,
        "seed": seed,
        "train_frac": train_frac,
        "train_subject_ids": sorted({r["subject_id"] for r in train}),
        "test_subject_ids": sorted({r["subject_id"] for r in test}),
        "n_train_rows": len(train),
        "n_test_rows": len(test),
    }
    path = ROOT / "outputs" / f"real_clinical_sru_split_manifest_{dataset}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return train, test, manifest


def load_split_manifest(rows, manifest_path=None, *, dataset="human_consensus"):
    path = Path(manifest_path or (
        ROOT / "outputs" / f"real_clinical_sru_split_manifest_{dataset}.json"
    ))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    train_sub = set(manifest["train_subject_ids"])
    test_sub = set(manifest["test_subject_ids"])
    train = [r for r in rows if r["subject_id"] in train_sub]
    test = [r for r in rows if r["subject_id"] not in train_sub]
    return train, test, manifest


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float)) or obj is None:
        return obj
    return obj


def evaluate_split(train_rows, test_rows, feature_fn, *, tau=0.9, with_ci=False):
    clf, sc, train_sr, ytr = _train_sr_classifier(train_rows, feature_fn)
    train_bundle = predict_rows(train_sr, clf, sc, feature_fn)
    test_bundle = predict_rows(test_rows, clf, sc, feature_fn)
    direction = validate_direction(
        np.asarray([y_from_label(r["y3"]) for r in train_sr]),
        train_bundle["p_ref"],
        context="train_sr",
    )
    test_records = [
        row_prediction_record(r, test_bundle, i)
        for i, r in enumerate(test_rows)
    ]
    return _json_safe({
        "classes": list(clf.classes_),
        "direction_check": direction,
        "train": metrics_from_predictions(train_bundle, tau=tau, with_ci=with_ci),
        "test": metrics_from_predictions(test_bundle, tau=tau, with_ci=with_ci),
        "test_predictions": test_records,
    })


def run_sru_trap(rows, E, P, emb, *, train_rows, test_rows, tau=0.9, feature_fn=None):
    feature_fn = feature_fn or (
        lambda r: pair_features(E[r["id"]], P[r["proposition"]])
    )
    return evaluate_split(train_rows, test_rows, feature_fn, tau=tau, with_ci=False)


def subject_kfold_splits(rows, n_folds=5, seed=0):
    from sklearn.model_selection import GroupKFold

    groups = [r["subject_id"] for r in rows]
    gkf = GroupKFold(n_splits=n_folds)
    for train_idx, test_idx in gkf.split(rows, groups=groups):
        yield [rows[i] for i in train_idx], [rows[i] for i in test_idx]
