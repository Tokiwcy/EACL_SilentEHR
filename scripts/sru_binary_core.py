#!/usr/bin/env python3
"""Shared binary Supported/Refuted prediction helpers."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from silentehr import (  # noqa: E402
    REFUTED,
    SUPPORTED,
    Y_REFUTED,
    Y_SUPPORTED,
    _predict_sr_from_proba,
    _sr_class_indices,
    _y_from_sr_label,
)

predict_sr_from_proba = _predict_sr_from_proba
class_indices = _sr_class_indices
y_from_label = _y_from_sr_label


def predict_sr(clf, sc, X: np.ndarray) -> dict:
    Xs = sc.transform(X)
    proba = clf.predict_proba(Xs)
    out = predict_sr_from_proba(proba, clf)
    out["pred_sklearn"] = clf.predict(Xs)
    return out


def validate_direction(y_labels: np.ndarray, p_ref: np.ndarray, *, context: str = "") -> dict:
    """Sanity-check that Refuted rows carry higher p_ref than Supported rows."""
    y_labels = np.asarray(y_labels)
    p_ref = np.asarray(p_ref)
    ref_mask = y_labels == Y_REFUTED
    sup_mask = y_labels == Y_SUPPORTED
    out = {
        "context": context,
        "classes_expected": [Y_REFUTED, Y_SUPPORTED],
        "mean_p_ref_on_refuted": float(p_ref[ref_mask].mean()) if ref_mask.any() else None,
        "mean_p_ref_on_supported": float(p_ref[sup_mask].mean()) if sup_mask.any() else None,
    }
    if ref_mask.any() and sup_mask.any():
        out["direction_ok"] = out["mean_p_ref_on_refuted"] > out["mean_p_ref_on_supported"]
    else:
        out["direction_ok"] = None
    return out


def row_prediction_record(row: dict, bundle: dict, index: int) -> dict:
    return {
        "row_id": row.get("id") or row.get("audit_id"),
        "subject_id": row["subject_id"],
        "hadm_id": row.get("hadm_id"),
        "concept": row.get("concept"),
        "y3": row["y3"],
        "state": row.get("state"),
        "observation_status": row.get("observation_status"),
        "pred": str(bundle["pred"][index]),
        "p_supported": float(bundle["p_sup"][index]),
        "p_refuted": float(bundle["p_ref"][index]),
    }
