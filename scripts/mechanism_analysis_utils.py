#!/usr/bin/env python3
"""Shared helpers for SilentEHR mechanism analyses (new-background protocol)."""
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from silentehr import (  # noqa: E402
    FULL_DATA,
    OUT,
    Embedder,
    load_rows,
    pair_features,
    protocol_split,
)

MODELS = [
    ("TF-IDF", "tfidf"),
    ("ClinicalBERT", "clinicalbert"),
    ("Qwen3-Emb", "qwen"),
    ("Jina v5", "jina"),
    ("BioLORD", "biolord"),
    ("MedCPT", "medcpt"),
    ("med-gte", "med-gte"),
]


def load_embeddings(model, rows, device=None):
    emb = Embedder(model, device=device)
    texts = [r["profile"] for r in rows]
    if model == "tfidf":
        train_rows, _ = protocol_split(rows, "new_background")
        fit_texts = [r["profile"] for r in train_rows]
        fit_texts += sorted({r["proposition"] for r in rows})
        emb.fit(fit_texts)
        Z = emb.encode(texts)
    else:
        cache_dir = OUT / "embedding_cache"
        fingerprint = hashlib.blake2b(
            (model + "\0" + "\0".join(texts)).encode("utf-8"), digest_size=12
        ).hexdigest()
        cache_path = cache_dir / f"{model.replace('/', '_')}_{fingerprint}.npy"
        if cache_path.exists():
            Z = np.load(cache_path)
        else:
            print(f"  encoding {len(texts)} profiles ...", flush=True)
            Z = emb.encode(texts)
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, Z)
    return emb, {r["id"]: Z[i] for i, r in enumerate(rows)}


def encode_propositions(emb, rows):
    props = {r["proposition"] for r in rows}
    return {p: emb.encode([p])[0] for p in props}


def pair_matrix(rows, E, P):
    X = np.asarray([
        pair_features(E[r["id"]], P[r["proposition"]]) for r in rows])
    y_sr = np.asarray([1 if r["y3"] == "Supported" else 0 for r in rows])
    return X, y_sr


def background_probe_splits(rows):
    """56/12/12 background splits for diagnostic probes (not binary_trap train/test)."""
    def eligible(r):
        return (
            r["concept_split"] == "train"
            and r["template_split"] in ("train", "na")
        )

    train = [r for r in rows if eligible(r) and r["background_split"] == "train"]
    val = [r for r in rows if eligible(r) and r["background_split"] == "val"]
    test = [r for r in rows if eligible(r) and r["background_split"] == "test"]
    return train, val, test


def fit_sr_classifier(train_rows, E, P):
    """Standardized class-balanced LR on Supported/Refuted (matches binary_trap)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    known = [r for r in train_rows if r["y3"] in ("Supported", "Refuted")]
    X, y = pair_matrix(known, E, P)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=0)
    clf.fit(scaler.transform(X), y)
    return clf, scaler, known


def sr_refuted_proba(rows, clf, scaler, E, P):
    if not rows:
        return np.array([])
    X = np.asarray([
        pair_features(E[r["id"]], P[r["proposition"]]) for r in rows])
    proba = clf.predict_proba(scaler.transform(X))
    i_ref = list(clf.classes_).index(0)
    return proba[:, i_ref]
