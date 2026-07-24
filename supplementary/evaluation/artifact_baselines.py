"""Artifact baseline detectors for SilentEHR (self-contained)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from silentehr import FULL_DATA, OUT, Embedder, _train_lr, load_rows, protocol_split
from supplementary.evaluation.trap_runner import run_binary_trap
from supplementary.paths import OUTPUT_DIR


NEG_CUES = (
    "denies", "negative for", " absent", "no current", "without ",
    "not receiving", "not on ", "explicitly absent", "was not assessed",
)
POS_CUES = (
    "reports ", "endorses ", "currently has", "is documented", "active ",
    "currently receiving", "allergy list includes",
)


def concept_in_text(row):
    text = row["profile"].lower()
    name = row["concept"].lower()
    cid = row["concept_id"].replace("_", " ")
    return name in text or cid in text


def artifact_features(row, mode):
    if mode == "length":
        nw = len(row["profile"].split())
        pw = len(row["proposition"].split())
        return np.array([nw, pw, nw * pw, np.log1p(nw), np.log1p(pw)], dtype=float)
    if mode == "concept_match":
        return np.array([float(concept_in_text(row))], dtype=float)
    if mode == "lexical_cue":
        t = row["profile"].lower()
        return np.array([
            float(any(x in t for x in NEG_CUES)),
            float(any(x in t for x in POS_CUES)),
            float(concept_in_text(row)),
        ], dtype=float)
    if mode == "bow_coverage":
        text = row["profile"].lower()
        tokens = set(re.findall(r"[a-z]+", text))
        concept_tokens = set(re.findall(r"[a-z]+", row["concept_id"].replace("_", " ")))
        concept_tokens.add(row["concept"].lower())
        cov = len(tokens & concept_tokens) / len(concept_tokens) if concept_tokens else 0.0
        return np.array([cov], dtype=float)
    raise ValueError(mode)


def run_artifact_trap(rows, mode, protocol="new_background", tau=0.9):
    from sklearn.metrics import roc_auc_score

    train_all, test = protocol_split(rows, protocol)
    train = [r for r in train_all if r["y3"] in ("Supported", "Refuted")]
    Xtr = np.asarray([artifact_features(r, mode) for r in train])
    ytr = np.asarray([1 if r["y3"] == "Supported" else 0 for r in train])
    clf, sc = _train_lr(Xtr, ytr)

    Xte = np.asarray([artifact_features(r, mode) for r in test])
    proba = clf.predict_proba(sc.transform(Xte))
    classes = list(clf.classes_)
    i_ref, i_sup = classes.index(0), classes.index(1)
    p_ref, p_sup = proba[:, i_ref], proba[:, i_sup]
    pred = np.where(p_sup >= p_ref, "Supported", "Refuted")

    known_mask = np.array([r["y3"] in ("Supported", "Refuted") for r in test])
    unk_mask = np.array([r["y3"] == "Unknown" for r in test])
    unm_mask = np.array([r["state"] == "unmentioned" for r in test])

    known_acc = float(np.mean(
        pred[known_mask] == np.array([r["y3"] for r in test])[known_mask]
    )) if known_mask.any() else float("nan")
    unr = float(np.mean(pred[unk_mask] == "Refuted")) if unk_mask.any() else float("nan")
    unr_unm = float(np.mean(pred[unm_mask] == "Refuted")) if unm_mask.any() else float("nan")
    hcunr = float(np.mean(
        (pred[unm_mask] == "Refuted") & (p_ref[unm_mask] > tau)
    )) if unm_mask.any() else float("nan")

    maxp = proba.max(axis=1)
    known_or_unm = known_mask | unm_mask
    auroc = float("nan")
    if unm_mask.any() and len(np.unique(unm_mask[known_or_unm].astype(int))) == 2:
        auroc = float(roc_auc_score(
            unm_mask[known_or_unm].astype(int),
            (1 - maxp)[known_or_unm],
        ))

    return {
        "baseline": mode,
        "known_acc": known_acc,
        "UNR": unr,
        "UNR_unmentioned": unr_unm,
        f"HCUNR@{tau}": hcunr,
        "unmentioned_auroc": auroc,
        "n_train": len(train),
        "n_test": len(test),
    }


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
            (model + "\0" + "\0".join(texts)).encode("utf-8"), digest_size=12,
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


def run_random_encoder_trap(rows, E, P, protocol="new_background", tau=0.9, seed=0):
    rng = np.random.default_rng(seed)
    dim = next(iter(E.values())).shape[0]
    RandE = {k: rng.standard_normal(dim).astype(np.float32) for k in E}
    RandE = {k: v / (np.linalg.norm(v) + 1e-9) for k, v in RandE.items()}
    RandP = {k: rng.standard_normal(dim).astype(np.float32) for k in P}
    RandP = {k: v / (np.linalg.norm(v) + 1e-9) for k, v in RandP.items()}
    return run_binary_trap(rows, RandE, RandP, protocol=protocol, tau=tau)


def run_all(output: Path | None = None, device: str | None = None) -> dict:
    rows = load_rows(FULL_DATA)
    results = {"protocol": "new_background", "baselines": {}, "encoders": {}}

    for mode in ("length", "concept_match", "lexical_cue", "bow_coverage"):
        print(f"artifact {mode} ...", flush=True)
        results["baselines"][mode] = run_artifact_trap(rows, mode)

    print("loading encoder caches for random-encoder control ...", flush=True)
    E, P = load_embeddings("qwen", rows, device=device)
    results["baselines"]["random_encoder"] = run_random_encoder_trap(rows, E, P)

    for model in ("qwen", "jina", "biolord"):
        print(f"encoder reference {model} ...", flush=True)
        E, P = load_embeddings(model, rows, device=device)
        results["encoders"][model] = run_binary_trap(rows, E, P)

    out = output or OUTPUT_DIR / "artifact_baselines.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {out}")
    return results
