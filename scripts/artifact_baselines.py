#!/usr/bin/env python3
"""Simple artifact baselines on SilentEHR binary-trap protocol."""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from silentehr import (  # noqa: E402
    FULL_DATA,
    _train_lr,
    load_rows,
    pair_features,
    protocol_split,
)
from silentehr_trap_utils import run_binary_trap  # noqa: E402


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


def neg_cue(text):
    t = text.lower()
    return any(x in t for x in NEG_CUES)


def pos_cue(text):
    t = text.lower()
    return any(x in t for x in POS_CUES)


def bow_coverage(row):
    text = row["profile"].lower()
    tokens = set(re.findall(r"[a-z]+", text))
    concept_tokens = set(re.findall(r"[a-z]+", row["concept_id"].replace("_", " ")))
    concept_tokens.add(row["concept"].lower())
    if not concept_tokens:
        return 0.0
    return len(tokens & concept_tokens) / len(concept_tokens)


def length_features(row):
    nw = len(row["profile"].split())
    pw = len(row["proposition"].split())
    return np.array([nw, pw, nw * pw, np.log1p(nw), np.log1p(pw)], dtype=float)


def artifact_features(row, mode):
    if mode == "length":
        return length_features(row)
    if mode == "concept_match":
        return np.array([float(concept_in_text(row))], dtype=float)
    if mode == "lexical_cue":
        return np.array([
            float(neg_cue(row["profile"])),
            float(pos_cue(row["profile"])),
            float(concept_in_text(row)),
        ], dtype=float)
    if mode == "bow_coverage":
        return np.array([bow_coverage(row)], dtype=float)
    raise ValueError(mode)


def run_artifact_trap(rows, mode, protocol="new_background", tau=0.9, seed=0):
    from sklearn.metrics import roc_auc_score

    train_all, test = protocol_split(rows, protocol)
    train = [r for r in train_all if r["y3"] in ("Supported", "Refuted")]
    Xtr = np.asarray([artifact_features(r, mode) for r in train])
    ytr = np.asarray([1 if r["y3"] == "Supported" else 0 for r in train])
    clf, sc = _train_lr(Xtr, ytr)

    Xte = np.asarray([artifact_features(r, mode) for r in test])
    Xtes = sc.transform(Xte)
    proba = clf.predict_proba(Xtes)
    classes = list(clf.classes_)
    i_ref, i_sup = classes.index(0), classes.index(1)
    p_ref = proba[:, i_ref]
    p_sup = proba[:, i_sup]
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


def run_random_encoder_trap(rows, E, P, protocol="new_background", tau=0.9, seed=0):
    rng = np.random.default_rng(seed)
    dim = next(iter(E.values())).shape[0]
    RandE = {k: rng.standard_normal(dim).astype(np.float32) for k in E}
    RandE = {k: v / (np.linalg.norm(v) + 1e-9) for k, v in RandE.items()}
    RandP = {k: rng.standard_normal(dim).astype(np.float32) for k in P}
    RandP = {k: v / (np.linalg.norm(v) + 1e-9) for k, v in RandP.items()}
    return run_binary_trap(rows, RandE, RandP, protocol=protocol, tau=tau)


def load_embeddings(rows, model):
    from mechanism_analysis_utils import encode_propositions, load_embeddings as load_emb

    emb, E = load_emb(model, rows)
    P = encode_propositions(emb, rows)
    return E, P


def main():
    rows = load_rows(FULL_DATA)
    out = ROOT / "outputs" / "artifact_baselines.json"
    results = {"protocol": "new_background", "baselines": {}, "encoders": {}}

    artifact_modes = [
        "length", "concept_match", "lexical_cue", "bow_coverage",
    ]
    for mode in artifact_modes:
        print(f"artifact {mode} ...", flush=True)
        results["baselines"][mode] = run_artifact_trap(rows, mode)

    print("loading encoder caches for random-encoder control ...", flush=True)
    E, P = load_embeddings(rows, "qwen")
    results["baselines"]["random_encoder"] = run_random_encoder_trap(rows, E, P)

    for model in ("qwen", "jina", "biolord"):
        print(f"encoder reference {model} ...", flush=True)
        E, P = load_embeddings(rows, model)
        results["encoders"][model] = run_binary_trap(rows, E, P)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {out}")
    for name, metrics in results["baselines"].items():
        print(
            f"{name:16s} UNR={metrics['UNR']:.3f} "
            f"HCUNR={metrics.get('HCUNR@0.9', float('nan')):.3f} "
            f"known={metrics['known_acc']:.3f}",
        )


if __name__ == "__main__":
    main()
