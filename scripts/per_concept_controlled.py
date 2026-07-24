#!/usr/bin/env python3
"""Per-concept Unmentioned UNR on SilentEHR (new-background protocol)."""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from silentehr import (  # noqa: E402
    CONCEPTS,
    FULL_DATA,
    OUT,
    Embedder,
    binary_trap,
    load_rows,
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

CONCEPT_META = {c["id"]: c for c in CONCEPTS}


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
        if not cache_path.exists():
            raise FileNotFoundError(f"missing cache {cache_path}")
        Z = np.load(cache_path)
    return emb, {r["id"]: Z[i] for i, r in enumerate(rows)}


def per_concept_unr(artifacts, state="unmentioned"):
    meta = artifacts["meta"]
    pred = artifacts["pred"]
    by_concept = defaultdict(list)
    for i, r in enumerate(meta):
        if r["state"] == state:
            by_concept[r["concept_id"]].append(pred[i] == "Refuted")
    return {
        cid: float(np.mean(vals)) if vals else None
        for cid, vals in by_concept.items()
    }


def per_group_summary(concept_unr):
    groups = defaultdict(list)
    for cid, unr in concept_unr.items():
        if unr is None:
            continue
        groups[CONCEPT_META[cid]["group"]].append(unr)
    return {
        g: {
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "n": len(v),
            "frac_gt_0_5": float(np.mean(np.asarray(v) > 0.5)),
        }
        for g, v in sorted(groups.items())
    }


def model_summary(concept_unr):
    vals = [v for v in concept_unr.values() if v is not None]
    arr = np.asarray(vals)
    return {
        "n_concepts": len(arr),
        "n_unr_gt_0_5": int((arr > 0.5).sum()),
        "frac_unr_gt_0_5": float(np.mean(arr > 0.5)),
        "mean_unr": float(arr.mean()),
        "std_unr": float(arr.std()),
        "min_unr": float(arr.min()),
        "max_unr": float(arr.max()),
    }


def run_tier(model, data_path, device=None):
    rows = load_rows(data_path)
    emb, E = load_embeddings(model, rows, device=device)
    trap = binary_trap(rows, E, emb, protocol="new_background")
    artifacts = trap.pop("_artifacts")
    unm = per_concept_unr(artifacts, "unmentioned")
    na = per_concept_unr(artifacts, "not_assessed")
    return {
        "concept_unr_unmentioned": unm,
        "concept_unr_not_assessed": na,
        "by_group": per_group_summary(unm),
        "summary": model_summary(unm),
        "known_acc": trap["known_acc"],
    }


def main():
    all_results = {"clean": {}, "indirect": {}}
    tiers = {
        "clean": FULL_DATA,
        "indirect": ROOT / "data" / "controlled_indirect.jsonl",
    }
    for tier, path in tiers.items():
        if not path.exists():
            continue
        for label, key in MODELS:
            print(f"{tier} {key} ...", flush=True)
            device = None if key == "tfidf" else "cuda"
            all_results[tier][key] = run_tier(key, path, device=device)

    out = OUT / "per_concept_unr.json"
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"saved {out}")

    # print heatmap-friendly table
    concept_ids = [c["id"] for c in CONCEPTS]
    print("\n=== Unmentioned UNR (clean tier) ===")
    header = "concept," + ",".join(k for _, k in MODELS)
    print(header)
    for cid in concept_ids:
        row = [cid]
        for _, key in MODELS:
            v = all_results["clean"][key]["concept_unr_unmentioned"].get(cid)
            row.append("" if v is None else f"{v:.3f}")
        print(",".join(row))

    print("\n=== n concepts with UNR>0.5 ===")
    for label, key in MODELS:
        s = all_results["clean"][key]["summary"]
        print(f"{label}: {s['n_unr_gt_0_5']}/{s['n_concepts']}")


if __name__ == "__main__":
    main()
