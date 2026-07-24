#!/usr/bin/env python3
"""Re-evaluate new-background / new-concept protocols under alternate splits."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from silentehr import (  # noqa: E402
    FULL_DATA,
    OUT,
    Embedder,
    _split_assign,
    binary_trap,
    load_rows,
    protocol_split,
)

MODELS = [
    "tfidf", "qwen", "jina", "biolord", "medcpt", "clinicalbert", "med-gte",
]
BG_SEEDS = list(range(10))
CONCEPT_FOLDS = 4


def load_embeddings(model, rows, device=None):
    emb = Embedder(model, device=device)
    texts = [r["profile"] for r in rows]
    cache_dir = OUT / "embedding_cache"
    fingerprint = hashlib.blake2b(
        (model + "\0" + "\0".join(texts)).encode("utf-8"), digest_size=12
    ).hexdigest()
    cache_path = cache_dir / f"{model.replace('/', '_')}_{fingerprint}.npy"
    if cache_path.exists():
        Z = np.load(cache_path)
        print(f"  loaded cache {cache_path.name}", flush=True)
    else:
        print(f"  encoding {len(texts)} profiles ...", flush=True)
        Z = emb.encode(texts)
    return emb, {r["id"]: Z[i] for i, r in enumerate(rows)}


def assign_background_splits(rows, seed):
    bg_ids = sorted({r["background_id"] for r in rows})
    tags = _split_assign(len(bg_ids), seed=seed)
    tag_map = dict(zip(bg_ids, tags))
    out = copy.deepcopy(rows)
    for r in out:
        r["background_split"] = tag_map[r["background_id"]]
    return out


def assign_concept_folds(rows, fold):
    concept_ids = sorted({r["concept_id"] for r in rows})
    groups = [concept_ids[i::CONCEPT_FOLDS] for i in range(CONCEPT_FOLDS)]
    test_ids = set(groups[fold])
    out = copy.deepcopy(rows)
    for r in out:
        r["concept_split"] = "test" if r["concept_id"] in test_ids else "train"
    return out


def refit_tfidf(tagged, protocol):
    train_rows, _ = protocol_split(tagged, protocol)
    fit_texts = [r["profile"] for r in train_rows]
    fit_texts += sorted({r["proposition"] for r in tagged})
    emb = Embedder("tfidf")
    emb.fit(fit_texts)
    Z = emb.encode([r["profile"] for r in tagged])
    E = {r["id"]: Z[i] for i, r in enumerate(tagged)}
    return emb, E


def summarize_runs(runs):
    def stats(key, state=None):
        if state:
            vals = [r[key][state] for r in runs if r[key].get(state) is not None]
        else:
            vals = [r[key] for r in runs]
        arr = np.asarray(vals, dtype=float)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "frac_gt_0_5": float(np.mean(arr > 0.5)),
            "n": len(arr),
        }

    return {
        "known_acc": stats("known_acc"),
        "UNR_unmentioned": stats("UNR_by_state", "unmentioned"),
        "UNR_not_assessed": stats("UNR_by_state", "not_assessed"),
        "UNR_uncertain": stats("UNR_by_state", "uncertain"),
    }


def run_trap(model, tagged, protocol, E=None, emb=None, device=None):
    if model == "tfidf":
        emb, E = refit_tfidf(tagged, protocol)
    return binary_trap(tagged, E, emb, protocol=protocol)


def main():
    base_rows = load_rows(FULL_DATA)
    all_out = {}

    for model in MODELS:
        print(f"\n=== {model} ===", flush=True)
        device = None if model == "tfidf" else "cuda"
        E = emb = None
        if model != "tfidf":
            emb, E = load_embeddings(model, base_rows, device=device)

        bg_runs = []
        for seed in BG_SEEDS:
            tagged = assign_background_splits(base_rows, seed)
            res = run_trap(model, tagged, "new_background", E=E, emb=emb, device=device)
            bg_runs.append(res)
            print(
                f"  bg seed {seed}: known={res['known_acc']:.3f} "
                f"U={res['UNR_by_state']['unmentioned']:.3f} "
                f"NA={res['UNR_by_state']['not_assessed']:.3f} "
                f"Unc={res['UNR_by_state']['uncertain']:.3f}",
                flush=True,
            )
        model_out = {"background_splits": summarize_runs(bg_runs)}

        cf_runs = []
        for fold in range(CONCEPT_FOLDS):
            tagged = assign_concept_folds(base_rows, fold)
            res = run_trap(model, tagged, "new_concept", E=E, emb=emb, device=device)
            cf_runs.append(res)
            print(
                f"  concept fold {fold}: known={res['known_acc']:.3f} "
                f"U={res['UNR_by_state']['unmentioned']:.3f}",
                flush=True,
            )
        model_out["concept_folds"] = summarize_runs(cf_runs)
        all_out[model] = model_out

    out_path = OUT / "resplit_protocol_metrics.json"
    out_path.write_text(json.dumps(all_out, indent=2), encoding="utf-8")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
