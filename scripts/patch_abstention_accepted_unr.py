#!/usr/bin/env python3
"""Patch accepted-set Unmentioned UNR into existing results_*_full.json files."""
import json
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from silentehr import FULL_DATA, Embedder, binary_trap, load_rows  # noqa: E402

MODELS = [
    "tfidf", "clinicalbert", "qwen", "jina", "biolord", "medcpt", "med-gte",
]


def load_embeddings(model, rows, device="cuda"):
    emb = Embedder(model, device=device if model != "tfidf" else None)
    texts = [r["profile"] for r in rows]
    if model == "tfidf":
        from silentehr import protocol_split

        train_rows, _ = protocol_split(rows, "new_background")
        fit_texts = [r["profile"] for r in train_rows]
        fit_texts += sorted({r["proposition"] for r in train_rows})
        emb.fit(fit_texts)
        Z = emb.encode(texts)
    else:
        cache_dir = ROOT / "outputs" / "embedding_cache"
        fingerprint = hashlib.blake2b(
            (model + "\0" + "\0".join(texts)).encode("utf-8"), digest_size=12,
        ).hexdigest()
        cache_path = cache_dir / f"{model.replace('/', '_')}_{fingerprint}.npy"
        if cache_path.exists():
            Z = np.load(cache_path)
        else:
            Z = emb.encode(texts)
    return emb, {r["id"]: Z[i] for i, r in enumerate(rows)}


def main():
    rows = load_rows(FULL_DATA)
    for model in MODELS:
        path = ROOT / "outputs" / f"results_{model}_full.json"
        if not path.exists():
            print(f"skip {model}: missing {path.name}")
            continue
        device = None if model == "tfidf" else "cuda"
        print(f"patching {model} ...", flush=True)
        emb, E = load_embeddings(model, rows, device=device)
        trap = binary_trap(rows, E, emb, protocol="new_background")
        data = json.loads(path.read_text(encoding="utf-8"))
        nb = data["binary_trap_by_protocol"]["new_background"]
        for key in ("confidence_abstention",):
            src = trap.get(key) or {}
            dst = nb.setdefault(key, {})
            for field in (
                "accepted_unmentioned_UNR",
                "accepted_unmentioned_n",
                "known_test_coverage",
                "unknown_reject_rate",
                "unmentioned_reject_rate",
                "threshold",
                "threshold_source",
            ):
                if field in src:
                    dst[field] = src[field]
        maha = (trap.get("mahalanobis_ood") or {}).get("abstention") or {}
        if maha:
            dst_maha = (nb.setdefault("mahalanobis_ood", {})).setdefault(
                "abstention", {},
            )
            for field in (
                "accepted_unmentioned_UNR",
                "accepted_unmentioned_n",
                "known_test_coverage",
                "unknown_reject_rate",
                "unmentioned_reject_rate",
            ):
                if field in maha:
                    dst_maha[field] = maha[field]
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        conf = nb["confidence_abstention"]
        print(
            f"  conf accepted UnmRR={conf.get('accepted_unmentioned_UNR')} "
            f"n={conf.get('accepted_unmentioned_n')} "
            f"cov={conf.get('known_test_coverage')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
