#!/usr/bin/env python3
"""Shared MIMIC split + embedding-cache helpers for audit/control scripts."""
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from silentehr import (  # noqa: E402
    OUT,
    MIMIC_CHUNK_CHARS,
    default_mimic_struct_path,
    load_rows,
    mimic_embedding_batch,
    Embedder,
)


def clean_observed(r):
    return (
        r["mention_status"] == "observed_pos" and r["structured_positive"]
    ) or (
        r["mention_status"] == "observed_neg"
        and not r["structured_positive"]
    )


def mimic_split(seed=0, cap_per_cell=500):
    struct_path = default_mimic_struct_path()
    rows = [r for r in load_rows(struct_path)
            if r.get("text") and "subject_id" in r
            and r.get("mention_status") not in (
                None, "observed_mixed", "observed_uncertain",
                "observed_historical")]
    subjects = sorted({r["subject_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(subjects)
    cut = int(0.7 * len(subjects))
    train_subjects, test_subjects = set(subjects[:cut]), set(subjects[cut:])
    train = [r for r in rows
             if r["subject_id"] in train_subjects and clean_observed(r)]
    test = [r for r in rows if r["subject_id"] in test_subjects
            and (clean_observed(r) or r["mention_status"] == "unmentioned")]
    eligible = []
    for concept in sorted({r["concept"] for r in rows}):
        tr = [r for r in train if r["concept"] == concept]
        n_pos = sum(r["structured_positive"] for r in tr)
        if min(n_pos, len(tr) - n_pos) >= 20:
            eligible.append(concept)
    train = [r for r in train if r["concept"] in eligible]
    test = [r for r in test if r["concept"] in eligible]

    if cap_per_cell > 0:
        def cap(rows_in, is_train):
            cells = defaultdict(list)
            for r in rows_in:
                observation = "observed" if clean_observed(r) else "unmentioned"
                cells[(r["concept"], observation, r["structured_positive"])].append(r)
            local_rng = random.Random(seed + (0 if is_train else 1))
            out_rows = []
            for values in cells.values():
                local_rng.shuffle(values)
                out_rows.extend(values[:cap_per_cell])
            return out_rows
        train = cap(train, True)
        test = cap(test, False)
    return train, test, eligible


def notes_dict(*row_lists):
    notes = {}
    for rows in row_lists:
        for r in rows:
            notes.setdefault(r["hadm_id"], r["text"])
    return notes


def load_mimic_embeddings(model, notes, train_rows, device="cuda"):
    """Load mimic_cxo-compatible cache; encode only missing hadm_ids."""
    emb = Embedder(model, device=device)
    if model == "tfidf":
        train_hadm = {r["hadm_id"] for r in train_rows}
        emb.fit([notes[h] for h in train_hadm if h in notes])
    ids = list(notes)
    mimic_texts = [notes[i] for i in ids]
    cache_dir = OUT / "embedding_cache"
    fingerprint = hashlib.blake2b(
        (f"mimic\0{model}\0chunk{MIMIC_CHUNK_CHARS}\0"
         + "\0".join(mimic_texts)).encode("utf-8"),
        digest_size=12,
    ).hexdigest()
    cache_path = cache_dir / f"mimic_{model.replace('/', '_')}_{fingerprint}.npy"
    Ez = {}
    if model != "tfidf" and cache_path.exists():
        Z = np.load(cache_path)
        if len(Z) == len(ids):
            print(f"loaded cache {cache_path.name}", flush=True)
            Ez = {hadm: Z[i] for i, hadm in enumerate(ids)}
            return emb, Ez
    # Partial hit: try any cache with same note count (legacy fingerprint drift)
    prefix = f"mimic_{model.replace('/', '_')}_"
    for legacy in sorted(cache_dir.glob(f"{prefix}*.npy"), key=lambda p: p.stat().st_mtime, reverse=True):
        Z = np.load(legacy)
        if len(Z) == len(ids):
            print(f"loaded legacy cache {legacy.name}", flush=True)
            Ez = {hadm: Z[i] for i, hadm in enumerate(ids)}
            return emb, Ez
    print(f"encoding {len(ids)} notes with {model} ...", flush=True)
    Z = emb.encode(
        mimic_texts,
        batch_size=mimic_embedding_batch(model, device),
        chunk_chars=MIMIC_CHUNK_CHARS,
    )
    if model != "tfidf":
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, Z)
        print(f"saved cache {cache_path.name}", flush=True)
    Ez = {hadm: Z[i] for i, hadm in enumerate(ids)}
    return emb, Ez


def encode_propositions(emb, rows, device="cuda"):
    props = {r["proposition"] for r in rows}
    return {p: emb.encode([p])[0] for p in props}


def encode_extra_notes(emb, Ez, notes, hadm_ids, device="cuda"):
    """Encode audit-only admissions without polluting mimic cache fingerprint."""
    missing = [h for h in hadm_ids if h not in Ez and h in notes]
    if not missing:
        return Ez
    print(f"encoding {len(missing)} extra audit notes ...", flush=True)
    Z = emb.encode(
        [notes[h] for h in missing],
        batch_size=mimic_embedding_batch(getattr(emb, "name", "qwen"), device),
        chunk_chars=MIMIC_CHUNK_CHARS,
    )
    for i, h in enumerate(missing):
        Ez[h] = Z[i]
    return Ez
