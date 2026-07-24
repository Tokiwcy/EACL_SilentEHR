#!/usr/bin/env python3
"""Label-withholding on clinical NLI (MedNLI if local) or MIMIC-struct bridge."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mimic_eval_utils import (  # noqa: E402
    clean_observed,
    encode_propositions,
    load_mimic_embeddings,
    mimic_split,
    notes_dict,
)
from real_clinical_sru_utils import run_sru_trap  # noqa: E402
from silentehr import Embedder, load_rows, pair_features  # noqa: E402


MEDNLI_DIR = ROOT / "data" / "mednli"
MEDNLI_FILES = {
    "train": "mli_train_v1.jsonl",
    "validation": "mli_dev_v1.jsonl",
    "test": "mli_test_v1.jsonl",
}

LABEL_TEXT = {
    "entailment": ("Supported", "supported"),
    "contradiction": ("Refuted", "refuted"),
    "neutral": ("Unknown", "unmentioned"),
}


def load_mednli_local():
    if not MEDNLI_DIR.exists():
        return None
    rows = []
    for split, fname in MEDNLI_FILES.items():
        path = MEDNLI_DIR / fname
        if not path.exists():
            return None
        for i, line in enumerate(path.open(encoding="utf-8")):
            ex = json.loads(line)
            label = ex["gold_label"]
            if isinstance(label, int):
                y3, state = {
                    0: ("Supported", "supported"),
                    1: ("Unknown", "unmentioned"),
                    2: ("Refuted", "refuted"),
                }[label]
            else:
                y3, state = LABEL_TEXT[label]
            rows.append({
                "id": f"mednli:{split}:{i}",
                "split": split,
                "profile": ex["sentence1"],
                "proposition": ex["sentence2"],
                "y3": y3,
                "state": state,
            })
    return rows


def mednli_protocol(rows, max_train=8000, max_test=2000, seed=0):
    import random

    rng = random.Random(seed)
    train = [r for r in rows if r["split"] == "train" and r["y3"] in ("Supported", "Refuted")]
    test = [r for r in rows if r["split"] in ("validation", "test")]
    rng.shuffle(train)
    rng.shuffle(test)
    return train[:max_train], test[:max_test]


def mimic_bridge_rows():
    rows = load_rows(ROOT / "data/mimic_struct.jsonl")
    out = []
    for row in rows:
        if not row.get("text") or not row.get("proposition"):
            continue
        ms = row.get("full_note_mention_status") or row.get("mention_status")
        if ms == "observed_pos" and row.get("structured_positive"):
            y3, state = "Supported", "supported"
        elif ms == "observed_neg" and not row.get("structured_positive"):
            y3, state = "Refuted", "refuted"
        elif ms == "unmentioned":
            y3, state = "Unknown", "unmentioned"
        else:
            continue
        out.append({
            **row,
            "profile": row["text"],
            "y3": y3,
            "state": state,
            "concept_id": row["concept"],
        })
    return out


def encode_profiles(emb, rows, batch_size=32):
    texts = [r["profile"] for r in rows]
    ids = [r["id"] if "id" in r else f"{r['hadm_id']}:{r['concept']}" for r in rows]
    Z = emb.encode(texts, batch_size=batch_size)
    return {i: z for i, z in zip(ids, Z)}


def row_id(r):
    return r.get("id") or f"{r['hadm_id']}:{r['concept']}"


def eval_mednli(rows, model, device="cuda"):
    train, test = mednli_protocol(rows)
    emb = Embedder(model, device=device)
    for r in train + test:
        r["id"] = row_id(r)
    E = encode_profiles(emb, train + test)
    props = {r["proposition"] for r in train + test}
    P = {p: emb.encode([p])[0] for p in props}
    feature_fn = lambda r: pair_features(E[r["id"]], P[r["proposition"]])
    metrics = run_sru_trap(rows, E, P, emb, train_rows=train, test_rows=test, feature_fn=feature_fn)
    return {
        "corpus": "mednli",
        "model": model,
        "n_train_sr": len(train),
        "n_test": len(test),
        "n_test_neutral": sum(r["y3"] == "Unknown" for r in test),
        "metrics": metrics,
    }


def eval_mimic_bridge(model, device="cuda"):
    train_mimic, test_mimic, _ = mimic_split()
    rows = mimic_bridge_rows()
    notes = notes_dict(train_mimic, test_mimic, rows)
    emb, Ez = load_mimic_embeddings(model, notes_dict(train_mimic, test_mimic), train_mimic, device=device)
    P = encode_propositions(emb, train_mimic + test_mimic + rows)

    train = [r for r in train_mimic if clean_observed(r)]
    test = [r for r in test_mimic if clean_observed(r) or r["mention_status"] == "unmentioned"]
    for r in train + test:
        if r["mention_status"] == "observed_pos" and r["structured_positive"]:
            r["y3"], r["state"] = "Supported", "supported"
        elif r["mention_status"] == "observed_neg" and not r["structured_positive"]:
            r["y3"], r["state"] = "Refuted", "refuted"
        else:
            r["y3"], r["state"] = "Unknown", "unmentioned"

    feature_fn = lambda r: pair_features(Ez[r["hadm_id"]], P[r["proposition"]])
    metrics = run_sru_trap(rows, Ez, P, emb, train_rows=train, test_rows=test, feature_fn=feature_fn)
    return {
        "corpus": "mimic_struct_bridge",
        "model": model,
        "n_train_sr": len(train),
        "n_test": len(test),
        "n_test_unmentioned": sum(r["y3"] == "Unknown" for r in test),
        "metrics": metrics,
        "note": "Withhold unmentioned; train observed pos/neg only (natural C×O bridge).",
    }


def main():
    out = ROOT / "outputs" / "label_withholding_experiment.json"
    results = {"protocol": "train_EC_or_SR_test_withheld_third_class", "runs": {}}
    if out.exists():
        results = json.loads(out.read_text(encoding="utf-8"))

    mednli = load_mednli_local()
    if mednli is None:
        results["mednli_status"] = (
            "skipped: place PhysioNet MedNLI files under data/mednli/ "
            "(mli_train_v1.jsonl, mli_dev_v1.jsonl, mli_test_v1.jsonl)"
        )
    else:
        results["mednli_status"] = f"loaded {len(mednli)} rows from {MEDNLI_DIR}"

    models = ["tfidf", "qwen", "jina", "biolord"]
    for model in models:
        device = None if model == "tfidf" else "cuda"
        mimic_key = f"mimic/{model}"
        if mimic_key not in results["runs"]:
            print(f"evaluating {mimic_key} ...", flush=True)
            results["runs"][mimic_key] = eval_mimic_bridge(model, device=device)
            out.write_text(json.dumps(results, indent=2), encoding="utf-8")
            test = results["runs"][mimic_key]["metrics"]["test"]
            print(f"  UNR={test.get('UNR')} known={test.get('known_acc')}", flush=True)

        if mednli is not None:
            med_key = f"mednli/{model}"
            if med_key not in results["runs"]:
                print(f"evaluating {med_key} ...", flush=True)
                results["runs"][med_key] = eval_mednli(mednli, model, device=device)
                out.write_text(json.dumps(results, indent=2), encoding="utf-8")
                test = results["runs"][med_key]["metrics"]["test"]
                print(f"  neutral UNR={test.get('UNR')} known={test.get('known_acc')}", flush=True)

    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
