#!/usr/bin/env python3
"""Real clinical S/R/U: train Supported/Refuted only; test Unknown."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mimic_eval_utils import (  # noqa: E402
    encode_extra_notes,
    encode_propositions,
    load_mimic_embeddings,
    mimic_split,
    notes_dict,
)
from real_clinical_sru_utils import (  # noqa: E402
    evaluate_split,
    save_split_manifest,
)
from silentehr import pair_features  # noqa: E402


def load_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def evaluate_dataset(rows, model, device="cuda", seed=0, dataset_name="human_consensus"):
    train_mimic, test_mimic, _ = mimic_split()
    notes = notes_dict(train_mimic, test_mimic)
    for r in rows:
        notes.setdefault(r["hadm_id"], r["profile"])
    emb, Ez = load_mimic_embeddings(
        model, notes_dict(train_mimic, test_mimic), train_mimic, device=device,
    )
    Ez = encode_extra_notes(
        emb, Ez, notes, {r["hadm_id"] for r in rows}, device=device,
    )
    P = encode_propositions(emb, train_mimic + test_mimic + rows)

    train, test, manifest = save_split_manifest(rows, seed=seed, dataset=dataset_name)
    feature_fn = lambda r: pair_features(Ez[r["hadm_id"]], P[r["proposition"]])
    result = evaluate_split(train, test, feature_fn)
    train_sr = [r for r in train if r["y3"] in ("Supported", "Refuted")]
    return {
        "model": model,
        "metrics": {"train": result["train"], "test": result["test"]},
        "classes": result["classes"],
        "direction_check": result["direction_check"],
        "test_predictions": result["test_predictions"],
        "split": {
            "manifest": manifest,
            "n_train_subjects": len(manifest["train_subject_ids"]),
            "n_test_subjects": len(manifest["test_subject_ids"]),
            "n_train_sr": len(train_sr),
            "n_test": len(test),
            "test_y3": dict(__import__("collections").Counter(r["y3"] for r in test)),
        },
    }


def main():
    human_path = ROOT / "data" / "real_clinical_sru.jsonl"
    silver_path = ROOT / "data" / "real_clinical_sru_silver.jsonl"
    out = ROOT / "outputs" / "real_clinical_sru_experiment.json"

    if not human_path.exists():
        raise SystemExit(f"missing {human_path}; run build_real_clinical_sru.py first")

    human_all = load_jsonl(human_path)
    human_consensus = [r for r in human_all if r["label_tier"] == "human_consensus"]
    silver = load_jsonl(silver_path) if silver_path.exists() else []

    models = ["tfidf", "qwen", "jina", "biolord"]
    results = {"protocol": "subject_split_70_30_train_SR_only", "datasets": {}}

    for dataset_name, subset in [
        ("human_all", human_all),
        ("human_consensus", human_consensus),
        ("silver", silver),
    ]:
        if not subset:
            continue
        results["datasets"][dataset_name] = {"n": len(subset), "models": {}}
        for model in models:
            key = f"{dataset_name}/{model}"
            print(f"evaluating {key} ...", flush=True)
            device = None if model == "tfidf" else "cuda"
            results["datasets"][dataset_name]["models"][model] = evaluate_dataset(
                subset, model, device=device, dataset_name=dataset_name,
            )
            out.write_text(json.dumps(results, indent=2), encoding="utf-8")
            test = results["datasets"][dataset_name]["models"][model]["metrics"]["test"]
            print(
                f"  test UNR={test.get('UNR')} known_acc={test.get('known_acc')} "
                f"classes={results['datasets'][dataset_name]['models'][model]['classes']}",
                flush=True,
            )

    print(f"saved {out}")


if __name__ == "__main__":
    main()
