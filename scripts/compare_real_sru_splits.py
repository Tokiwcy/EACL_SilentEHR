#!/usr/bin/env python3
"""Row-level consistency check: experiment.py vs evaluate_split on same 70/30 manifest."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mimic_eval_utils import (
    encode_extra_notes,
    encode_propositions,
    load_mimic_embeddings,
    mimic_split,
    notes_dict,
)
from real_clinical_sru_utils import evaluate_split, load_split_manifest
from silentehr import pair_features


def load_consensus():
    rows = [
        json.loads(line)
        for line in (ROOT / "data/real_clinical_sru.jsonl").open(encoding="utf-8")
    ]
    return [r for r in rows if r["label_tier"] == "human_consensus"]


def predictions_for_model(model, rows, train, test, device="cuda"):
    train_mimic, test_mimic, _ = mimic_split()
    notes = notes_dict(train_mimic, test_mimic)
    for r in rows:
        notes.setdefault(r["hadm_id"], r["profile"])
    emb, Ez = load_mimic_embeddings(
        model, notes_dict(train_mimic, test_mimic), train_mimic, device=device,
    )
    Ez = encode_extra_notes(emb, Ez, notes, {r["hadm_id"] for r in rows}, device=device)
    P = encode_propositions(emb, train_mimic + test_mimic + rows)
    feature_fn = lambda r: pair_features(Ez[r["hadm_id"]], P[r["proposition"]])
    return evaluate_split(train, test, feature_fn)


def main():
    rows = load_consensus()
    train, test, manifest = load_split_manifest(rows, dataset="human_consensus")
    exp_path = ROOT / "outputs" / "real_clinical_sru_experiment.json"
    exp = json.loads(exp_path.read_text(encoding="utf-8")) if exp_path.exists() else None

    models = ["tfidf", "qwen", "jina", "biolord"]
    report = {"manifest": manifest, "models": {}, "all_match": True}

    for model in models:
        device = None if model == "tfidf" else "cuda"
        fresh = predictions_for_model(model, rows, train, test, device=device)
        fresh_by_id = {r["row_id"]: r for r in fresh["test_predictions"]}

        mismatches = []
        if exp and model in exp["datasets"]["human_consensus"]["models"]:
            saved = exp["datasets"]["human_consensus"]["models"][model]["test_predictions"]
            saved_by_id = {r["row_id"]: r for r in saved}
            for rid, a in saved_by_id.items():
                b = fresh_by_id.get(rid)
                if b is None:
                    mismatches.append({"row_id": rid, "error": "missing in fresh"})
                    continue
                for key in ("pred", "p_supported", "p_refuted", "y3"):
                    if a[key] != b[key] and abs(float(a.get(key, 0)) - float(b.get(key, 0))) > 1e-9:
                        mismatches.append({
                            "row_id": rid,
                            "field": key,
                            "saved": a[key],
                            "fresh": b[key],
                        })
        else:
            mismatches.append({"error": "no saved experiment predictions"})

        report["models"][model] = {
            "classes": fresh["classes"],
            "direction_check": fresh["direction_check"],
            "test_metrics": fresh["test"],
            "n_mismatches": len(mismatches),
            "mismatches": mismatches[:10],
        }
        if mismatches:
            report["all_match"] = False
        print(
            f"{model}: classes={fresh['classes']} "
            f"direction_ok={fresh['direction_check'].get('direction_ok')} "
            f"UNR={fresh['test'].get('UNR')} mismatches={len(mismatches)}"
        )

    out = ROOT / "outputs" / "real_clinical_sru_split_parity.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {out}")
    if not report["all_match"]:
        raise SystemExit("split parity check failed")


if __name__ == "__main__":
    main()
