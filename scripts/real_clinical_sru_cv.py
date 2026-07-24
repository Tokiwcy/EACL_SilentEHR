#!/usr/bin/env python3
"""Subject-level 5-fold CV for real clinical S/R/U benchmark."""
import json
import sys
from pathlib import Path

import numpy as np

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
    _json_safe,
    evaluate_split,
    metrics_from_predictions,
    predict_rows,
    subject_kfold_splits,
    _train_sr_classifier,
)
from silentehr import pair_features  # noqa: E402
from stats_utils import format_wilson_tex  # noqa: E402


def load_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def evaluate_cv(rows, model, *, n_folds=5, seed=0, device="cuda", tau=0.9):
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
    feature_fn = lambda r: pair_features(Ez[r["hadm_id"]], P[r["proposition"]])

    fold_metrics = []
    pooled_rows = []
    pooled_pred = []
    pooled_pref = []
    pooled_psup = []
    pooled_proba = []

    for fold_id, (train, test) in enumerate(subject_kfold_splits(rows, n_folds, seed)):
        clf, sc, train_sr, _ = _train_sr_classifier(train, feature_fn)
        bundle = predict_rows(test, clf, sc, feature_fn)
        fold = metrics_from_predictions(bundle, tau=tau, with_ci=False)
        fold["fold"] = fold_id
        fold["classes"] = list(clf.classes_)
        fold["n_train_subjects"] = len({r["subject_id"] for r in train})
        fold["n_test_subjects"] = len({r["subject_id"] for r in test})
        fold_metrics.append(fold)
        pooled_rows.extend(bundle["rows"])
        pooled_pred.extend(bundle["pred"].tolist())
        pooled_pref.extend(bundle["p_ref"].tolist())
        pooled_psup.extend(bundle["p_sup"].tolist())
        pooled_proba.extend(bundle["proba"].tolist())

    pooled = metrics_from_predictions(
        {
            "rows": pooled_rows,
            "pred": np.asarray(pooled_pred, dtype=object),
            "p_ref": np.asarray(pooled_pref),
            "p_sup": np.asarray(pooled_psup),
            "proba": np.asarray(pooled_proba),
            "classes": fold_metrics[0]["classes"] if fold_metrics else [],
        },
        tau=tau,
        with_ci=True,
    )
    return {
        "model": model,
        "protocol": f"subject_grouped_{n_folds}fold_cv_train_SR_only",
        "n_folds": n_folds,
        "seed": seed,
        "folds": fold_metrics,
        "pooled_oof": pooled,
    }


def evaluate_split7030(rows, model, *, seed=0, device="cuda", dataset="human_consensus"):
    """Single 70/30 split using saved manifest (parity with experiment.py)."""
    from real_clinical_sru_utils import load_split_manifest

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
    train, test, manifest = load_split_manifest(rows, dataset=dataset)
    feature_fn = lambda r: pair_features(Ez[r["hadm_id"]], P[r["proposition"]])
    return evaluate_split(train, test, feature_fn), manifest


def main():
    human_path = ROOT / "data" / "real_clinical_sru.jsonl"
    out = ROOT / "outputs" / "real_clinical_sru_cv.json"
    if not human_path.exists():
        raise SystemExit(f"missing {human_path}; run build_real_clinical_sru.py first")

    human_all = load_jsonl(human_path)
    human_consensus = [r for r in human_all if r["label_tier"] == "human_consensus"]
    models = ["tfidf", "qwen", "jina", "biolord"]
    results = {
        "protocol": "subject_grouped_5fold_cv_train_SR_only",
        "datasets": {},
    }

    dataset_name = "human_consensus"
    results["datasets"].setdefault(dataset_name, {"n": len(human_consensus), "models": {}})
    for model in models:
        print(f"evaluating 5-fold CV {dataset_name}/{model} ...", flush=True)
        device = None if model == "tfidf" else "cuda"
        results["datasets"][dataset_name]["models"][model] = evaluate_cv(
            human_consensus, model, device=device,
        )
        out.write_text(json.dumps(_json_safe(results), indent=2), encoding="utf-8")
        pooled = results["datasets"][dataset_name]["models"][model]["pooled_oof"]
        print(
            f"  pooled UNR={format_wilson_tex(pooled.get('UNR'))} "
            f"known={format_wilson_tex(pooled.get('known_acc'))}",
            flush=True,
        )

    print(f"saved {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--parity-7030":
        rows = [
            r for r in load_jsonl(ROOT / "data" / "real_clinical_sru.jsonl")
            if r["label_tier"] == "human_consensus"
        ]
        exp = json.loads(
            (ROOT / "outputs" / "real_clinical_sru_experiment.json").read_text(
                encoding="utf-8",
            ),
        )
        for model in ("tfidf", "qwen", "jina", "biolord"):
            device = None if model == "tfidf" else "cuda"
            fresh, _ = evaluate_split7030(rows, model, device=device)
            saved = exp["datasets"]["human_consensus"]["models"][model][
                "test_predictions"
            ]
            fresh_by_id = {r["row_id"]: r for r in fresh["test_predictions"]}
            mismatches = sum(
                1 for a in saved
                if fresh_by_id[a["row_id"]]["pred"] != a["pred"]
            )
            print(
                f"{model}: direction_ok={fresh['direction_check']['direction_ok']} "
                f"UNR={fresh['test']['UNR']} mismatches={mismatches}",
            )
        raise SystemExit(0)
    main()
