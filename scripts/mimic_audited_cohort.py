#!/usr/bin/env python3
"""Per-concept MIMIC predictions for audited C+O- cohort (matches mimic_cxo JSON)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mimic_eval_utils import (  # noqa: E402
    encode_propositions,
    load_mimic_embeddings,
    mimic_split,
    notes_dict,
)
from silentehr import pair_features  # noqa: E402

AUDITED = {"anemia", "thrombocytopenia"}
TAU = 0.9
HCWE_THRESH = 1 - TAU  # 0.1, matches silentehr.mimic_cxo_experiment

MODEL_TO_MAIN = {
    "tfidf": "TF-IDF",
    "qwen": "Qwen3-Emb",
    "jina": "Jina v5",
    "biolord": "BioLORD",
    "clinicalbert": "ClinicalBERT",
    "medcpt": "MedCPT",
    "med-gte": "med-gte",
}


def load_main_table_audited():
    path = ROOT / "outputs" / "mimic_concept_subset_pooled.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        model: data[label]["audited"]
        for model, label in MODEL_TO_MAIN.items()
    }


def audited_cplus_ominus_predictions(model, seed=0, cap_per_cell=500, device="cuda"):
    """Per-concept fit/predict — same protocol as silentehr.mimic_cxo_experiment."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    train, test, eligible = mimic_split(seed=seed, cap_per_cell=cap_per_cell)
    notes = notes_dict(train, test)
    emb, Ez = load_mimic_embeddings(model, notes, train, device=device)
    P = encode_propositions(emb, train + test)

    def features(subset):
        return np.asarray([
            pair_features(Ez[r["hadm_id"]], P[r["proposition"]]) for r in subset])

    rows, probs = [], []
    by_concept = {}

    for concept in eligible:
        tr = [r for r in train if r["concept"] == concept]
        te = [r for r in test if r["concept"] == concept]
        if not tr or not te:
            continue
        Xtr = features(tr)
        Xte = features(te)
        ytr = np.asarray([int(r["structured_positive"]) for r in tr])
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=seed)
        clf.fit(scaler.transform(Xtr), ytr)
        p = clf.predict_proba(scaler.transform(Xte))[
            :, list(clf.classes_).index(1)]

        cplus_unm_idx = [
            i for i, r in enumerate(te)
            if r["structured_positive"] and r["mention_status"] == "unmentioned"
        ]
        if concept in AUDITED:
            for i in cplus_unm_idx:
                rows.append(te[i])
                probs.append(float(p[i]))

        if cplus_unm_idx:
            sub_p = p[cplus_unm_idx]
            by_concept[concept] = {
                "n": len(cplus_unm_idx),
                "UPFNR": float(np.mean(sub_p < 0.5)),
                "HCWE@0.9": float(np.mean(sub_p < HCWE_THRESH)),
            }

    probs = np.asarray(probs, dtype=float)
    return rows, probs, by_concept


def pooled_audited_metrics(probs, by_concept):
    n = len(probs)
    e = int(np.sum(probs < 0.5))
    observed = e / n if n else float("nan")
    hcwe = float(np.mean(probs < HCWE_THRESH)) if n else float("nan")
    weighted_upfnr = None
    if by_concept:
        num = sum(v["UPFNR"] * v["n"] for v in by_concept.values())
        den = sum(v["n"] for v in by_concept.values())
        weighted_upfnr = num / den if den else None
    return {
        "N": n,
        "E": e,
        "observed_UPFNR": observed,
        "weighted_UPFNR_by_concept": weighted_upfnr,
        "HCWE@0.9": hcwe,
        "by_concept": by_concept,
    }


def assert_matches_main_table(model, metrics, tol=1e-9):
    main = load_main_table_audited()[model]
    ref_upfnr = main["UPFNR"]
    ref_n = main["n_unmentioned_positive"]
    ref_hcwe = main["HCWE@0.9"]
    obs = metrics["observed_UPFNR"]
    assert metrics["N"] == 75, f"{model}: N={metrics['N']} != 75"
    assert metrics["N"] == ref_n, f"{model}: N={metrics['N']} != main n={ref_n}"
    assert abs(obs - ref_upfnr) < tol, (
        f"{model}: observed UPFNR {obs} != main table {ref_upfnr}")
    assert metrics["E"] == round(ref_upfnr * metrics["N"]), (
        f"{model}: E={metrics['E']} != round({ref_upfnr}*{metrics['N']})")
    assert abs(metrics["HCWE@0.9"] - ref_hcwe) < 1e-6, (
        f"{model}: HCWE {metrics['HCWE@0.9']} != main {ref_hcwe}")


if __name__ == "__main__":
    for model in MODEL_TO_MAIN:
        device = None if model == "tfidf" else "cuda"
        _, probs, by_concept = audited_cplus_ominus_predictions(
            model, device=device)
        metrics = pooled_audited_metrics(probs, by_concept)
        assert_matches_main_table(model, metrics)
        print(
            f"{model}: N={metrics['N']} E={metrics['E']} "
            f"UPFNR={metrics['observed_UPFNR']:.6f} OK")
