#!/usr/bin/env python3
"""Pool MIMIC metrics over concept subsets from by_concept JSON fields."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

MODELS = [
    ("TF-IDF", "mimic_cxo_tfidf_matched.json"),
    ("ClinicalBERT", "mimic_cxo_clinicalbert.json"),
    ("Qwen3-Emb", "mimic_cxo_qwen.json"),
    ("Jina v5", "mimic_cxo_jina.json"),
    ("BioLORD", "mimic_cxo_biolord.json"),
    ("MedCPT", "mimic_cxo_medcpt.json"),
    ("med-gte", "mimic_cxo_med-gte.json"),
]

SUBSETS = {
    "audited": ["anemia", "thrombocytopenia"],
    "hyperkalemia_only": ["hyperkalemia"],
    "all_three": ["anemia", "thrombocytopenia", "hyperkalemia"],
}

METRICS = [
    "UPFNR", "HCWE@0.9", "documentation_gap", "unmentioned_detection_AUROC",
]


def weighted_pool(by_concept, concepts, metric):
    num = den = 0.0
    for c in concepts:
        cell = by_concept.get(c, {})
        n = cell.get("n_unmentioned_positive")
        v = cell.get(metric)
        if n is None or v is None or n == 0:
            continue
        num += v * n
        den += n
    return None if den == 0 else num / den


def pool_row(by_concept, concepts):
    out = {}
    for m in METRICS:
        out[m] = weighted_pool(by_concept, concepts, m)
    out["n_unmentioned_positive"] = sum(
        by_concept.get(c, {}).get("n_unmentioned_positive") or 0
        for c in concepts
    )
    return out


def main():
    results = {}
    for label, fname in MODELS:
        path = OUT / fname
        if not path.exists():
            path = OUT / fname.replace("_matched", "")
        data = json.loads(path.read_text(encoding="utf-8"))
        by_concept = data["by_concept"]
        results[label] = {
            subset: pool_row(by_concept, concepts)
            for subset, concepts in SUBSETS.items()
        }

    out = OUT / "mimic_concept_subset_pooled.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {out}\n")
    print(f"{'Model':<14} {'Subset':<18} {'UPFNR':>7} {'HCWE':>7} {'n_C+O-':>7}")
    for label, _ in MODELS:
        for subset in SUBSETS:
            r = results[label][subset]
            print(
                f"{label:<14} {subset:<18} "
                f"{r['UPFNR']:7.3f} {r['HCWE@0.9']:7.3f} "
                f"{r['n_unmentioned_positive']:7.0f}"
            )


if __name__ == "__main__":
    main()
