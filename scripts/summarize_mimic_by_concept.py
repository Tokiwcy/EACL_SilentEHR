#!/usr/bin/env python3
"""Extract per-concept MIMIC metrics from mimic_cxo_*.json files."""
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

CONCEPTS = ["anemia", "hyperkalemia", "thrombocytopenia"]
METRICS = ["UPFNR", "HCWE@0.9", "documentation_gap", "unmentioned_detection_AUROC"]


def main():
    table = {}
    for label, fname in MODELS:
        path = OUT / fname
        if not path.exists():
            path = OUT / fname.replace("_matched", "")
        data = json.loads(path.read_text(encoding="utf-8"))
        joint = data.get("by_concept") or data.get("joint_query_conditioned", {})
        if "UPFNR" in joint:
            joint = {c: joint for c in CONCEPTS}  # fallback
        table[label] = {}
        for concept in CONCEPTS:
            c = data.get("by_concept", {}).get(concept, {})
            table[label][concept] = {m: c.get(m) for m in METRICS}
            table[label][concept]["n_unmentioned_positive"] = c.get(
                "n_unmentioned_positive")

    out = OUT / "mimic_per_concept.json"
    out.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"saved {out}\n")

    print("UPFNR by model × concept:")
    print(f"{'Model':<14}", " ".join(f"{c[:4]:>8}" for c in CONCEPTS))
    for label, _ in MODELS:
        vals = table.get(label, {})
        print(
            f"{label:<14}",
            " ".join(
                f"{vals.get(c, {}).get('UPFNR', float('nan')):8.3f}"
                for c in CONCEPTS
            ),
        )


if __name__ == "__main__":
    main()
