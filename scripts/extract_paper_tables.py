#!/usr/bin/env python3
"""Extract LaTeX-ready summaries for abstention and mitigation tables."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

MODELS = [
    ("TF--IDF", "tfidf"),
    ("ClinicalBERT", "clinicalbert"),
    ("Qwen3-Emb", "qwen"),
    ("Jina v5", "jina"),
    ("BioLORD", "biolord"),
    ("MedCPT", "medcpt"),
    ("med-gte", "med-gte"),
]

MITIGATION_MODELS = [
    ("Qwen3-Emb", "qwen"),
    ("Jina v5", "jina"),
    ("BioLORD", "biolord"),
    ("MedCPT", "medcpt"),
    ("med-gte", "med-gte"),
]


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def fmt(x, digits=3):
    if x is None:
        return "---"
    return f"{x:.{digits}f}"


def extract_abstention():
    rows = []
    for label, key in MODELS:
        d = load(f"results_{key}_full.json")
        nb = d["binary_trap_by_protocol"]["new_background"]
        conf = nb.get("confidence_abstention") or {}
        maha = (nb.get("mahalanobis_ood") or {}).get("abstention") or {}
        rows.append({
            "encoder": label,
            "known_coverage": conf.get("known_test_coverage"),
            "unm_reject_conf": conf.get("unmentioned_reject_rate"),
            "unk_reject_conf": conf.get("unknown_reject_rate"),
            "accepted_unm_unr_conf": conf.get("accepted_unmentioned_UNR"),
            "unm_reject_maha": maha.get("unmentioned_reject_rate"),
            "unk_reject_maha": maha.get("unknown_reject_rate"),
            "accepted_unm_unr_maha": maha.get("accepted_unmentioned_UNR"),
        })
    return rows


def extract_mitigation_curve():
    curves = {}
    for label, key in MITIGATION_MODELS:
        d = load(f"results_{key}_full.json")
        curve = d["mitigation_by_protocol"]["new_background"]["fewshot_unknown_10_seed"]
        base_unr = d["binary_trap_by_protocol"]["new_background"]["UNR_by_state"]["unmentioned"]
        entry = {"base_UNR": base_unr}
        for n in (10, 50, 100):
            entry[str(n)] = {
                metric: curve[str(n)][metric]["mean"]
                for metric in ("UNR", "known_acc", "unknown_recall")
            }
        curves[label] = entry
    return curves


def extract_real_sru_cv():
    path = OUT / "real_clinical_sru_cv.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["datasets"]["human_consensus"]["models"]


def main():
    abst = extract_abstention()
    print("=== Abstention (new-background; thresholds fit on known training only) ===")
    print("Encoder | Known cov | Unm reject | Accepted UnmRR | Maha reject")
    for row in abst:
        print(
            row["encoder"],
            fmt(row["known_coverage"]),
            fmt(row["unm_reject_conf"]),
            fmt(row["accepted_unm_unr_conf"]),
            fmt(row["unm_reject_maha"]),
        )

    curves = extract_mitigation_curve()
    print("\n=== Unknown supervision curve (new-background, 10-seed mean) ===")
    header = "Encoder | base UNR | 10: UNR/Urec/Known | 50: ... | 100: ..."
    print(header)
    for label, entry in curves.items():
        parts = [label, fmt(entry["base_UNR"])]
        for n in (10, 50, 100):
            c = entry[str(n)]
            parts.append(
                f"{fmt(c['UNR'])}/{fmt(c['unknown_recall'])}/{fmt(c['known_acc'])}"
            )
        print(" | ".join(parts))

    cv = extract_real_sru_cv()
    if cv:
        print("\n=== Real-note 5-fold pooled OOF (human consensus) ===")
        for model, res in cv.items():
            p = res["pooled_oof"]
            from stats_utils import format_wilson_tex
            print(
                model,
                "known", format_wilson_tex(p.get("known_acc")),
                "UNR", format_wilson_tex(p.get("UNR")),
                "HC", format_wilson_tex(p.get("HCUNR@0.9")),
                "adm", format_wilson_tex(p.get("UNR_admission_only")),
                "strict", format_wilson_tex(p.get("UNR_strict_unmentioned")),
            )

    summary = {
        "abstention_new_background": abst,
        "mitigation_curve_new_background": curves,
        "real_clinical_sru_cv": cv,
    }
    out = OUT / "paper_table_summaries.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
