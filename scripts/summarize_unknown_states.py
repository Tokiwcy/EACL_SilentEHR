#!/usr/bin/env python3
"""Extract per-unknown-state metrics from existing controlled-benchmark runs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

MODELS = [
    ("TF-IDF", "tfidf"),
    ("ClinicalBERT", "clinicalbert"),
    ("Qwen3-Emb", "qwen"),
    ("Jina v5", "jina"),
    ("BioLORD", "biolord"),
    ("MedCPT", "medcpt"),
    ("med-gte", "med-gte"),
]

STATES = ["unmentioned", "not_assessed", "uncertain"]


def load_model(key):
    path = OUT / f"results_{key}_full.json"
    nb = json.loads(path.read_text(encoding="utf-8"))[
        "binary_trap_by_protocol"]["new_background"]
    row = {
        "model": key,
        "known_acc": nb["known_acc"],
        "unmentioned_auroc": nb["unmentioned_auroc"],
        "unknown_auroc": nb["unknown_auroc"],
    }
    for st in STATES:
        row[f"UNR_{st}"] = nb["UNR_by_state"][st]
        row[f"HCUNR_{st}"] = nb["HCUNR_by_state@0.9"][st]
        da = nb["decision_axis_projection"].get(st, {})
        row[f"logit_mean_{st}"] = da.get("mean_logit")
        row[f"logit_q50_{st}"] = da.get("q50")
    return row


def main():
    rows = [load_model(key) for _, key in MODELS]
    out = OUT / "unknown_state_metrics.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved {out}\n")
    print(
        f"{'Model':<14} {'Known':>6} "
        f"{'U→R':>7} {'NA→R':>7} {'Unc→R':>7} "
        f"{'HC-U':>6} {'HC-NA':>6} {'HC-Unc':>7} "
        f"{'Det-U':>6} {'logU':>7} {'logNA':>7} {'logUnc':>7}"
    )
    for r, (label, _) in zip(rows, MODELS):
        print(
            f"{label:<14} {r['known_acc']:>6.3f} "
            f"{r['UNR_unmentioned']:>8.3f} {r['UNR_not_assessed']:>8.3f} "
            f"{r['UNR_uncertain']:>8.3f} "
            f"{r['HCUNR_unmentioned']:>7.3f} {r['HCUNR_not_assessed']:>7.3f} "
            f"{r['HCUNR_uncertain']:>7.3f} "
            f"{r['unmentioned_auroc']:>6.3f} "
            f"{r['logit_mean_unmentioned']:>8.2f} "
            f"{r['logit_mean_not_assessed']:>9.2f} "
            f"{r['logit_mean_uncertain']:>10.2f}"
        )


if __name__ == "__main__":
    main()
