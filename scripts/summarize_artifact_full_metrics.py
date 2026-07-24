#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "outputs"
MODELS = ["tfidf", "clinicalbert", "qwen", "jina", "biolord", "medcpt", "med-gte"]
TIERS = [("_full", "clean"), ("_lengthmatched", "length"),
         ("_randpos", "randpos"), ("_indirect", "indirect")]

rows = []
for model in MODELS:
    for suffix, tier in TIERS:
        path = ROOT / f"results_{model}{suffix}.json"
        if not path.exists():
            continue
        trap = json.loads(path.read_text())["binary_trap_by_protocol"]["new_background"]
        rows.append({
            "model": model,
            "tier": tier,
            "known_acc": trap["known_acc"],
            "unm_unr": trap["UNR_by_state"]["unmentioned"],
            "hcunr": trap["HCUNR_by_state@0.9"]["unmentioned"],
            "det_auroc": trap.get("unmentioned_auroc"),
        })

print("model\ttier\tknown\tunm_unr\thcunr\tdet")
for r in rows:
    print(f"{r['model']}\t{r['tier']}\t{r['known_acc']:.3f}\t{r['unm_unr']:.3f}\t{r['hcunr']:.3f}\t{r['det_auroc']:.3f}")

(ROOT / "artifact_tier_full_metrics.json").write_text(
    json.dumps(rows, indent=2), encoding="utf-8")
