#!/usr/bin/env python3
"""Compute Wilson 95% CIs for audit-subset UPFNR (four encoders)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from stats_utils import format_wilson_tex, wilson_summary  # noqa: E402


def upfnr_wilson(metrics):
    n = metrics.get("n_C+", 0)
    if n <= 0:
        return None
    recall = metrics.get("C+_recall", 0.0)
    errors = round(recall * n)
    # UPFNR = 1 - recall(C+)
    return wilson_summary(n - errors, n)


def main():
    path = ROOT / "outputs" / "mimic_audit_human_sensitivity.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for model, res in data.items():
        out[model] = {}
        for subset, metrics in res["subsets"].items():
            if metrics.get("n", 0) == 0:
                continue
            w = upfnr_wilson(metrics)
            if w:
                out[model][subset] = {
                    "UPFNR": metrics["UPFNR"],
                    "wilson": w,
                    "tex": format_wilson_tex(w),
                }

    save = ROOT / "outputs" / "mimic_audit_wilson_ci.json"
    save.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved {save}\n")

    key_subsets = [
        "human_consensus_unmentioned_C+",
        "llm_unmentioned_C+",
    ]
    for model in ["tfidf", "qwen", "jina", "biolord"]:
        print(f"=== {model} ===")
        for subset in key_subsets:
            entry = out.get(model, {}).get(subset)
            if entry:
                print(f"  {subset}: {entry['tex']}")


if __name__ == "__main__":
    main()
