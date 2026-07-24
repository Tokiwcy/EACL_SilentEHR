#!/usr/bin/env python3
"""Summarize clean vs artifact-resistant tier UNR results."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MODELS = ["tfidf", "clinicalbert", "qwen", "jina", "biolord", "medcpt", "med-gte"]
TIERS = [
    ("clean", "_full"),
    ("lengthmatched", "_lengthmatched"),
    ("randpos", "_randpos"),
    ("indirect", "_indirect"),
]


def load_metric(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    trap = data["binary_trap_by_protocol"]["new_background"]
    return {
        "known": trap["known_acc"],
        "unr": trap["UNR"],
        "unm_unr": trap["UNR_by_state"]["unmentioned"],
        "hcunr": trap["HCUNR@0.9"],
        "unm_hcunr": trap["HCUNR_by_state@0.9"]["unmentioned"],
        "det_auroc": trap.get("unmentioned_auroc"),
    }


def main() -> None:
    rows = []
    for model in MODELS:
        row = {"model": model}
        for tier_name, suffix in TIERS:
            metrics = load_metric(OUT / f"results_{model}{suffix}.json")
            if metrics:
                row[f"{tier_name}_unm_unr"] = metrics["unm_unr"]
                row[f"{tier_name}_known"] = metrics["known"]
                row[f"{tier_name}_det"] = metrics["det_auroc"]
        rows.append(row)

    print("SilentEHR artifact-resistant tier summary (new-background protocol)")
    print("Columns: Unmentioned UNR by tier\n")
    header = f"{'Model':12s}" + "".join(f"{t:16s}" for t, _ in TIERS)
    print(header)
    print("-" * len(header))
    for row in rows:
        if not any(k.endswith("_unm_unr") for k in row):
            continue
        line = f"{row['model']:12s}"
        for tier_name, _ in TIERS:
            val = row.get(f"{tier_name}_unm_unr")
            line += f"{val:16.3f}" if val is not None else f"{'—':>16s}"
        print(line)

    summary = {"models": rows, "tiers": [t for t, _ in TIERS]}
    out = OUT / "artifact_tier_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
