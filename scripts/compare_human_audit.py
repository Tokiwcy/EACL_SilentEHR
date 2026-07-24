#!/usr/bin/env python3
"""Compare human audit adjudication against LLM observation labels."""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from silentehr import (  # noqa: E402
    OBSERVATION_LABEL_MODEL,
    llm_observation_cell,
)

VALID = {"C+O+", "C-O+", "C+O-", "C-O-", "exclude"}


def load_struct_index():
    rows = [json.loads(line) for line in
            (ROOT / "data/mimic_struct_deepseek.jsonl").open(encoding="utf-8")]
    index = defaultdict(list)
    for row in rows:
        if row["concept"] not in ("anemia", "thrombocytopenia"):
            continue
        struct_float = (
            float(row["struct_value"])
            if row.get("struct_value") not in (None, "") else None
        )
        key = (
            row["concept"],
            struct_float,
            row.get("struct_charttime"),
            row.get("dischtime"),
            bool(row["structured_positive"]),
        )
        index[key].append(row)
    return index


def load_audit_rows():
    audit_csv = ROOT / "data/mimic_audit_300.csv"
    key_csv = ROOT / "data/mimic_audit_300_key.csv"
    audit = list(csv.DictReader(audit_csv.open(encoding="utf-8-sig")))
    key = {
        row["audit_id"]: row
        for row in csv.DictReader(key_csv.open(encoding="utf-8-sig"))
    }
    struct_index = load_struct_index()
    matched = []
    for audit_row in audit:
        audit_id = audit_row["audit_id"]
        key_row = key[audit_id]
        struct_float = (
            float(audit_row["struct_value"])
            if audit_row["struct_value"] else None
        )
        match_key = (
            audit_row["concept"],
            struct_float,
            audit_row["struct_charttime"],
            audit_row["dischtime"],
            key_row["structured_positive"] == "True",
        )
        candidates = struct_index[match_key]
        if len(candidates) != 1:
            raise RuntimeError(
                f"{audit_id}: expected 1 struct match, got {len(candidates)}")
        matched.append({
            "audit_id": audit_id,
            "key_row": key_row,
            "struct_row": candidates[0],
        })
    return matched


def main():
    human = pd.read_excel(
        ROOT / "data" / "human_annotator1.xlsx",
        sheet_name="mimic_audit_300_annotated",
    )
    audit_matches = {row["audit_id"]: row for row in load_audit_rows()}

    records = []
    for _, hr in human.iterrows():
        audit_id = hr["audit_id"]
        match = audit_matches[audit_id]
        struct_row = match["struct_row"]
        key_row = match["key_row"]
        structured_positive = key_row["structured_positive"] == "True"
        raw_status = struct_row.get("deepseek_observation_status", "unmentioned")
        llm_cell = llm_observation_cell(structured_positive, raw_status)
        human_cell = str(hr["adjudicated_cell"]).strip()
        records.append({
            "audit_id": audit_id,
            "concept": struct_row["concept"],
            "llm_cell": llm_cell,
            "llm_status": raw_status,
            "human_adjudicated": human_cell,
        })

    human_valid = [r for r in records if r["human_adjudicated"] in VALID]
    human_eval = [r for r in human_valid if r["human_adjudicated"] != "exclude"]
    excluded = sum(r["human_adjudicated"] == "exclude" for r in human_valid)

    def acc():
        return sum(
            r["llm_cell"] == r["human_adjudicated"] for r in human_eval
        ) / len(human_eval)

    kappa = float(cohen_kappa_score(
        [r["llm_cell"] for r in human_eval],
        [r["human_adjudicated"] for r in human_eval],
    ))

    print(f"=== Human vs {OBSERVATION_LABEL_MODEL} (audit 300) ===")
    print(f"rows: {len(records)}, evaluated: {len(human_eval)}, "
          f"excluded: {excluded}")
    print(f"cell accuracy: {acc():.1%}")
    print(f"cohen kappa: {kappa:.3f}")

    by_concept = {}
    for concept in ("anemia", "thrombocytopenia"):
        sub = [r for r in human_eval if r["concept"] == concept]
        by_concept[concept] = sum(
            r["llm_cell"] == r["human_adjudicated"] for r in sub) / len(sub)
        print(f"  {concept}: {by_concept[concept]:.1%} (n={len(sub)})")

    summary = {
        "observation_label_model": OBSERVATION_LABEL_MODEL,
        "n_total": len(records),
        "n_evaluated": len(human_eval),
        "n_excluded": excluded,
        "cell_assignment_accuracy": acc(),
        "cohen_kappa": kappa,
        "structured_C_accuracy": 1.0,
        "by_concept_accuracy": by_concept,
        "human_distribution": dict(
            Counter(r["human_adjudicated"] for r in human_valid)),
        "confusion_human_llm": Counter(
            (r["human_adjudicated"], r["llm_cell"]) for r in human_eval
        ).most_common(),
    }
    out = ROOT / "outputs" / "mimic_audit_metrics.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
