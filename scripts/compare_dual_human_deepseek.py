#!/usr/bin/env python3
"""Compare LLM observation labels vs human1.csv and human2.csv."""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from silentehr import OBSERVATION_LABEL_MODEL  # noqa: E402

ALLOWED = {
    "observed_positive", "observed_negative", "uncertain",
    "historical_only", "admission_only", "temporal_conflict",
    "unmentioned",
}


def load_struct_index():
    index = defaultdict(list)
    rows = [json.loads(line) for line in
            (ROOT / "data/mimic_struct_qwen3.jsonl").open(encoding="utf-8")
            if (ROOT / "data/mimic_struct_qwen3.jsonl").exists()
            else (ROOT / "data/mimic_struct_deepseek.jsonl").open(encoding="utf-8")]
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


def load_audit_lookup():
    audit = list(csv.DictReader(
        (ROOT / "data/mimic_audit_300.csv").open(encoding="utf-8-sig")))
    key = {
        row["audit_id"]: row
        for row in csv.DictReader(
            (ROOT / "data/mimic_audit_300_key.csv").open(encoding="utf-8-sig"))
    }
    struct_index = load_struct_index()
    by_audit = {}
    by_hadm = {}
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
        struct_row = candidates[0]
        by_audit[audit_id] = {
            "audit_row": audit_row,
            "key_row": key_row,
            "struct_row": struct_row,
            "hadm_id": struct_row["hadm_id"],
        }
        by_hadm.setdefault(struct_row["hadm_id"], struct_row)
    return by_audit, by_hadm


def load_qwen3_by_hadm():
    labels_path = ROOT / "data/mimic_qwen3_labels.jsonl"
    if not labels_path.exists():
        labels_path = ROOT / "data/mimic_deepseek_labels.jsonl"
    labels = {}
    for line in labels_path.open(encoding="utf-8"):
        obj = json.loads(line)
        for lab in obj["labels"]:
            labels[(obj["hadm_id"], lab["concept"])] = lab["observation_status"]
    return labels


load_deepseek_by_hadm = load_qwen3_by_hadm


def load_human1():
    import csv
    import io

    csv_path = ROOT / "data" / "human1.csv"
    xls_path = ROOT / "data" / "human1.xls"
    if csv_path.exists():
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        df["audit_id"] = df["audit_id"].astype(str).str.strip()
        df["concept"] = df["concept"].astype(str).str.strip()
        df["label"] = df["label"].astype(str).str.strip()
        if "evidence_span" not in df.columns:
            df["evidence_span"] = ""
        else:
            df["evidence_span"] = df["evidence_span"].fillna("").astype(str).str.strip()
        return df

    raw = pd.read_excel(xls_path, sheet_name="Sheet1")
    col = raw.columns[0]
    rows = []
    for line in raw[col].astype(str):
        for parts in csv.reader(io.StringIO(line)):
            if len(parts) < 3:
                continue
            rows.append({
                "audit_id": parts[0].strip(),
                "concept": parts[1].strip(),
                "label": parts[2].strip(),
                "evidence_span": parts[3].strip() if len(parts) > 3 else "",
            })
            break
    return pd.DataFrame(rows)


def load_human2():
    df = pd.read_csv(ROOT / "data" / "human2.csv", encoding="utf-8-sig")
    df["label"] = df["label"].astype(str).str.strip()
    return df


def compare_status(human_labels, llm_labels, name):
    pairs = list(zip(human_labels, llm_labels))
    n = len(pairs)
    acc = sum(a == b for a, b in pairs) / n if n else float("nan")
    kappa = float(cohen_kappa_score(human_labels, llm_labels)) if n >= 2 else None
    print(f"\n=== {name} vs {OBSERVATION_LABEL_MODEL} (observation_status) ===")
    print(f"n={n}, accuracy={acc:.1%}, cohen_kappa={kappa:.3f}")
    print("human distribution:", dict(Counter(human_labels)))
    print("llm distribution:", dict(Counter(llm_labels)))
    print("confusion (human -> llm), top 10:")
    for pair, count in Counter(pairs).most_common(10):
        print(f"  {pair[0]} -> {pair[1]}: {count}")
    by_concept = defaultdict(lambda: {"n": 0, "ok": 0})
    return acc, kappa, pairs


def main():
    by_audit, _ = load_audit_lookup()
    deepseek = load_deepseek_by_hadm()
    h1 = load_human1()
    h2 = load_human2()

    # human1: 600 rows (both concepts per audit_id)
    h1_human, h1_llm, h1_meta = [], [], []
    for _, row in h1.iterrows():
        audit_id = row["audit_id"]
        concept = row["concept"]
        if row["label"] not in ALLOWED:
            continue
        hadm_id = by_audit[audit_id]["hadm_id"]
        llm = deepseek.get((hadm_id, concept))
        if llm is None:
            continue
        h1_human.append(row["label"])
        h1_llm.append(llm)
        h1_meta.append((audit_id, concept, row["label"], llm))

    acc1, kappa1, pairs1 = compare_status(h1_human, h1_llm, "Human1 (600 pairs)")
    for concept in ("anemia", "thrombocytopenia"):
        sub = [(h, l) for (_, c, h, l) in h1_meta if c == concept]
        if sub:
            a = sum(x == y for x, y in sub) / len(sub)
            print(f"  {concept}: {a:.1%} (n={len(sub)})")

    # human2: 300 audit rows (one concept each, matches audit csv)
    h2_human, h2_llm, h2_meta = [], [], []
    for _, row in h2.iterrows():
        audit_id = row["audit_id"]
        concept = row["concept"]
        label = row["label"]
        if label not in ALLOWED or pd.isna(label):
            continue
        info = by_audit[audit_id]
        if info["audit_row"]["concept"] != concept:
            print(f"warning: {audit_id} concept mismatch h2={concept} "
                  f"audit={info['audit_row']['concept']}")
        hadm_id = info["hadm_id"]
        llm = deepseek.get((hadm_id, concept))
        if llm is None:
            continue
        h2_human.append(label)
        h2_llm.append(llm)
        h2_meta.append((audit_id, concept, label, llm))

    acc2, kappa2, _ = compare_status(h2_human, h2_llm, "Human2 (300 rows)")
    for concept in ("anemia", "thrombocytopenia"):
        sub = [(h, l) for (_, c, h, l) in h2_meta if c == concept]
        if sub:
            a = sum(x == y for x, y in sub) / len(sub)
            print(f"  {concept}: {a:.1%} (n={len(sub)})")

    # human1 vs human2 on shared audit_id+concept (150 anemia + 150 thrombocytopenia)
    h2_map = {
        (r["audit_id"], r["concept"]): r["label"]
        for _, r in h2.iterrows()
        if str(r["label"]) in ALLOWED
    }
    inter_h1, inter_h2 = [], []
    for audit_id, concept, label1, _ in h1_meta:
        key = (audit_id, concept)
        if key in h2_map and key[1] == by_audit[audit_id]["audit_row"]["concept"]:
            inter_h1.append(label1)
            inter_h2.append(h2_map[key])
    if inter_h1:
        acc12 = sum(a == b for a, b in zip(inter_h1, inter_h2)) / len(inter_h1)
        k12 = float(cohen_kappa_score(inter_h1, inter_h2))
        print(f"\n=== Human1 vs Human2 (matched audit concept, n={len(inter_h1)}) ===")
        print(f"accuracy={acc12:.1%}, cohen_kappa={k12:.3f}")
        print("confusion human1 -> human2:")
        for pair, count in Counter(zip(inter_h1, inter_h2)).most_common(10):
            print(f"  {pair[0]} -> {pair[1]}: {count}")

    # Collapsed binary: unmentioned vs any-mention
    def collapse(s):
        return "unmentioned" if s == "unmentioned" else "mentioned"

    for name, human, llm in (
        ("Human1", h1_human, h1_llm),
        ("Human2", h2_human, h2_llm),
    ):
        h = [collapse(x) for x in human]
        l = [collapse(x) for x in llm]
        acc = sum(a == b for a, b in zip(h, l)) / len(h)
        print(f"\n{name} binary mention (unmentioned vs mentioned) vs LLM: "
              f"{acc:.1%} (n={len(h)})")

    summary = {
        "observation_label_model": OBSERVATION_LABEL_MODEL,
        "human1": {
            "n": len(h1_human),
            "accuracy": acc1,
            "cohen_kappa": kappa1,
            "human_distribution": dict(Counter(h1_human)),
            "llm_distribution": dict(Counter(h1_llm)),
            "by_concept": {
                concept: sum(h == l for (_, c, h, l) in h1_meta if c == concept)
                / sum(1 for _, c, _, _ in h1_meta if c == concept)
                for concept in ("anemia", "thrombocytopenia")
            },
        },
        "human2": {
            "n": len(h2_human),
            "accuracy": acc2,
            "cohen_kappa": kappa2,
            "human_distribution": dict(Counter(h2_human)),
            "llm_distribution": dict(Counter(h2_llm)),
            "by_concept": {
                concept: sum(h == l for (_, c, h, l) in h2_meta if c == concept)
                / sum(1 for _, c, _, _ in h2_meta if c == concept)
                for concept in ("anemia", "thrombocytopenia")
            },
        },
        "human1_vs_human2_matched": {
            "n": len(inter_h1),
            "accuracy": acc12 if inter_h1 else None,
            "cohen_kappa": k12 if inter_h1 else None,
        },
        "confusion_human1_llm": Counter(pairs1).most_common(),
        "confusion_human2_llm": Counter(
            zip(h2_human, h2_llm)).most_common(),
    }
    out = ROOT / "outputs" / "dual_human_deepseek_comparison.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
