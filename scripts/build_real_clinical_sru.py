#!/usr/bin/env python3
"""Build human-gold and silver real clinical S/R/U benchmark rows."""
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_dual_human_deepseek import (  # noqa: E402
    ALLOWED,
    load_audit_lookup,
    load_human1,
    load_human2,
)
from real_clinical_sru_utils import (  # noqa: E402
    EXPANSION_CONCEPTS,
    concept_group,
    map_qwen_to_sru,
    observation_to_sru,
)
from silentehr import load_rows  # noqa: E402

OUT_HUMAN = ROOT / "data" / "real_clinical_sru.jsonl"
OUT_SILVER = ROOT / "data" / "real_clinical_sru_silver.jsonl"
OUT_STATS = ROOT / "outputs" / "real_clinical_sru_stats.json"


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


def struct_mention_to_sru(row):
    ms = row.get("full_note_mention_status") or row.get("mention_status")
    if ms == "observed_pos":
        return "Supported", "supported"
    if ms == "observed_neg":
        return "Refuted", "refuted"
    if ms in ("unmentioned", None):
        return "Unknown", "unmentioned"
    return "Unknown", "uncertain"


def build_human_rows():
    h1 = load_human1()
    h2 = load_human2()
    by_audit, _ = load_audit_lookup()
    h1_map = {
        (r["audit_id"], r["concept"]): r["label"].strip()
        for _, r in h1.iterrows() if r["label"] in ALLOWED
    }
    h2_map = {
        (r["audit_id"], r["concept"]): r["label"].strip()
        for _, r in h2.iterrows() if r["label"] in ALLOWED
    }
    rows = []
    for audit_id, concept in sorted(h2_map):
        if concept != by_audit[audit_id]["audit_row"]["concept"]:
            continue
        if (audit_id, concept) not in h1_map:
            continue
        struct_row = by_audit[audit_id]["struct_row"]
        h1_label = h1_map[(audit_id, concept)]
        h2_label = h2_map[(audit_id, concept)]
        consensus = h1_label if h1_label == h2_label else None
        label_obs = consensus or h2_label
        y3, state = observation_to_sru(label_obs)
        tier = "human_consensus" if consensus else "human_adjudicated"
        rows.append({
            "id": f"audit:{audit_id}:{concept}",
            "audit_id": audit_id,
            "subject_id": struct_row["subject_id"],
            "hadm_id": struct_row["hadm_id"],
            "concept": concept,
            "concept_id": concept,
            "concept_group": concept_group(concept),
            "proposition": struct_row["proposition"],
            "profile": struct_row["text"],
            "y3": y3,
            "state": state,
            "observation_status": label_obs,
            "human1": h1_label,
            "human2": h2_label,
            "human_consensus": consensus is not None,
            "label_tier": tier,
            "source": "mimic_audit_300",
        })
    return rows


def build_silver_expansion(human_rows, target=200, seed=0):
    """Sample diverse concept-group pairs from MIMIC struct (silver S/R/U)."""
    audit_hadm = {r["hadm_id"] for r in human_rows}
    struct_rows = load_rows(ROOT / "data/mimic_struct.jsonl")
    deepseek = load_deepseek_by_hadm()
    rng = random.Random(seed)
    by_group = defaultdict(list)
    for row in struct_rows:
        concept = row["concept"]
        if concept not in EXPANSION_CONCEPTS:
            continue
        if row["hadm_id"] in audit_hadm:
            continue
        if not row.get("text") or not row.get("proposition"):
            continue
        ms = row.get("full_note_mention_status") or row.get("mention_status")
        if ms in (None, "observed_mixed", "observed_uncertain", "observed_historical"):
            continue
        y3, state = struct_mention_to_sru(row)
        by_group[concept_group(concept)].append({
            **row,
            "y3": y3,
            "state": state,
            "concept_id": concept,
            "concept_group": concept_group(concept),
            "profile": row["text"],
            "label_tier": "silver_struct",
            "source": "mimic_struct",
        })

    picked = []
    per_group = max(1, target // max(len(by_group), 1))
    for group in sorted(by_group):
        pool = by_group[group]
        rng.shuffle(pool)
        picked.extend(pool[:per_group])
    if len(picked) < target:
        rest = [r for g in by_group for r in by_group[g] if r not in picked]
        rng.shuffle(rest)
        picked.extend(rest[: target - len(picked)])

    # Same-note multi-concept silver from audit admissions (Qwen full-note labels)
    audit_by_hadm = {r["hadm_id"]: r for r in human_rows}
    for hadm_id, base in audit_by_hadm.items():
        for concept in EXPANSION_CONCEPTS:
            if concept == base["concept"]:
                continue
            obs = deepseek.get((hadm_id, concept))
            if obs is None:
                continue
            if obs == "observed_positive":
                y3, state = "Supported", "supported"
            elif obs == "observed_negative":
                y3, state = "Refuted", "refuted"
            else:
                y3, state = "Unknown", "unmentioned"
            prop_map = {
                "chest_pain": "The patient currently has chest pain.",
                "nausea": "The patient currently has nausea.",
                "dyspnea": "The patient currently has dyspnea.",
                "penicillin_allergy": "The patient has a penicillin allergy.",
                "contrast_allergy": "The patient has a contrast allergy.",
                "anticoagulation": "The patient is currently on anticoagulation.",
                "insulin_therapy": "The patient is currently receiving insulin therapy.",
                "renal_dysfunction": "The patient currently has renal dysfunction.",
                "liver_dysfunction": "The patient currently has liver dysfunction.",
                "anemia": "The patient had anemia near discharge.",
                "thrombocytopenia": "The patient had thrombocytopenia near discharge.",
                "hyperkalemia": "The patient had hyperkalemia near discharge.",
            }
            picked.append({
                "id": f"audit_silver:{base['audit_id']}:{concept}",
                "audit_id": base["audit_id"],
                "subject_id": base["subject_id"],
                "hadm_id": hadm_id,
                "concept": concept,
                "concept_id": concept,
                "concept_group": concept_group(concept),
                "proposition": prop_map.get(concept, f"The patient has {concept.replace('_', ' ')}."),
                "profile": base["profile"],
                "y3": y3,
                "state": state,
                "observation_status": obs,
                "label_tier": "silver_qwen_note",
                "source": "mimic_audit_note_expansion",
            })
    return picked[: target + len(audit_by_hadm) * 8]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows, name):
    return {
        "name": name,
        "n": len(rows),
        "y3": dict(Counter(r["y3"] for r in rows)),
        "concept_group": dict(Counter(r["concept_group"] for r in rows)),
        "tier": dict(Counter(r.get("label_tier", "?") for r in rows)),
        "consensus_rate": (
            sum(r.get("human_consensus", False) for r in rows) / len(rows)
            if rows and "human_consensus" in rows[0] else None
        ),
    }


def main():
    human = build_human_rows()
    silver = build_silver_expansion(human, target=200)
    expansion_h1 = ROOT / "data" / "human1_new.csv.csv"
    if expansion_h1.exists():
        import subprocess
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "ingest_expansion_annotations.py")],
            check=True,
        )
        human = [
            json.loads(line)
            for line in (ROOT / "data" / "real_clinical_sru.jsonl").open(encoding="utf-8")
        ]
    write_jsonl(OUT_HUMAN, human)
    write_jsonl(OUT_SILVER, silver)
    stats = {
        "human": summarize(human, "real_clinical_sru"),
        "silver": summarize(silver, "real_clinical_sru_silver"),
        "mapping": {
            "observed_positive/negative": "Supported/Refuted",
            "unmentioned/uncertain/historical_only/admission_only/temporal_conflict": "Unknown",
        },
    }
    OUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    OUT_STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"wrote {OUT_HUMAN} ({len(human)} rows)")
    print(f"wrote {OUT_SILVER} ({len(silver)} rows)")


if __name__ == "__main__":
    main()
