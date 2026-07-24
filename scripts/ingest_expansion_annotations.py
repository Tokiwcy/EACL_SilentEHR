#!/usr/bin/env python3
"""Ingest dual-human expansion annotations into real_clinical_sru.jsonl."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from real_clinical_sru_utils import concept_group, observation_to_sru  # noqa: E402

HUMAN1 = ROOT / "data" / "human1_new.csv.csv"
HUMAN2 = ROOT / "data" / "human2_new.csv.csv"
OUT_JSONL = ROOT / "data" / "real_clinical_sru.jsonl"
OUT_STATS = ROOT / "outputs" / "real_clinical_sru_expansion_ingest.json"
ALLOWED = {
    "observed_positive", "observed_negative", "uncertain",
    "historical_only", "admission_only", "temporal_conflict",
    "unmentioned",
}


def load_csv(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = row["candidate_id"]
            rows[key] = row
    return rows


def pick_label(row: dict, prefix: str) -> str:
    label = (row.get(f"{prefix}_label") or "").strip()
    if not label:
        return ""
    if label not in ALLOWED:
        raise ValueError(f"{row['candidate_id']}: invalid {prefix}_label={label!r}")
    return label


def build_expansion_rows(h1: dict, h2: dict) -> tuple[list[dict], dict]:
    base_by_audit = {}
    for line in OUT_JSONL.open(encoding="utf-8"):
        r = json.loads(line)
        base_by_audit[r["audit_id"]] = r

    merged = []
    disagreements = []
    for cid in sorted(set(h1) | set(h2)):
        if cid not in h1 or cid not in h2:
            raise SystemExit(f"missing paired annotation for {cid}")
        r1, r2 = h1[cid], h2[cid]
        label1 = pick_label(r1, "human1")
        label2 = pick_label(r2, "human2")
        if not label1 or not label2:
            raise SystemExit(f"missing label on {cid}: h1={label1!r} h2={label2!r}")

        audit_id = r1["audit_id"]
        concept = r1["concept_id"]
        base = base_by_audit.get(audit_id)
        if base is None:
            raise SystemExit(f"unknown audit_id {audit_id} for {cid}")

        consensus = label1 if label1 == label2 else None
        if consensus is None:
            disagreements.append({
                "candidate_id": cid,
                "human1": label1,
                "human2": label2,
            })
            continue

        y3, state = observation_to_sru(consensus)
        tier = "human_consensus"
        row = {
            "id": f"audit:{audit_id}:{concept}",
            "audit_id": audit_id,
            "subject_id": int(r1["subject_id"]),
            "hadm_id": int(r1["hadm_id"]),
            "concept": concept,
            "concept_id": concept,
            "concept_group": concept_group(concept),
            "proposition": r1["proposition"],
            "profile": base["profile"],
            "y3": y3,
            "state": state,
            "observation_status": consensus,
            "human1": label1,
            "human2": label2,
            "human1_evidence_span": (r1.get("human1_evidence_span") or "").strip(),
            "human2_evidence_span": (r2.get("human2_evidence_span") or "").strip(),
            "human_consensus": True,
            "label_tier": tier,
            "source": "mimic_audit_300_expansion",
            "expansion_candidate_id": cid,
        }
        merged.append(row)

    stats = {
        "n_candidates": len(h1),
        "n_consensus": len(merged),
        "n_disagreements": len(disagreements),
        "disagreements": disagreements,
        "consensus_labels": dict(Counter(r["observation_status"] for r in merged)),
        "consensus_y3": dict(Counter(r["y3"] for r in merged)),
        "consensus_by_group": dict(Counter(r["concept_group"] for r in merged)),
        "strict_unmentioned": sum(
            1 for r in merged
            if r["observation_status"] == "unmentioned"
        ),
    }
    return merged, stats


def merge_jsonl(expansion_rows: list[dict]) -> list[dict]:
    existing = [json.loads(line) for line in OUT_JSONL.open(encoding="utf-8")]
    exp_ids = {r["id"] for r in expansion_rows}
    kept = [r for r in existing if r["id"] not in exp_ids]
    combined = kept + expansion_rows
    combined.sort(key=lambda r: (r["audit_id"], r["concept_id"]))
    return combined


def main():
    if not HUMAN1.exists() or not HUMAN2.exists():
        raise SystemExit(f"missing {HUMAN1} or {HUMAN2}")

    h1 = load_csv(HUMAN1)
    h2 = load_csv(HUMAN2)
    expansion, stats = build_expansion_rows(h1, h2)
    combined = merge_jsonl(expansion)

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for row in combined:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats["n_total_jsonl"] = len(combined)
    stats["n_human_consensus_total"] = sum(
        1 for r in combined if r.get("label_tier") == "human_consensus"
    )
    stats["strict_unmentioned_total"] = sum(
        1 for r in combined
        if r.get("label_tier") == "human_consensus"
        and r.get("observation_status") == "unmentioned"
    )
    stats["strict_unmentioned_nonlab"] = sum(
        1 for r in combined
        if r.get("label_tier") == "human_consensus"
        and r.get("observation_status") == "unmentioned"
        and r.get("concept_group") != "laboratory"
    )
    OUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    OUT_STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"updated {OUT_JSONL} (+{len(expansion)} expansion rows)")


if __name__ == "__main__":
    main()
