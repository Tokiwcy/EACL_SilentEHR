#!/usr/bin/env python3
"""Build human-annotation CSV for non-laboratory strict-unmentioned expansion."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Non-laboratory concepts for natural human-gold expansion.
EXPANSION_NONLAB = [
    {
        "concept_id": "chest_pain",
        "concept_group": "symptom",
        "proposition": "The patient currently has chest pain.",
        "pos_patterns": [
            r"\bchest pain\b", r"\bangina\b", r"\bsubsternal\b",
            r"\bpleuritic chest\b",
        ],
        "neg_patterns": [
            r"\bno chest pain\b", r"\bdenies chest pain\b",
            r"\bwithout chest pain\b", r"\bnegative for chest pain\b",
        ],
    },
    {
        "concept_id": "dyspnea",
        "concept_group": "symptom",
        "proposition": "The patient currently has dyspnea.",
        "pos_patterns": [
            r"\bdyspnea\b", r"\bdyspnoea\b", r"\bshortness of breath\b",
            r"\bshort of breath\b", r"\bsob\b", r"\bbreathless\b",
        ],
        "neg_patterns": [
            r"\bno dyspnea\b", r"\bdenies dyspnea\b", r"\bdenies sob\b",
            r"\bno shortness of breath\b", r"\bwithout dyspnea\b",
        ],
    },
    {
        "concept_id": "nausea",
        "concept_group": "symptom",
        "proposition": "The patient currently has nausea.",
        "pos_patterns": [
            r"\bnausea\b", r"\bnauseated\b", r"\bemesis\b", r"\bvomiting\b",
        ],
        "neg_patterns": [
            r"\bno nausea\b", r"\bdenies nausea\b", r"\bwithout nausea\b",
            r"\bno vomiting\b", r"\bdenies vomiting\b",
        ],
    },
    {
        "concept_id": "anticoagulation",
        "concept_group": "medication",
        "proposition": "The patient is currently on anticoagulation.",
        "pos_patterns": [
            r"\bwarfarin\b", r"\bcoumadin\b", r"\bheparin\b", r"\benoxaparin\b",
            r"\bapixaban\b", r"\beliquis\b", r"\brivaroxaban\b", r"\bxarelto\b",
            r"\bdabigatran\b", r"\banticoagulation\b", r"\banticoagulated\b",
        ],
        "neg_patterns": [
            r"\bnot on anticoagulation\b", r"\bno anticoagulation\b",
            r"\bheld warfarin\b", r"\bheld coumadin\b",
        ],
    },
    {
        "concept_id": "insulin_therapy",
        "concept_group": "medication",
        "proposition": "The patient is currently receiving insulin therapy.",
        "pos_patterns": [
            r"\binsulin\b", r"\bglargine\b", r"\blispro\b", r"\baspart\b",
            r"\bhumalog\b", r"\blantus\b", r"\bnovolog\b",
        ],
        "neg_patterns": [
            r"\bno insulin\b", r"\bnot on insulin\b", r"\bwithout insulin\b",
            r"\bhold insulin\b", r"\bheld insulin\b",
        ],
    },
    {
        "concept_id": "renal_dysfunction",
        "concept_group": "diagnosis",
        "proposition": "The patient currently has renal dysfunction.",
        "pos_patterns": [
            r"\brenal dysfunction\b", r"\bak[iI]\b", r"\bacute kidney injury\b",
            r"\bckd\b", r"\besrd\b", r"\bkidney failure\b",
            r"\bon dialysis\b", r"\bhemodialysis\b", r"\bdialysis\b",
            r"\belevated creatinine\b", r"\bcreatinine .* elevated\b",
        ],
        "neg_patterns": [
            r"\bno renal dysfunction\b", r"\bnormal creatinine\b",
            r"\bcreatinine normal\b", r"\bwithout renal dysfunction\b",
        ],
    },
    {
        "concept_id": "liver_dysfunction",
        "concept_group": "diagnosis",
        "proposition": "The patient currently has liver dysfunction.",
        "pos_patterns": [
            r"\bliver dysfunction\b", r"\bhepatic dysfunction\b",
            r"\bcirrhosis\b", r"\bhepatic failure\b", r"\bliver failure\b",
            r"\belevated transaminase\b", r"\btransaminitis\b",
            r"\belevated lfts\b", r"\belevated alt\b", r"\belevated ast\b",
        ],
        "neg_patterns": [
            r"\bno liver dysfunction\b", r"\bnormal lfts\b", r"\blfts normal\b",
            r"\bwithout liver dysfunction\b",
        ],
    },
    {
        "concept_id": "penicillin_allergy",
        "concept_group": "allergy",
        "proposition": "The patient has a penicillin allergy.",
        "pos_patterns": [
            r"\bpenicillins?\b", r"\bpcn\b", r"\ballergy.*penicillin\b",
            r"\bpenicillin allergy\b",
        ],
        "neg_patterns": [
            r"\bno known allergies\b", r"\bnka\b", r"\bno allergy\b",
            r"\btolerates penicillin\b", r"\bno drug allergies\b",
        ],
    },
    {
        "concept_id": "contrast_allergy",
        "concept_group": "allergy",
        "proposition": "The patient has a contrast allergy.",
        "pos_patterns": [
            r"\bcontrast allergy\b", r"\biodine allergy\b",
            r"\ballergy.*contrast\b", r"\bcontrast reaction\b",
        ],
        "neg_patterns": [
            r"\bno contrast allergy\b", r"\btolerates contrast\b",
        ],
    },
]

ALLOWED_LABELS = (
    "observed_positive,observed_negative,uncertain,historical_only,"
    "admission_only,temporal_conflict,unmentioned"
)


def load_audit_notes():
    human = [
        json.loads(line)
        for line in (ROOT / "data/real_clinical_sru.jsonl").open(encoding="utf-8")
    ]
    by_audit = {r["audit_id"]: r for r in human}
    return by_audit


def lexical_screen(text: str, spec: dict) -> tuple[str, str]:
    lower = text.lower()
    neg_hits = []
    pos_hits = []
    for pat in spec["neg_patterns"]:
        for m in re.finditer(pat, lower):
            neg_hits.append(m.group(0))
    for pat in spec["pos_patterns"]:
        for m in re.finditer(pat, lower):
            pos_hits.append(m.group(0))
    if neg_hits:
        return "lexical_negative", "; ".join(sorted(set(neg_hits))[:3])
    if pos_hits:
        return "lexical_positive", "; ".join(sorted(set(pos_hits))[:3])
    return "lexical_unmentioned", ""


def excerpt(text: str, *, max_chars=400) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def chief_complaint(text: str) -> str:
    m = re.search(
        r"Chief Complaint:\s*(.+?)(?:\n\n|\nMajor|\nHistory of Present Illness:)",
        text,
        flags=re.I | re.S,
    )
    if m:
        return excerpt(m.group(1).strip(), max_chars=200)
    return excerpt(text, max_chars=200)


def score_candidate(row: dict) -> float:
    score = 0.0
    if row["lexical_screen"] == "lexical_unmentioned":
        score += 5.0
    elif row["lexical_screen"] == "lexical_negative":
        score += 1.0  # explicit neg; not target but useful negative controls
    else:
        score -= 3.0
    if row["existing_lab_observation_status"] == "unmentioned":
        score += 2.0
    if row["existing_lab_y3"] == "Unknown" and row["existing_lab_state"] == "unmentioned":
        score += 3.0
    # prefer moderate-length notes
    n = int(row["note_chars"])
    if 4000 <= n <= 20000:
        score += 1.0
    return score


def select_candidates(candidates: list[dict], target: int = 50) -> list[dict]:
    """Pick ~target rows with concept-group diversity and one row per note first."""
    by_score = sorted(candidates, key=lambda r: (-score_candidate(r), r["audit_id"]))
    picked = []
    used_audit = set()
    group_counts = Counter()

    # Round 1: one high-quality lexical_unmentioned per admission, diversify groups.
    for row in by_score:
        if row["lexical_screen"] != "lexical_unmentioned":
            continue
        if row["audit_id"] in used_audit:
            continue
        g = row["concept_group"]
        if group_counts[g] >= max(8, target // len({c["concept_group"] for c in EXPANSION_NONLAB})):
            continue
        picked.append(row)
        used_audit.add(row["audit_id"])
        group_counts[g] += 1
        if len(picked) >= target:
            return picked

    # Round 2: fill remaining slots with best unmentioned, allow second concept/note.
    for row in by_score:
        if row in picked:
            continue
        if row["lexical_screen"] != "lexical_unmentioned":
            continue
        picked.append(row)
        if len(picked) >= target:
            return picked

    # Round 3: if still short, include lexical_negative as explicit-refuted controls.
    for row in by_score:
        if row in picked:
            continue
        if row["lexical_screen"] == "lexical_negative":
            picked.append(row)
        if len(picked) >= target:
            break
    return picked[:target]


def main():
    by_audit = load_audit_notes()
    candidates = []
    for audit_id, base in sorted(by_audit.items()):
        text = base["profile"]
        for spec in EXPANSION_NONLAB:
            screen, hits = lexical_screen(text, spec)
            candidates.append({
                "candidate_id": f"expansion:{audit_id}:{spec['concept_id']}",
                "audit_id": audit_id,
                "subject_id": base["subject_id"],
                "hadm_id": base["hadm_id"],
                "concept_id": spec["concept_id"],
                "concept_group": spec["concept_group"],
                "proposition": spec["proposition"],
                "existing_lab_concept": base["concept"],
                "existing_lab_observation_status": base.get("observation_status", ""),
                "existing_lab_y3": base["y3"],
                "existing_lab_state": base.get("state", ""),
                "existing_lab_human1": base.get("human1", ""),
                "existing_lab_human2": base.get("human2", ""),
                "lexical_screen": screen,
                "lexical_hits": hits,
                "note_chars": len(text),
                "chief_complaint_excerpt": chief_complaint(text),
                "note_path_hint": f"data/real_clinical_sru.jsonl (audit_id={audit_id})",
                "_text_lower": text.lower(),
            })

    selected = select_candidates(candidates, target=50)
    for i, row in enumerate(selected, start=1):
        row["priority_rank"] = i
        row["target_strict_unmentioned"] = (
            "YES" if row["lexical_screen"] == "lexical_unmentioned" else "REVIEW"
        )
        row["lexical_review_flag"] = (
            "check_allergies_header"
            if row["concept_id"] == "penicillin_allergy"
            and re.search(r"allergies?:[^\n]{0,80}penicillin", row["_text_lower"], re.I)
            else ""
        )
        del row["_text_lower"]

    out_primary = ROOT / "data" / "real_clinical_sru_expansion_candidates.csv"
    out_pool = ROOT / "data" / "real_clinical_sru_expansion_pool.csv"

    fieldnames = [
        "priority_rank",
        "candidate_id",
        "target_strict_unmentioned",
        "audit_id",
        "subject_id",
        "hadm_id",
        "concept_id",
        "concept_group",
        "proposition",
        "lexical_screen",
        "lexical_hits",
        "lexical_review_flag",
        "existing_lab_concept",
        "existing_lab_y3",
        "existing_lab_observation_status",
        "chief_complaint_excerpt",
        "note_chars",
        "human1_label",
        "human1_evidence_span",
        "human2_label",
        "human2_evidence_span",
        "consensus_label",
        "maps_to_y3",
        "maps_to_state",
        "include_in_human_gold",
        "reviewer_notes",
        "note_path_hint",
        "allowed_labels",
    ]

    def write_csv(path, rows):
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                row.pop("_text_lower", None)
                out = {
                    **row,
                    "human1_label": "",
                    "human1_evidence_span": "",
                    "human2_label": "",
                    "human2_evidence_span": "",
                    "consensus_label": "",
                    "maps_to_y3": "",
                    "maps_to_state": "",
                    "include_in_human_gold": "",
                    "reviewer_notes": "",
                    "allowed_labels": ALLOWED_LABELS,
                }
                w.writerow(out)

    write_csv(out_primary, selected)

    # Full pre-screened pool (lexical_unmentioned only) for backup picks.
    pool = [
        r for r in candidates
        if r["lexical_screen"] == "lexical_unmentioned"
        and r["candidate_id"] not in {x["candidate_id"] for x in selected}
    ]
    pool = sorted(pool, key=lambda r: (-score_candidate(r), r["audit_id"]))
    write_csv(out_pool, pool[:100])

    stats = {
        "audit_notes": len(by_audit),
        "nonlab_concepts": len(EXPANSION_NONLAB),
        "total_pairs_screened": len(candidates),
        "lexical_unmentioned_pairs": sum(
            1 for r in candidates if r["lexical_screen"] == "lexical_unmentioned"
        ),
        "selected_n": len(selected),
        "selected_by_group": dict(Counter(r["concept_group"] for r in selected)),
        "selected_by_concept": dict(Counter(r["concept_id"] for r in selected)),
        "selected_lexical_unmentioned": sum(
            1 for r in selected if r["lexical_screen"] == "lexical_unmentioned"
        ),
        "current_human_gold_strict_unmentioned": 40,
        "projected_if_all_selected_unmentioned": 40 + sum(
            1 for r in selected if r["lexical_screen"] == "lexical_unmentioned"
        ),
    }
    stats_path = ROOT / "outputs" / "real_clinical_sru_expansion_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))
    print(f"wrote {out_primary} ({len(selected)} prioritized rows)")
    print(f"wrote {out_pool} ({min(100, len(pool))} backup lexical-unmentioned rows)")


if __name__ == "__main__":
    main()
