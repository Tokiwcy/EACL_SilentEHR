#!/usr/bin/env python3
"""Pilot: real MIMIC discharge backgrounds + controlled S/R/U insertion.

Generates a local-only JSONL for replication experiments. Raw MIMIC text must
NOT be redistributed; this file is for internal evaluation only.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from silentehr import (  # noqa: E402
    CONCEPTS,
    FAMILY_TEMPLATES,
    default_mimic_struct_path,
    load_rows,
)

PILOT_CONCEPTS = ["chest_pain", "nausea", "penicillin_allergy"]
N_BACKGROUNDS = 50
SEED = 0


def pick_template(concept, state):
    group = next(c["group"] for c in CONCEPTS if c["id"] == concept)
    family = {
        "symptom": "symptom", "allergy": "allergy", "medication": "medication",
        "disease": "disease", "safety": "disease",
    }.get(group, "symptom")
    templates = FAMILY_TEMPLATES[family][state]
    name = next(c["name"] for c in CONCEPTS if c["id"] == concept)
    tpl = templates[0]
    sent = tpl.format(c=name, C=name[0].upper() + name[1:])
    return sent


def main():
    from silentehr import background_is_clean
    struct_path = default_mimic_struct_path()
    rows = load_rows(struct_path)
    by_hadm = {}
    for r in rows:
        if r.get("text"):
            by_hadm.setdefault(r["hadm_id"], r["text"])
    hadm_ids = sorted(by_hadm)
    rng = random.Random(SEED)
    rng.shuffle(hadm_ids)

    out_rows = []
    rid = 0
    picked = 0
    for hadm in hadm_ids:
        if picked >= N_BACKGROUNDS:
            break
        base = by_hadm[hadm]
        concept = PILOT_CONCEPTS[picked % len(PILOT_CONCEPTS)]
        if not background_is_clean(base, concept):
            continue
        prop = next(c["prop"] for c in CONCEPTS if c["id"] == concept)
        for state, y3 in (
            ("unmentioned", "Unknown"),
            ("supported", "Supported"),
            ("refuted", "Refuted"),
        ):
            if state == "unmentioned":
                profile = base
            else:
                profile = base.rstrip() + " " + pick_template(concept, state)
            out_rows.append({
                "id": f"mimic_pilot_{rid}",
                "hadm_id": hadm,
                "concept": concept,
                "concept_id": concept,
                "proposition": prop,
                "state": state,
                "y3": y3,
                "profile": profile,
                "background_split": "train" if picked % 5 else "test",
                "concept_split": "train",
                "template_split": "na",
                "iid_split": "train",
            })
            rid += 1
        picked += 1

    out = ROOT / "data" / "mimic_real_background_pilot.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out_rows)} rows ({picked} backgrounds) to {out}")
    print("NOTE: do not publish raw profiles; run locally with silentehr.py run --data-path ...")


if __name__ == "__main__":
    main()
