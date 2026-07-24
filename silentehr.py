#!/usr/bin/env python3
"""When Silence Becomes Negative: frozen-embedding closed-world extrapolation.

  python silentehr.py all                 # make + outline + run (hash smoke)
  python silentehr.py run --model qwen
  python silentehr.py mimic --mimic-dir D:/mimic-iv-3.1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import math
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "pilot.jsonl"
FULL_DATA = ROOT / "data" / "controlled_full.jsonl"
ARTIFACT_LENGTHMATCHED = ROOT / "data" / "controlled_lengthmatched.jsonl"
ARTIFACT_RANDPOS = ROOT / "data" / "controlled_randpos.jsonl"
ARTIFACT_INDIRECT = ROOT / "data" / "controlled_indirect.jsonl"
ARTIFACT_TIERS = {
    "lengthmatched": ARTIFACT_LENGTHMATCHED,
    "randpos": ARTIFACT_RANDPOS,
    "indirect": ARTIFACT_INDIRECT,
}
OUT = ROOT / "outputs"
RNG = random.Random(42)

# ---------------------------------------------------------------------------
# Concepts & templates
# ---------------------------------------------------------------------------
CONCEPTS = [
    {"id": "chest_pain", "name": "chest pain", "group": "symptom",
     "prop": "The patient currently has chest pain."},
    {"id": "fever", "name": "fever", "group": "symptom",
     "prop": "The patient currently has fever."},
    {"id": "penicillin_allergy", "name": "penicillin allergy", "group": "allergy",
     "prop": "The patient has a penicillin allergy."},
    {"id": "active_bleeding", "name": "active bleeding", "group": "safety",
     "prop": "The patient currently has active bleeding."},
    {"id": "anticoagulation", "name": "anticoagulation", "group": "medication",
     "prop": "The patient is currently on anticoagulation."},
    {"id": "renal_dysfunction", "name": "renal dysfunction", "group": "disease",
     "prop": "The patient currently has renal dysfunction."},
    {"id": "nausea", "name": "nausea", "group": "symptom",
     "prop": "The patient currently has nausea."},
    {"id": "dyspnea", "name": "dyspnea", "group": "symptom",
     "prop": "The patient currently has dyspnea.", "full_holdout": True},
    {"id": "contrast_allergy", "name": "contrast allergy", "group": "allergy",
     "prop": "The patient has a contrast allergy.", "full_holdout": True},
    {"id": "sulfonamide_allergy", "name": "sulfonamide allergy", "group": "allergy",
     "prop": "The patient has a sulfonamide allergy."},
    {"id": "latex_allergy", "name": "latex allergy", "group": "allergy",
     "prop": "The patient has a latex allergy."},
    {"id": "insulin_therapy", "name": "insulin therapy", "group": "medication",
     "prop": "The patient is currently receiving insulin therapy."},
    {"id": "opioid_therapy", "name": "opioid therapy", "group": "medication",
     "prop": "The patient is currently receiving opioid therapy.",
     "full_holdout": True},
    {"id": "antiplatelet_therapy", "name": "antiplatelet therapy", "group": "medication",
     "prop": "The patient is currently receiving antiplatelet therapy."},
    {"id": "cognitive_impairment", "name": "cognitive impairment", "group": "disease",
     "prop": "The patient currently has cognitive impairment."},
    {"id": "liver_dysfunction", "name": "liver dysfunction", "group": "disease",
     "prop": "The patient currently has liver dysfunction.", "full_holdout": True},
]

# 6 states for pilot (Supported/Refuted + 4 unknowns). Fine states for analysis.
STATES = [
    "supported", "refuted", "unmentioned", "not_assessed", "uncertain", "family",
]
KNOWN = {"supported", "refuted"}
UNKNOWN = ("unmentioned", "not_assessed", "uncertain", "family")

FAMILY_TEMPLATES = {
    "symptom": {
        "supported": ["The patient reports {c}.", "Current symptoms include {c}.",
                      "The patient endorses {c} today.", "Active {c} is documented."],
        "refuted": ["The patient denies {c}.", "No current {c} was reported.",
                    "Review of systems was negative for {c}.", "{C} is explicitly absent."],
        "uncertain": ["It is unclear whether the patient has {c}.",
                      "Possible {c} was not confirmed.", "{C} status remains uncertain.",
                      "Findings regarding {c} are inconclusive."],
        "not_assessed": ["{C} was not assessed.", "No assessment of {c} was performed.",
                         "The clinician did not evaluate {c}.", "{C} screening was deferred."],
        "family": ["The patient's father had {c}.", "Family history includes {c}.",
                   "A sibling has a history of {c}.", "The patient's mother experienced {c}."],
    },
    "allergy": {
        "supported": ["A {c} is documented.", "The allergy list includes {c}.",
                      "The patient reports a {c}.", "The record confirms {c}."],
        "refuted": ["The record states no known {c}.", "The patient denies a {c}.",
                    "The allergy list is negative for {c}.", "{C} is explicitly absent."],
        "uncertain": ["The reported {c} could not be confirmed.",
                      "It is unclear whether the patient has a {c}.",
                      "{C} status remains uncertain.", "The possible {c} requires clarification."],
        "not_assessed": ["{C} status was not documented.", "The {c} history was not assessed.",
                         "No evaluation of {c} was recorded.", "{C} reconciliation was deferred."],
        "family": ["A first-degree relative has a {c}.", "Family history includes {c}.",
                   "The patient's mother reports a {c}.", "A sibling has a {c}."],
    },
    "medication": {
        "supported": ["The patient is currently receiving {c}.", "Current therapy includes {c}.",
                      "{C} is active on the medication list.", "The patient remains on {c}."],
        "refuted": ["The patient is not receiving {c}.", "{C} is not on the current medication list.",
                    "The record confirms no current {c}.", "{C} was explicitly withheld."],
        "uncertain": ["Current {c} status is unclear.", "It is uncertain whether {c} is active.",
                      "Medication reconciliation did not confirm {c}.",
                      "Possible current {c} use requires clarification."],
        "not_assessed": ["Current {c} status was not documented.",
                         "{C} use was not assessed.", "Medication reconciliation omitted {c}.",
                         "No assessment of current {c} was recorded."],
    },
    "disease": {
        "supported": ["Current findings indicate {c}.", "Active {c} is documented.",
                      "The assessment confirms {c}.", "The patient currently has {c}."],
        "refuted": ["Current findings show no {c}.", "{C} is explicitly absent.",
                    "The assessment is negative for {c}.", "Current status excludes {c}."],
        "uncertain": ["Possible {c} was not confirmed.", "{C} status remains uncertain.",
                      "Current findings are inconclusive for {c}.",
                      "It is unclear whether {c} is present."],
        "not_assessed": ["{C} was not assessed.", "No current evaluation of {c} was recorded.",
                         "Assessment of {c} was deferred.", "Current {c} status was not documented."],
        "family": ["A first-degree relative had {c}.", "Family history includes {c}.",
                   "The patient's father had {c}.", "A sibling has a history of {c}."],
    },
}

# Safety concepts use disease-state grammar. Family medication exposure is invalid.
GROUP_FAMILY = {"safety": "disease"}
VALID_STATES = {
    "symptom": set(STATES), "allergy": set(STATES), "disease": set(STATES),
    "safety": set(STATES), "medication": set(STATES) - {"family"},
}

CONTAMINATION_TERMS = {
    "chest_pain": [r"\bchest pain\b", r"\bchest discomfort\b"],
    "fever": [r"\bfever\b", r"\bfebrile\b", r"\bafebrile\b"],
    "penicillin_allergy": [r"\bpenicillin\b", r"\ballerg(?:y|ic)\b"],
    "active_bleeding": [r"\bbleed(?:ing)?\b", r"\bhemorrhag"],
    "anticoagulation": [r"\banticoag", r"\bwarfarin\b", r"\bheparin\b", r"\bapixaban\b"],
    "renal_dysfunction": [r"\brenal\b", r"\bkidney\b", r"\bcreatinine\b", r"\begfr\b", r"\baki\b"],
    "nausea": [r"\bnausea\b", r"\bnauseated\b"],
    "dyspnea": [r"\bdyspnea\b", r"\bshortness of breath\b"],
    "contrast_allergy": [r"\bcontrast\b"],
    "sulfonamide_allergy": [r"\bsulfa", r"\bsulfonamide\b"],
    "latex_allergy": [r"\blatex\b"],
    "insulin_therapy": [r"\binsulin\b"],
    "opioid_therapy": [r"\bopioid\b", r"\bmorphine\b", r"\boxycodone\b"],
    "antiplatelet_therapy": [r"\bantiplatelet\b", r"\baspirin\b", r"\bclopidogrel\b"],
    "cognitive_impairment": [r"\bcognitive\b", r"\bdementia\b", r"\bconfusion\b"],
    "liver_dysfunction": [r"\bliver\b", r"\bhepatic\b", r"\btransaminase"],
}

def concept_anchors(name: str, group: str) -> tuple[list[str], list[str]]:
    family = GROUP_FAMILY.get(group, group)
    if family == "symptom":
        return (
            [f"The patient currently reports {name}.", f"Current {name} is present.",
             f"The patient endorses {name}."],
            [f"The patient currently denies {name}.", f"Current {name} is absent.",
             f"The patient reports no {name}."],
        )
    if family == "allergy":
        return (
            [f"The patient has a documented {name}.", f"The allergy list includes {name}.",
             f"The record confirms {name}."],
            [f"The patient has no known {name}.", f"The allergy list excludes {name}.",
             f"The record explicitly denies {name}."],
        )
    if family == "medication":
        return (
            [f"The patient is currently receiving {name}.", f"Current therapy includes {name}.",
             f"{name.capitalize()} is active."],
            [f"The patient is not receiving {name}.", f"Current therapy excludes {name}.",
             f"{name.capitalize()} is not active."],
        )
    return (
        [f"Current findings indicate {name}.", f"Active {name} is present.",
         f"The assessment confirms {name}."],
        [f"Current findings show no {name}.", f"Active {name} is absent.",
         f"The assessment excludes {name}."],
    )

BACKGROUNDS = [
    # 20 short synthetic profiles (~150–250 words when combined with state)
    "The patient is a 67-year-old woman admitted with fatigue and poor oral intake. "
    "Her medical history includes hypertension and hyperlipidemia. She was hemodynamically "
    "stable on admission. Physical examination was otherwise unremarkable. Labs showed mild "
    "hyponatremia. She lives with her daughter and is independent in most ADLs. "
    "Home medications include amlodipine and atorvastatin. Nursing notes describe improved "
    "appetite after IV fluids. Plan includes further workup for failure to thrive.",

    "A 54-year-old man presented after several days of progressive weakness. Past history "
    "includes type 2 diabetes and obesity. Vital signs were within normal limits. "
    "Abdominal examination was soft and nontender. He works as a warehouse supervisor. "
    "He reports good sleep and no recent travel. Glucose was moderately elevated. "
    "He was started on a diabetic diet and monitored overnight. Discharge planning "
    "involves outpatient endocrine follow-up.",

    "This is a 72-year-old man with COPD and prior stroke who was admitted for confusion. "
    "He was oriented to person and place after hydration. CT head showed chronic changes "
    "only. He resides in an assisted living facility. Medications include inhalers and "
    "aspirin. Speech therapy evaluated swallowing and cleared a regular diet. Social work "
    "was consulted for disposition. The hospital course has been otherwise uneventful.",

    "A 41-year-old woman was evaluated for recurrent headaches. Neurologic examination was "
    "nonfocal. MRI was deferred pending outpatient follow-up. She has a history of migraine "
    "and anxiety. She denies recent head trauma. She works remotely and reports high stress. "
    "She was given supportive care and counseling on trigger avoidance. No acute intervention "
    "was required during this encounter.",

    "The patient is an 80-year-old man admitted from clinic with weight loss. Colonoscopy "
    "was previously incomplete. He has osteoarthritis and mild hearing loss. He ambulates "
    "with a cane. Nutrition consult recommended protein supplementation. He is widowed and "
    "has two adult children nearby. Laboratory studies were largely unremarkable aside from "
    "mild anemia of unclear chronicity. Further outpatient imaging was arranged.",

    "A 59-year-old woman presented with lower extremity edema. Echo last year showed normal "
    "EF. She has chronic venous insufficiency and takes hydrochlorothiazide. Lungs were clear "
    "bilaterally. Compression stockings were advised. She is a retired teacher. She reports "
    "adherence to a low-salt diet with occasional lapses. Follow-up with vascular medicine "
    "was recommended.",

    "This 33-year-old man was admitted after a motor vehicle collision with soft-tissue "
    "injuries only. Imaging ruled out fracture. He was observed overnight for pain control. "
    "He has no chronic medical conditions. Tetanus status was updated. He was ambulatory "
    "prior to discharge. Case management arranged outpatient primary care. He lives alone "
    "and has reliable transportation.",

    "A 68-year-old woman with hypothyroidism presented with dizziness. Orthostatic vitals "
    "were borderline. TSH was within target. She takes levothyroxine daily. She reported "
    "adequate fluid intake. Physical therapy evaluated gait and recommended a walker "
    "trial. Telemetry was unremarkable. She was discharged with fall precautions education.",

    "The patient is a 47-year-old man with ulcerative colitis in clinical remission, "
    "admitted for dehydration after gastroenteritis. IV fluids improved his status. "
    "Abdominal CT was deferred. He continues mesalamine. He works in IT and is currently "
    "working from home. Stool studies were sent. He was advanced to a regular diet without "
    "incident.",

    "An 75-year-old woman was hospitalized for community contact isolation after a viral "
    "exposure at her nursing facility; she remained asymptomatic. Vaccination records were "
    "reviewed. She has dementia and hypertension. She required assistance with meals. "
    "No new focal neurologic deficits were observed. She was cleared to return to the "
    "facility after observation.",

    "A 62-year-old man presented with back pain after lifting. Neurologic exam showed no "
    "deficit. Plain films were negative for fracture. He has hyperlipidemia on a statin. "
    "He was treated with NSAIDs and muscle relaxants. Physical therapy referral was placed. "
    "He works as a mechanic. Return precautions were discussed.",

    "This 29-year-old woman was seen for palpitations. ECG showed sinus tachycardia. "
    "TSH and CBC were pending at the time of note. She drinks coffee heavily. She has no "
    "known structural heart disease. She was advised to reduce caffeine and follow up with "
    "cardiology if symptoms persist. She is otherwise healthy.",

    "A 70-year-old man with prior prostate cancer in remission was admitted for failure to "
    "thrive workup. He had poor appetite for three weeks. He takes tamsulosin. Abdominal "
    "exam was benign. Oncology was notified for continuity. He lives with his spouse. "
    "Calorie counts were initiated. No acute surgical issues were identified.",

    "The patient is a 55-year-old woman with rheumatoid arthritis on methotrexate, admitted "
    "for evaluation of elevated liver enzymes found in clinic. She was asymptomatic. "
    "Hepatitis serologies were pending. Joint exam was stable. Alcohol history was remote. "
    "Methotrexate was held pending workup. Hepatology follow-up was arranged.",

    "A 64-year-old man presented with cellulitis of the left leg. He was started on IV "
    "antibiotics with clinical improvement. He has peripheral vascular disease. Wound care "
    "was consulted. He is a former smoker who quit 10 years ago. Blood cultures remained "
    "negative at 24 hours. He was transitioned to oral antibiotics.",

    "An 38-year-old woman was evaluated for abdominal pain; surgical abdomen was excluded. "
    "Ultrasound was unremarkable. She has IBS and takes fiber supplements. She was observed "
    "and improved with supportive care. She works as a nurse. She was discharged with "
    "return precautions and GI clinic follow-up.",

    "This 81-year-old man was admitted after a ground-level fall without fracture. He uses "
    "a hearing aid. Cognitive screening was at baseline per family. PT/OT recommended short "
    "rehab. He has BPH and glaucoma. Medications were reconciled. He was hemodynamically "
    "stable throughout the stay.",

    "A 50-year-old woman with asthma presented with cough; wheezing responded to nebulizers. "
    "Chest x-ray was clear. Peak flows improved. She continues inhaled corticosteroids. "
    "She has seasonal allergies. She was discharged on a short steroid burst with PCP "
    "follow-up in one week.",

    "The patient is a 73-year-old woman admitted for elective hernia repair preoperative "
    "optimization. She has well-controlled hypertension. Anesthesia evaluation was completed. "
    "She ambulates independently. Labs were acceptable for surgery. She lives with her "
    "husband. The procedure was scheduled for the following morning.",

    "A 46-year-old man was hospitalized for IV iron after clinic labs showed iron deficiency "
    "without acute bleeding source on prior outpatient workup. He tolerated infusion well. "
    "He has GERD on PPI. He reports fatigue improving. Outpatient GI follow-up remains "
    "scheduled. No new acute complaints during admission.",
]


def expanded_backgrounds(target_n: int = 80) -> list[str]:
    """Create varied, target-neutral profiles for the scaled controlled audit."""
    ages = [34, 41, 48, 55, 62, 69, 76, 83]
    presentations = [
        "progressive fatigue and reduced oral intake",
        "an elective functional assessment",
        "recent weight loss and general deconditioning",
        "a scheduled diagnostic evaluation",
        "difficulty managing activities of daily living",
    ]
    histories = [
        "hypertension and hyperlipidemia",
        "osteoarthritis and gastroesophageal reflux",
        "well-controlled hypothyroidism and chronic back stiffness",
        "benign prostatic disease and glaucoma",
    ]
    plans = [
        "supportive care and outpatient follow-up",
        "physical therapy assessment and nutrition counseling",
        "medication reconciliation and routine monitoring",
        "case management review and primary-care follow-up",
    ]
    generated = []
    i = 0
    while len(BACKGROUNDS) + len(generated) < target_n:
        age = ages[i % len(ages)]
        presentation = presentations[(i // len(ages)) % len(presentations)]
        history = histories[(i // (len(ages) * len(presentations))) % len(histories)]
        plan = plans[(i * 3) % len(plans)]
        sex = "woman" if i % 2 else "man"
        generated.append(
            f"The patient is a {age}-year-old {sex} admitted for {presentation}. "
            f"Past medical history includes {history}. Vital signs remained stable "
            "during observation. The examination showed no new focal findings. "
            "Routine studies were reviewed by the treating team. The patient lives "
            "with family and was independent before this encounter. The care plan "
            f"includes {plan}. Disposition planning was completed without incident."
        )
        i += 1
    return BACKGROUNDS + generated


def _fmt(tmpl: str, concept: str) -> str:
    return tmpl.format(c=concept, C=concept[:1].upper() + concept[1:])


def stable_index(text: str, dim: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % dim


def background_is_clean(background: str, concept_id: str) -> bool:
    return not any(re.search(p, background, flags=re.I)
                   for p in CONTAMINATION_TERMS[concept_id])


def _split_assign(n: int, seed: int = 0) -> list[str]:
    """Background-held-out style: 70/15/15 over backgrounds."""
    idx = list(range(n))
    RNG_local = random.Random(seed)
    RNG_local.shuffle(idx)
    n_tr, n_va = int(0.7 * n), int(0.15 * n)
    tags = [""] * n
    for i in idx[:n_tr]:
        tags[i] = "train"
    for i in idx[n_tr:n_tr + n_va]:
        tags[i] = "val"
    for i in idx[n_tr + n_va:]:
        tags[i] = "test"
    return tags


def make_pilot(n_para: int = 3, out: Path | None = None,
               full: bool = False) -> Path:
    """Build clean matched variants with independent generalization axes."""
    out = out or (FULL_DATA if full else DATA)
    backgrounds = expanded_backgrounds(80) if full else BACKGROUNDS
    concepts = CONCEPTS if full else CONCEPTS[:6]
    states = (["supported", "refuted", "unmentioned",
               "not_assessed", "uncertain"] if full else STATES)
    bg_split = _split_assign(len(backgrounds), seed=0)
    rows = []
    contamination = []
    rid = 0
    for ci, concept in enumerate(concepts):
        concept_split = (
            "test" if concept.get("full_holdout") else "train"
        ) if full else ("test" if ci >= 4 else "train")
        for bi, bg in enumerate(backgrounds):
            if not background_is_clean(bg, concept["id"]):
                contamination.append({
                    "background_id": bi, "concept_id": concept["id"],
                    "background": bg,
                })
                continue
            iid_key = stable_index(f"{bi}:{concept['id']}", 10)
            iid_split = "train" if iid_key < 7 else ("val" if iid_key < 8 else "test")
            family = GROUP_FAMILY.get(concept["group"], concept["group"])
            for state in states:
                if state not in VALID_STATES[concept["group"]]:
                    continue
                if state == "unmentioned":
                    # Genuine non-observation has one independent profile, not
                    # three duplicate pseudo-paraphrases.
                    variants = [("", "none", "na")]
                else:
                    tmpls = FAMILY_TEMPLATES[family][state]
                    train_pool, held_pool = tmpls[:-1], tmpls[-1:]
                    train_n = 1 if full else n_para
                    variants = [
                        (t, f"train_{p}", "train")
                        for p, t in enumerate(train_pool[:train_n])
                    ]
                    # One separate held-out template; never duplicate it.
                    variants += [(held_pool[0], "held_0", "test")]
                for p, (tmpl, tid, template_split) in enumerate(variants):
                    if state == "unmentioned":
                        profile = bg
                        mention = 0
                        assertion = "none"
                    else:
                        sent = _fmt(tmpl, concept["name"])
                        profile = bg + " " + sent
                        mention = 1
                        assertion = {
                            "supported": "present",
                            "refuted": "absent",
                            "uncertain": "uncertain",
                            "not_assessed": "other",
                            "family": "other",
                        }[state]
                    # coarse label for main task
                    if state == "supported":
                        y3 = "Supported"
                    elif state == "refuted":
                        y3 = "Refuted"
                    else:
                        y3 = "Unknown"
                    # eligibility toy label for exclusion criterion = concept present
                    if state == "supported":
                        elig = "ineligible"
                    elif state == "refuted":
                        elig = "eligible"
                    else:
                        elig = "manual_review"
                    rows.append({
                        "id": f"p{rid:05d}",
                        "concept_id": concept["id"],
                        "concept": concept["name"],
                        "proposition": concept["prop"],
                        "state": state,
                        "y3": y3,
                        "profile": profile,
                        "background_id": bi,
                        "template_id": tid,
                        "paraphrase": p,
                        "mention": mention,
                        "assertion": assertion,
                        "eligibility": elig,
                        "background_split": bg_split[bi],
                        "template_split": template_split,
                        "concept_split": concept_split,
                        "iid_split": iid_split,
                        "group": concept["group"],
                    })
                    rid += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    audit = out.parent / "background_contamination.json"
    audit.write_text(json.dumps(contamination, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} pairs -> {out}")
    print(f"excluded {len(contamination)} contaminated background-concept pairs -> {audit}")
    return out


NEUTRAL_FILLERS = [
    "Discharge planning and mobility needs were reviewed with the patient.",
    "Medication reconciliation was completed before discharge.",
    "Follow-up appointments were arranged with the primary care provider.",
    "The patient was counseled on return precautions and activity guidance.",
    "Case management reviewed home services and transportation needs.",
    "Nutrition counseling and oral intake were discussed with the patient.",
    "Physical therapy evaluated ambulation and recommended outpatient follow-up.",
    "Social work assessed caregiver support and community resources.",
    "The care team reviewed pending outpatient laboratory monitoring.",
    "Patient education materials were provided regarding the treatment plan.",
]

INDIRECT_TEMPLATES = {
    "chest_pain": {
        "supported": [
            "Troponin I was elevated at 0.8 ng/mL.",
            "Serial cardiac enzymes remained above the reference range.",
            "ECG changes were consistent with active ischemia.",
            "Cardiac biomarkers were elevated on repeat testing.",
        ],
        "refuted": [
            "Serial troponins remained within normal limits.",
            "ECG showed no ischemic changes.",
            "Cardiac enzymes were negative on repeat testing.",
            "No dynamic ECG changes were observed.",
        ],
    },
    "fever": {
        "supported": [
            "Temperature remained 38.4°C.",
            "The patient was febrile to 38.7°C overnight.",
            "Recorded temperature was 38.2°C.",
            "Temperature readings remained above 38.0°C.",
        ],
        "refuted": [
            "The patient remained afebrile.",
            "Temperature was 36.8°C on discharge.",
            "No febrile readings were recorded.",
            "Temperature normalized to 37.0°C.",
        ],
    },
    "penicillin_allergy": {
        "supported": [
            "Penicillin was listed on the allergy band.",
            "The allergy record documents a penicillin reaction.",
            "Penicillin is marked as an active allergy.",
            "The patient carries a documented penicillin allergy.",
        ],
        "refuted": [
            "No drug allergies were documented.",
            "The allergy list was negative for penicillin.",
            "Penicillin was not listed among active allergies.",
            "Allergy reconciliation showed no penicillin allergy.",
        ],
    },
    "active_bleeding": {
        "supported": [
            "Hemoglobin dropped from 14.1 to 9.2 g/dL.",
            "Guaiac-positive stools were noted.",
            "There was ongoing hemoptysis during admission.",
            "Repeat hemoglobin was 8.8 g/dL.",
        ],
        "refuted": [
            "No overt bleeding was observed.",
            "Hemoglobin remained stable at 13.4 g/dL.",
            "Stool guaiac testing was negative.",
            "There was no hemodynamic evidence of active blood loss.",
        ],
    },
    "anticoagulation": {
        "supported": [
            "Apixaban was continued at discharge.",
            "Warfarin remained on the active medication list.",
            "Therapeutic anticoagulation was maintained.",
            "Rivaroxaban was prescribed at discharge.",
        ],
        "refuted": [
            "Apixaban was discontinued and no anticoagulant was prescribed.",
            "Anticoagulation was held and not restarted.",
            "No anticoagulant appeared on the discharge medication list.",
            "Warfarin was stopped before discharge.",
        ],
    },
    "renal_dysfunction": {
        "supported": [
            "Creatinine was 2.8 mg/dL on discharge labs.",
            "Estimated GFR was 28 mL/min/1.73 m2.",
            "BUN was 48 mg/dL with elevated creatinine.",
            "Discharge creatinine remained above 2.0 mg/dL.",
        ],
        "refuted": [
            "Creatinine was 0.9 mg/dL on discharge labs.",
            "Renal function remained within normal limits.",
            "Estimated GFR was 92 mL/min/1.73 m2.",
            "Discharge creatinine was 1.0 mg/dL.",
        ],
    },
    "nausea": {
        "supported": [
            "Episodes of emesis continued during admission.",
            "The patient reported persistent nausea.",
            "Anti-emetic use was required overnight.",
            "Nausea limited oral intake during the stay.",
        ],
        "refuted": [
            "The patient tolerated a regular diet without nausea.",
            "No emesis was reported during admission.",
            "Anti-emetics were not required.",
            "Oral intake was tolerated without nausea.",
        ],
    },
    "dyspnea": {
        "supported": [
            "Oxygen saturation fell to 88% on room air.",
            "The patient required 2 L nasal cannula.",
            "Respiratory rate remained elevated at 24/min.",
            "The patient reported shortness of breath at rest.",
        ],
        "refuted": [
            "The patient remained on room air with normal saturation.",
            "No supplemental oxygen was required.",
            "Respiratory rate was 16/min on discharge.",
            "The patient denied shortness of breath at rest.",
        ],
    },
    "contrast_allergy": {
        "supported": [
            "Iodinated contrast was listed as an active allergy.",
            "The allergy record documents a prior contrast reaction.",
            "Contrast allergy was confirmed in the chart.",
            "Premedication was ordered because of contrast allergy.",
        ],
        "refuted": [
            "No contrast allergy was documented.",
            "The allergy list was negative for iodinated contrast.",
            "Contrast was tolerated without reaction.",
            "Allergy reconciliation showed no contrast allergy.",
        ],
    },
    "sulfonamide_allergy": {
        "supported": [
            "Sulfonamide allergy was listed on the allergy band.",
            "The chart documents a prior sulfa reaction.",
            "Sulfa agents were marked as an active allergy.",
            "The allergy record includes sulfonamide allergy.",
        ],
        "refuted": [
            "No sulfonamide allergy was documented.",
            "The allergy list was negative for sulfa agents.",
            "Sulfa antibiotics were not listed among allergies.",
            "Allergy reconciliation showed no sulfonamide allergy.",
        ],
    },
    "latex_allergy": {
        "supported": [
            "Latex allergy was documented in the chart.",
            "The allergy list includes latex sensitivity.",
            "Latex precautions were ordered.",
            "The patient carries a documented latex allergy.",
        ],
        "refuted": [
            "No latex allergy was documented.",
            "The allergy list was negative for latex.",
            "Latex precautions were not required.",
            "Allergy reconciliation showed no latex allergy.",
        ],
    },
    "insulin_therapy": {
        "supported": [
            "Insulin glargine was continued at discharge.",
            "Sliding-scale insulin remained active.",
            "Basal insulin was on the discharge medication list.",
            "Insulin therapy was maintained throughout admission.",
        ],
        "refuted": [
            "Insulin was discontinued before discharge.",
            "No insulin appeared on the discharge medication list.",
            "The patient was not receiving insulin therapy.",
            "Insulin was held and not restarted.",
        ],
    },
    "opioid_therapy": {
        "supported": [
            "Oxycodone remained on the active medication list.",
            "Scheduled opioid therapy was continued.",
            "Morphine was prescribed at discharge.",
            "Opioid analgesia was maintained during admission.",
        ],
        "refuted": [
            "No opioid was prescribed at discharge.",
            "Opioid therapy was discontinued before discharge.",
            "The discharge list contained no opioid analgesic.",
            "Scheduled opioids were not continued.",
        ],
    },
    "antiplatelet_therapy": {
        "supported": [
            "Aspirin was continued at discharge.",
            "Clopidogrel remained on the active medication list.",
            "Antiplatelet therapy was maintained.",
            "Dual antiplatelet therapy was continued.",
        ],
        "refuted": [
            "Antiplatelet therapy was discontinued before discharge.",
            "No antiplatelet agent was prescribed at discharge.",
            "Aspirin was held and not restarted.",
            "The discharge list contained no antiplatelet medication.",
        ],
    },
    "cognitive_impairment": {
        "supported": [
            "MoCA score was 19/30.",
            "The patient failed serial cognitive screening.",
            "Documented confusion persisted during admission.",
            "Orientation remained impaired on discharge exam.",
        ],
        "refuted": [
            "MoCA score was 28/30.",
            "Cognitive screening was within expected limits.",
            "The patient remained alert and oriented.",
            "No cognitive impairment was documented on exam.",
        ],
    },
    "liver_dysfunction": {
        "supported": [
            "ALT was 186 U/L on discharge labs.",
            "Bilirubin was 3.2 mg/dL with elevated transaminases.",
            "INR was 1.8 with elevated liver enzymes.",
            "Discharge LFTs remained markedly abnormal.",
        ],
        "refuted": [
            "Liver enzymes were within normal limits.",
            "ALT was 24 U/L on discharge labs.",
            "Bilirubin was 0.8 mg/dL.",
            "Discharge LFTs were unremarkable.",
        ],
    },
}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def insert_sentence(bg: str, sent: str, position_idx: int) -> str:
    parts = split_sentences(bg)
    sent = sent.rstrip(".")
    if not parts:
        return f"{bg} {sent}.".strip()
    idx = min(max(position_idx, 0), len(parts))
    parts.insert(idx, sent)
    return " ".join(parts)


def evidence_tail(profile: str, background: str) -> str:
    bg = background.strip()
    prof = profile.strip()
    if prof.startswith(bg):
        return prof[len(bg):].strip()
    return prof


def word_count(text: str) -> int:
    return len(text.split())


def pick_neutral_filler(target_words: int, key: str) -> str:
    best = min(
        NEUTRAL_FILLERS,
        key=lambda s: (abs(word_count(s) - target_words), stable_index(f"{key}:{s}", 10_000)),
    )
    words = best.split()
    if len(words) > target_words + 2:
        return " ".join(words[:target_words]) + "."
    if len(words) < target_words - 2:
        extra = NEUTRAL_FILLERS[
            (stable_index(f"{key}:extra", len(NEUTRAL_FILLERS)) + 1)
            % len(NEUTRAL_FILLERS)
        ]
        merged = (best.rstrip(".") + " " + extra).strip()
        words = merged.split()
        if len(words) > target_words + 2:
            return " ".join(words[:target_words]) + "."
        return merged if merged.endswith(".") else merged + "."
    return best


def reference_evidence_words(rows: list[dict], background_id: int,
                             concept_id: str) -> int:
    bg = None
    for row in rows:
        if (row["background_id"] == background_id
                and row["concept_id"] == concept_id
                and row["state"] == "unmentioned"):
            bg = row["profile"]
            break
    if bg is None:
        return 8
    for row in rows:
        if (row["background_id"] == background_id
                and row["concept_id"] == concept_id
                and row["state"] == "supported"):
            tail = evidence_tail(row["profile"], bg)
            if tail:
                return word_count(tail)
    return 8


def make_artifact_tiers(source: Path | None = None) -> dict[str, Path]:
    """Build artifact-resistant tiers from the clean controlled benchmark."""
    source = source or FULL_DATA
    if not source.exists():
        make_pilot(out=source, full=True)
    rows = load_rows(source)
    bg_only = {}
    for row in rows:
        if row["state"] == "unmentioned":
            bg_only[(row["background_id"], row["concept_id"])] = row["profile"]

    lengthmatched, randpos, indirect = [], [], []
    for row in rows:
        base = dict(row)
        bg = bg_only.get((row["background_id"], row["concept_id"]), row["profile"])
        if row["state"] == "unmentioned":
            target_w = reference_evidence_words(
                rows, row["background_id"], row["concept_id"])
            filler = pick_neutral_filler(
                target_w, f"{row['background_id']}:{row['concept_id']}")
            lm = dict(base)
            lm["profile"] = f"{bg} {filler}".strip()
            lm["artifact_tier"] = "lengthmatched"
            lm["neutral_filler"] = filler
            lengthmatched.append(lm)

            rp = dict(base)
            rp["profile"] = bg
            rp["artifact_tier"] = "randpos"
            randpos.append(rp)
        else:
            tail = evidence_tail(row["profile"], bg)
            pos = stable_index(
                f"{row['id']}:randpos", max(len(split_sentences(bg)), 1))
            rp = dict(base)
            rp["profile"] = insert_sentence(bg, tail, pos) if tail else bg
            rp["artifact_tier"] = "randpos"
            rp["insert_position"] = pos
            randpos.append(rp)

            lm = dict(base)
            lengthmatched.append(lm)

        ind = dict(base)
        if (row["state"] in ("supported", "refuted")
                and row["concept_id"] in INDIRECT_TEMPLATES):
            pool = INDIRECT_TEMPLATES[row["concept_id"]][row["state"]]
            sent = pool[row["paraphrase"] % len(pool)]
            ind["profile"] = f"{bg} {sent}".strip()
            ind["artifact_tier"] = "indirect"
            ind["indirect_evidence"] = sent
        else:
            ind["artifact_tier"] = "indirect"
        indirect.append(ind)

    outputs = {
        "lengthmatched": ARTIFACT_LENGTHMATCHED,
        "randpos": ARTIFACT_RANDPOS,
        "indirect": ARTIFACT_INDIRECT,
    }
    tier_rows = {
        "lengthmatched": lengthmatched,
        "randpos": randpos,
        "indirect": indirect,
    }
    for tier, path in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for item in tier_rows[tier]:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"wrote {len(tier_rows[tier])} pairs -> {path}")

    unm_lm = [r for r in lengthmatched if r["state"] == "unmentioned"]
    unm_clean = [r for r in rows if r["state"] == "unmentioned"]
    other_lm = [r for r in lengthmatched if r["state"] != "unmentioned"]
    other_clean = [r for r in rows if r["state"] != "unmentioned"]
    if unm_lm and other_lm:
        print(
            "lengthmatched avg words: "
            f"unmentioned={sum(word_count(r['profile']) for r in unm_lm)/len(unm_lm):.1f}, "
            f"other={sum(word_count(r['profile']) for r in other_lm)/len(other_lm):.1f} "
            f"(clean unmentioned="
            f"{sum(word_count(r['profile']) for r in unm_clean)/len(unm_clean):.1f}, "
            f"other="
            f"{sum(word_count(r['profile']) for r in other_clean)/len(other_clean):.1f})"
        )
    rp_pos = Counter(r.get("insert_position", -1) for r in randpos
                     if r["state"] != "unmentioned")
    print(f"randpos insert positions used: {len(rp_pos)} distinct slots")
    ind_sr = sum(
        1 for r in indirect if r["state"] in ("supported", "refuted")
        and r.get("indirect_evidence"))
    print(f"indirect evidence sentences replaced: {ind_sr}")
    return outputs


def load_rows(path: Path = DATA) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
HF_MAX_SEQ = 512
MIMIC_CHUNK_CHARS = 4800
MIMIC_CHUNK_OVERLAP = 400


def split_text_chunks(text: str, max_chars: int = MIMIC_CHUNK_CHARS,
                      overlap: int = MIMIC_CHUNK_OVERLAP) -> list[str]:
    """Split long clinical notes into overlapping chunks for pooled encoding."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def pool_embeddings(embs: np.ndarray) -> np.ndarray:
    pooled = embs.mean(axis=0)
    norm = np.linalg.norm(pooled) + 1e-9
    return (pooled / norm).astype(np.float32)


class Embedder:
    def __init__(self, name: str = "hash", device: str | None = None):
        self.name = name
        self.device = device
        self._model = None
        self._tok = None
        self._tfidf = None
        self._pooling = "mean"
        self.dim = 384

    def fit(self, texts: list[str]):
        """Fit TF-IDF on train texts (no-op for frozen neural embedders)."""
        if self.name != "tfidf":
            return
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(
            max_features=4096, ngram_range=(1, 2), min_df=2,
            sublinear_tf=True, stop_words="english",
        )
        self._tfidf.fit(texts)
        self.dim = len(self._tfidf.vocabulary_)

    def _load(self):
        if self._model is not None or self.name in ("hash", "tfidf"):
            return
        if self.name in ("qwen", "qwen3", "Qwen/Qwen3-Embedding-0.6B"):
            from sentence_transformers import SentenceTransformer
            mid = "Qwen/Qwen3-Embedding-0.6B"
            self._model = SentenceTransformer(mid, device=self.device or "cpu")
            self.dim = self._model.get_sentence_embedding_dimension()
        elif self.name in (
            "jina", "jina-v5", "jina-v5-text",
            "jinaai/jina-embeddings-v5-text-nano-text-matching",
        ):
            from sentence_transformers import SentenceTransformer
            mid = "jinaai/jina-embeddings-v5-text-nano-text-matching"
            self._model = SentenceTransformer(mid, trust_remote_code=True,
                                              device=self.device or "cpu")
            self.dim = self._model.get_sentence_embedding_dimension()
        elif self.name in ("snowflake", "arctic",
                           "Snowflake/snowflake-arctic-embed-m-v2.0"):
            from sentence_transformers import SentenceTransformer
            mid = "Snowflake/snowflake-arctic-embed-m-v2.0"
            self._model = SentenceTransformer(
                mid, trust_remote_code=True, device=self.device or "cpu",
                config_kwargs={"use_memory_efficient_attention": False},
                model_kwargs={"attn_implementation": "eager"},
            )
            self.dim = self._model.get_sentence_embedding_dimension()
        elif self.name in ("clinicalbert", "emilyalsentzer/Bio_ClinicalBERT"):
            import torch
            from transformers import AutoModel, AutoTokenizer
            mid = "emilyalsentzer/Bio_ClinicalBERT"
            self._tok = AutoTokenizer.from_pretrained(mid)
            self._model = AutoModel.from_pretrained(mid)
            self._model.eval()
            if self.device:
                self._model.to(self.device)
            self.dim = self._model.config.hidden_size
            self._torch = torch
        elif self.name in ("biolord", "FremyCompany/BioLORD-2023"):
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                "FremyCompany/BioLORD-2023", device=self.device or "cpu")
            self.dim = self._model.get_sentence_embedding_dimension()
        elif self.name in ("medcpt", "ncbi/MedCPT-Article-Encoder"):
            import torch
            from transformers import AutoModel, AutoTokenizer
            mid = "ncbi/MedCPT-Article-Encoder"
            self._tok = AutoTokenizer.from_pretrained(mid)
            self._model = AutoModel.from_pretrained(mid)
            self._model.eval()
            if self.device:
                self._model.to(self.device)
            self.dim = self._model.config.hidden_size
            self._torch = torch
            self._pooling = "cls"
        elif self.name in ("med-gte", "med-gte-hybrid",
                           "MedAI-HS/med-gte-hybrid"):
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                "MedAI-HS/med-gte-hybrid", trust_remote_code=True,
                device=self.device or "cpu")
            self.dim = self._model.get_sentence_embedding_dimension()
        else:
            raise ValueError(f"unknown model: {self.name}")

    def _encode_st(self, texts: list[str], batch_size: int, **kwargs) -> np.ndarray:
        import torch
        bs = batch_size
        while True:
            try:
                return np.asarray(self._model.encode(
                    texts, batch_size=bs, **kwargs))
            except RuntimeError as exc:
                oom = "out of memory" in str(exc).lower()
                if not oom or bs <= 1:
                    raise
                if self.device and str(self.device).startswith("cuda"):
                    torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                print(f"CUDA OOM; retry SentenceTransformer encode "
                      f"with batch_size={bs}", flush=True)

    def encode(self, texts: list[str], batch_size: int = 32,
               chunk_chars: int | None = None) -> np.ndarray:
        self._load()
        if self.name == "hash":
            return np.vstack([self._hash_vec(t) for t in texts])
        if self.name == "tfidf":
            if self._tfidf is None:
                self.fit(texts)
            X = self._tfidf.transform(texts).astype(np.float32).toarray()
            n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
            return X / n
        if chunk_chars and chunk_chars > 0:
            return np.vstack([
                self._encode_one_pooled(text, batch_size, chunk_chars)
                for text in texts
            ])
        if self._tok is not None:
            return self._encode_hf(texts, batch_size)
        if self.name in (
            "jina", "jina-v5", "jina-v5-text",
            "jinaai/jina-embeddings-v5-text-nano-text-matching",
        ):
            texts = [t if t.startswith("Document: ") else f"Document: {t}"
                     for t in texts]
            return self._encode_st(
                texts, batch_size, prompt="", show_progress_bar=len(texts) > 200,
                normalize_embeddings=True,
            )
        return self._encode_st(
            texts, batch_size, show_progress_bar=len(texts) > 200,
            normalize_embeddings=True,
        )

    def _encode_one_pooled(self, text: str, batch_size: int,
                           chunk_chars: int) -> np.ndarray:
        chunks = split_text_chunks(text, max_chars=chunk_chars)
        if self._tok is not None:
            embs = self._encode_hf(chunks, batch_size)
        elif self.name in (
            "jina", "jina-v5", "jina-v5-text",
            "jinaai/jina-embeddings-v5-text-nano-text-matching",
        ):
            chunks = [c if c.startswith("Document: ") else f"Document: {c}"
                      for c in chunks]
            embs = self._encode_st(
                chunks, min(batch_size, len(chunks)), prompt="",
                show_progress_bar=False, normalize_embeddings=True,
            )
        else:
            embs = self._encode_st(
                chunks, min(batch_size, len(chunks)),
                show_progress_bar=False, normalize_embeddings=True,
            )
        return pool_embeddings(embs) if len(embs) > 1 else embs[0]

    def _hash_vec(self, text: str) -> np.ndarray:
        """Deterministic bag-of-hashed-ngrams for offline smoke tests."""
        v = np.zeros(self.dim, dtype=np.float64)
        toks = re.findall(r"[a-z0-9]+", text.lower())
        for i in range(len(toks)):
            for n in (1, 2):
                if i + n <= len(toks):
                    g = " ".join(toks[i:i + n])
                    h = stable_index(g, self.dim)
                    v[h] += 1.0
        # light negation / absence cues so trap is detectable offline
        for cue, w in (("denies", -2), ("negative for", -2), ("absent", -2),
                       ("no ", -1), ("reports", 1.5), ("confirms", 1.5),
                       ("endorses", 1.5), ("not assessed", 0.2),
                       ("unclear", 0.1), ("father", 0.3), ("family", 0.3)):
            if cue in text.lower():
                v[stable_index(cue, self.dim)] += w
        n = np.linalg.norm(v) + 1e-9
        return (v / n).astype(np.float32)

    def _encode_hf(self, texts: list[str], batch_size: int) -> np.ndarray:
        torch = self._torch
        outs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self._tok(batch, padding=True, truncation=True,
                            max_length=HF_MAX_SEQ, return_tensors="pt")
            if self.device:
                enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                h = self._model(**enc).last_hidden_state
                if self._pooling == "cls":
                    pooled = h[:, 0]
                else:
                    mask = enc["attention_mask"].unsqueeze(-1)
                    pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outs.append(pooled.cpu().numpy())
        return np.vstack(outs)


def pair_features(ex: np.ndarray, eh: np.ndarray) -> np.ndarray:
    return np.concatenate([ex, eh, np.abs(ex - eh), ex * eh], axis=-1)


# ---------------------------------------------------------------------------
# Metrics / experiments
# ---------------------------------------------------------------------------
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def margins(rows: list[dict], E: dict[str, np.ndarray], emb: Embedder) -> dict:
    """M(x,c) = cos(E(x), pos) - cos(E(x), neg); avg over anchors."""
    by_state = defaultdict(list)
    cache = {}
    for r in rows:
        c = r["concept"]
        if c not in cache:
            pos_text, neg_text = concept_anchors(c, r["group"])
            pos = emb.encode(pos_text)
            neg = emb.encode(neg_text)
            cache[c] = (pos, neg)
        pos, neg = cache[c]
        zx = E[r["id"]]
        m = np.mean([cosine(zx, p) for p in pos]) - np.mean(
            [cosine(zx, n) for n in neg])
        by_state[r["state"]].append(m)
    summary = {s: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
               for s, v in by_state.items()}
    # Matched-pair ECI. By definition, U→R gives ECI→1:
    # ECI = 1 - d(U,R) / d(S,R). Compute within background×concept,
    # then average; global centroids would confound concept identity.
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grouped[(r["background_id"], r["concept_id"])][r["state"]].append(E[r["id"]])
    eci = []
    for states in grouped.values():
        if not {"supported", "refuted", "unmentioned"} <= states.keys():
            continue
        z_s = np.mean(states["supported"], axis=0)
        z_r = np.mean(states["refuted"], axis=0)
        z_u = np.mean(states["unmentioned"], axis=0)
        d_sr = np.linalg.norm(z_s - z_r)
        d_ur = np.linalg.norm(z_u - z_r)
        if d_sr > 1e-9:
            eci.append(1.0 - d_ur / d_sr)
    summary["ECI"] = {
        "mean": float(np.mean(eci)), "std": float(np.std(eci)), "n": len(eci),
        "definition": "1 - d(U,R)/d(S,R), matched background-concept",
    }
    return summary


def protocol_split(rows: list[dict], protocol: str) -> tuple[list[dict], list[dict]]:
    """Return disjoint train/test conditions for one generalization axis."""
    base = [r for r in rows if r["concept_split"] == "train"
            and r["background_split"] == "train"
            and r["template_split"] in ("train", "na")]
    if protocol == "iid":
        return ([r for r in base if r["iid_split"] == "train"],
                [r for r in base if r["iid_split"] == "test"])
    if protocol == "new_background":
        train = base
        test = [r for r in rows if r["concept_split"] == "train"
                and r["background_split"] == "test"
                and r["template_split"] in ("train", "na")]
        return train, test
    if protocol == "new_template":
        train = base
        test = [r for r in rows if r["concept_split"] == "train"
                and r["background_split"] == "train"
                and r["template_split"] == "test"]
        return train, test
    if protocol == "new_concept":
        train = base
        test = [r for r in rows if r["concept_split"] == "test"
                and r["background_split"] == "train"
                and r["template_split"] in ("train", "na")]
        return train, test
    raise ValueError(f"unknown protocol: {protocol}")


def _train_lr(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
    clf.fit(Xs, y)
    return clf, sc


SUPPORTED = "Supported"
REFUTED = "Refuted"
Y_SUPPORTED = 1
Y_REFUTED = 0


def _y_from_sr_label(label: str) -> int:
    if label == SUPPORTED:
        return Y_SUPPORTED
    if label == REFUTED:
        return Y_REFUTED
    raise ValueError(f"not a binary S/R label: {label!r}")


def _sr_class_indices(clf) -> tuple[int, int, list]:
    classes = list(clf.classes_)
    assert Y_REFUTED in classes, classes
    assert Y_SUPPORTED in classes, classes
    return classes.index(Y_REFUTED), classes.index(Y_SUPPORTED), classes


def _predict_sr_from_proba(proba: np.ndarray, clf) -> dict:
    i_ref, i_sup, classes = _sr_class_indices(clf)
    p_ref = proba[:, i_ref]
    p_sup = proba[:, i_sup]
    pred = np.where(p_sup >= p_ref, SUPPORTED, REFUTED)
    return {
        "classes": classes,
        "i_ref": i_ref,
        "i_sup": i_sup,
        "p_ref": p_ref,
        "p_sup": p_sup,
        "pred": pred,
        "proba": proba,
    }


def binary_trap(rows: list[dict], E: dict[str, np.ndarray], emb: Embedder,
                tau: float = 0.9, protocol: str = "new_background") -> dict:
    """Train on Supported∪Refuted only; evaluate Unknown→Refuted collapse."""
    # encode propositions once
    props = {r["proposition"] for r in rows}
    P = {p: emb.encode([p])[0] for p in props}

    def feats(subset):
        X, y, meta = [], [], []
        for r in subset:
            X.append(pair_features(E[r["id"]], P[r["proposition"]]))
            y.append(1 if r["y3"] == "Supported" else 0)  # 1=Supported, 0=Refuted
            meta.append(r)
        return np.asarray(X), np.asarray(y), meta

    protocol_train, test = protocol_split(rows, protocol)
    train = [r for r in protocol_train if r["y3"] in ("Supported", "Refuted")]
    Xtr, ytr, _ = feats(train)
    clf, sc = _train_lr(Xtr, ytr)

    Xte, yte, meta = feats(test)
    Xtes = sc.transform(Xte)
    proba = clf.predict_proba(Xtes)
    sr = _predict_sr_from_proba(proba, clf)
    classes = sr["classes"]
    p_ref = sr["p_ref"]
    p_sup = sr["p_sup"]
    pred = sr["pred"]
    logits = clf.decision_function(Xtes)  # positive = Supported

    known_mask = np.array([m["y3"] in ("Supported", "Refuted") for m in meta])
    unk_mask = ~known_mask
    known_acc = float(np.mean(
        (pred[known_mask] == np.array([m["y3"] for m in meta])[known_mask])
    )) if known_mask.any() else float("nan")

    unk_pref = p_ref[unk_mask]
    unr = float(np.mean(pred[unk_mask] == "Refuted")) if unk_mask.any() else float("nan")
    hcunr = float(np.mean((pred[unk_mask] == REFUTED) & (unk_pref >= tau))) if unk_mask.any() else float("nan")

    # Background×concept is the independent unit; templates are repeated
    # measurements. Bootstrap groups, never individual rows.
    grouped_idx = defaultdict(list)
    for i, r in enumerate(meta):
        grouped_idx[(r["background_id"], r["concept_id"])].append(i)
    group_known, group_unm_unr, group_unm_hc = [], [], []
    group_unm_minus_refuted_logit = []
    for idxs in grouped_idx.values():
        ki = [i for i in idxs if known_mask[i]]
        ui = [i for i in idxs if meta[i]["state"] == "unmentioned"]
        ri = [i for i in idxs if meta[i]["state"] == "refuted"]
        if ki:
            group_known.append(float(np.mean(
                pred[ki] == np.asarray([meta[i]["y3"] for i in ki]))))
        if ui:
            group_unm_unr.append(float(np.mean(pred[ui] == "Refuted")))
            group_unm_hc.append(float(np.mean(
                (pred[ui] == REFUTED) & (p_ref[ui] >= tau))))
        if ui and ri:
            group_unm_minus_refuted_logit.append(
                float(np.mean(logits[ui]) - np.mean(logits[ri])))

    def bootstrap_ci(values, n_boot=2000):
        if not values:
            return None
        arr = np.asarray(values)
        rng = np.random.default_rng(0)
        means = np.mean(rng.choice(arr, size=(n_boot, len(arr)), replace=True), axis=1)
        return {
            "mean": float(arr.mean()),
            "lower": float(np.quantile(means, 0.025)),
            "upper": float(np.quantile(means, 0.975)),
            "n_groups": len(arr),
        }

    # State-specific metrics: Unmentioned is the non-observation headline;
    # other unknown states are assertion/generalization analyses.
    per_state, hc_per_state, state_counts = {}, {}, {}
    decision_axis = {}
    for st in UNKNOWN:
        m = np.array([x["state"] == st for x in meta])
        state_counts[st] = int(m.sum())
        if m.any():
            per_state[st] = float(np.mean(pred[m] == "Refuted"))
            hc_per_state[st] = float(np.mean(
                (pred[m] == REFUTED) & (p_ref[m] >= tau)))
        else:
            per_state[st] = None
            hc_per_state[st] = None
        if m.any():
            decision_axis[st] = {
                "mean_logit": float(np.mean(logits[m])),
                "std_logit": float(np.std(logits[m])),
                "q05": float(np.quantile(logits[m], 0.05)),
                "q50": float(np.quantile(logits[m], 0.50)),
                "q95": float(np.quantile(logits[m], 0.95)),
            }
    for label, state in (("Supported", "supported"), ("Refuted", "refuted")):
        m = np.array([x["state"] == state for x in meta])
        if m.any():
            decision_axis[label] = {
                "mean_logit": float(np.mean(logits[m])),
                "std_logit": float(np.std(logits[m])),
                "q05": float(np.quantile(logits[m], 0.05)),
                "q50": float(np.quantile(logits[m], 0.50)),
                "q95": float(np.quantile(logits[m], 0.95)),
            }

    ood = None
    if protocol == "new_background":
        try:
            from sklearn.covariance import LedoitWolf
            from sklearn.decomposition import PCA
            from sklearn.metrics import roc_auc_score
            dim = min(64, Xtr.shape[0] - 1, Xtr.shape[1])
            pca = PCA(n_components=dim, random_state=0)
            tr_p = pca.fit_transform(sc.transform(Xtr))
            te_p = pca.transform(Xtes)
            lw = LedoitWolf().fit(tr_p)
            distances = np.sqrt(np.maximum(lw.mahalanobis(te_p), 0))
            train_distances = np.sqrt(np.maximum(
                lw.mahalanobis(tr_p), 0))
            distance_threshold = float(np.quantile(train_distances, 0.95))
            distance_keep = distances <= distance_threshold
            unm_mask_ood = np.array(
                [x["state"] == "unmentioned" for x in meta])
            known_or_unm_ood = known_mask | unm_mask_ood
            ood = {
                "pca_dim": dim,
                "distance_by_state": {
                    st: {
                        "mean": float(np.mean(distances[
                            np.array([x["state"] == st for x in meta])])),
                        "std": float(np.std(distances[
                            np.array([x["state"] == st for x in meta])])),
                    }
                    for st in STATES
                    if any(x["state"] == st for x in meta)
                },
                "unknown_AUROC": float(roc_auc_score(
                    unk_mask.astype(int), distances)),
                "unmentioned_AUROC": float(roc_auc_score(
                    unm_mask_ood[known_or_unm_ood].astype(int),
                    distances[known_or_unm_ood],
                )) if unm_mask_ood.any() else None,
                "abstention": {
                    "threshold_source":
                        "95th percentile of known training distance",
                    "threshold": distance_threshold,
                    "known_test_coverage": float(np.mean(
                        distance_keep[known_mask])),
                    "unknown_reject_rate": float(np.mean(
                        ~distance_keep[unk_mask])),
                    "unmentioned_reject_rate": float(np.mean(
                        ~distance_keep[unm_mask_ood])),
                    "accepted_unmentioned_UNR": float(np.mean(
                        pred[unm_mask_ood & distance_keep] == REFUTED
                    )) if (unm_mask_ood & distance_keep).any() else None,
                    "accepted_unmentioned_n": int(
                        (unm_mask_ood & distance_keep).sum()),
                },
            }
        except Exception as exc:
            ood = {"error": str(exc)}

    # entropy on unknown
    eps = 1e-12
    ent = -(proba * np.log(proba + eps)).sum(1)
    unk_ent = float(np.mean(ent[unk_mask])) if unk_mask.any() else float("nan")

    # Unknown detection AUROC via max-prob (low maxprob => unknown)
    try:
        from sklearn.metrics import roc_auc_score
        maxp = proba.max(1)
        # unknown = positive for detection; score = 1 - maxp
        auroc = float(roc_auc_score(unk_mask.astype(int), 1 - maxp))
        unm_mask = np.array([x["state"] == "unmentioned" for x in meta])
        known_or_unm = known_mask | unm_mask
        unmentioned_auroc = float(roc_auc_score(
            unm_mask[known_or_unm].astype(int),
            (1 - maxp)[known_or_unm],
        )) if unm_mask.any() else float("nan")
    except Exception:
        auroc = float("nan")
        unmentioned_auroc = float("nan")

    # Train-calibrated confidence abstention: retain 95% of known training
    # examples, then evaluate coverage and rejection on held-out cases.
    train_maxp = clf.predict_proba(sc.transform(Xtr)).max(1)
    confidence_threshold = float(np.quantile(train_maxp, 0.05))
    confidence_keep = maxp >= confidence_threshold
    confidence_abstention = {
        "threshold_source": "5th percentile of known training confidence",
        "threshold": confidence_threshold,
        "known_test_coverage": float(np.mean(
            confidence_keep[known_mask])) if known_mask.any() else None,
        "unknown_reject_rate": float(np.mean(
            ~confidence_keep[unk_mask])) if unk_mask.any() else None,
        "unmentioned_reject_rate": float(np.mean(
            ~confidence_keep[unm_mask])) if unm_mask.any() else None,
    }
    if unm_mask.any():
        accepted_unm = unm_mask & confidence_keep
        confidence_abstention["accepted_unmentioned_UNR"] = float(np.mean(
            pred[accepted_unm] == REFUTED)) if accepted_unm.any() else None
        confidence_abstention["accepted_unmentioned_n"] = int(accepted_unm.sum())

    # Risk-coverage: abstain when maxp < t
    rc = []
    for t in np.linspace(0.5, 0.99, 20):
        keep = maxp >= t
        cov = float(keep.mean())
        if keep.sum() == 0:
            continue
        # known-state error among kept known
        km = keep & known_mask
        err = float(1 - np.mean(
            pred[km] == np.array([m["y3"] for m in meta])[km]
        )) if km.any() else float("nan")
        unk_rej = float((~keep & unk_mask).sum() / max(unk_mask.sum(), 1))
        rc.append({"tau": float(t), "coverage": cov, "known_error": err,
                   "unknown_reject_rate": unk_rej})

    # eligibility unsafe auto-clearance on unknown (predict Refuted => eligible for exclusion)
    elig_unk = [m for m in meta if m["y3"] == "Unknown"]
    if elig_unk:
        idx = [i for i, m in enumerate(meta) if m["y3"] == "Unknown"]
        # exclusion criterion: Supported => ineligible; Refuted => eligible; Unknown => review
        auto_clear = sum(1 for i in idx if pred[i] == "Refuted") / len(idx)
        auto_excl = sum(1 for i in idx if pred[i] == "Supported") / len(idx)
    else:
        auto_clear, auto_excl = float("nan"), float("nan")

    return {
        "protocol": protocol,
        "n_train_known": len(train),
        "n_test": len(test),
        "known_acc": known_acc,
        "UNR": unr,
        f"HCUNR@{tau}": hcunr,
        "group_bootstrap": {
            "known_acc": bootstrap_ci(group_known),
            "unmentioned_UNR": bootstrap_ci(group_unm_unr),
            f"unmentioned_HCUNR@{tau}": bootstrap_ci(group_unm_hc),
            "unmentioned_minus_refuted_logit": bootstrap_ci(
                group_unm_minus_refuted_logit),
        },
        "unknown_entropy": unk_ent,
        "unknown_auroc": auroc,
        "unmentioned_auroc": unmentioned_auroc,
        "confidence_abstention": confidence_abstention,
        "UNR_by_state": per_state,
        f"HCUNR_by_state@{tau}": hc_per_state,
        "test_state_counts": state_counts,
        "decision_axis_projection": decision_axis,
        "mahalanobis_ood": ood,
        "protocol_note": (
            "Assertion-template generalization; contains no Unmentioned cases."
            if protocol == "new_template" else None
        ),
        "UACR_unsafe_clearance": auto_clear,
        "UER_unsupported_exclusion": auto_excl,
        "risk_coverage": rc,
        "_artifacts": {
            "meta": meta, "pred": pred, "p_ref": p_ref, "p_sup": p_sup,
            "proba": proba, "clf": clf, "sc": sc, "P": P,
        },
    }


def capr_oracle(rows: list[dict], E: dict[str, np.ndarray], emb: Embedder,
                tau: float = 0.9, protocol: str = "new_background") -> dict:
    """Fair mitigation: oracle gate plus three-way learning without gold leakage."""
    props = {r["proposition"] for r in rows}
    P = {p: emb.encode([p])[0] for p in props}
    train_all, test = protocol_split(rows, protocol)

    def pack(subset, ternary=False):
        X, y = [], []
        for r in subset:
            X.append(pair_features(E[r["id"]], P[r["proposition"]]))
            if ternary:
                y.append({"Supported": 0, "Refuted": 1, "Unknown": 2}[r["y3"]])
            else:
                y.append(1 if r["y3"] == "Supported" else 0)
        return np.asarray(X), np.asarray(y)

    # Binary classifier never receives gold mention/assertion features.
    train_k = [r for r in train_all if r["y3"] != "Unknown"]
    Xtr, ytr = pack(train_k, ternary=False)
    clf, sc = _train_lr(Xtr, ytr)
    Xte, _ = pack(test, ternary=False)
    proba = clf.predict_proba(sc.transform(Xte))
    classes = list(clf.classes_)
    binary_pred = np.where(
        proba[:, classes.index(1)] >= proba[:, classes.index(0)],
        "Supported", "Refuted",
    )
    gold = np.array([r["y3"] for r in test])
    known_mask = gold != "Unknown"
    unk_mask = ~known_mask
    known_acc = float(np.mean(binary_pred[known_mask] == gold[known_mask]))

    # Oracle observation gate is a rule, not a learned feature:
    # genuine unknown states always abstain; known states retain binary output.
    gate_pred = binary_pred.astype(object)
    gate_pred[unk_mask] = "Unknown"

    # Three-way model: only representation features, stratified unknown samples,
    # ten seeds. This estimates supervision benefit without assertion leakage.
    curves = defaultdict(list)
    unk_train_all = [r for r in train_all if r["y3"] == "Unknown"]
    known_train = [r for r in train_all if r["y3"] != "Unknown"]
    grouped_unknown = defaultdict(list)
    for r in unk_train_all:
        grouped_unknown[(r["background_id"], r["concept_id"], r["state"])].append(r)
    strata = defaultdict(list)
    for group_key in grouped_unknown:
        strata[(group_key[1], group_key[2])].append(group_key)

    for seed in range(10):
        rng = random.Random(seed)
        shuffled = {}
        for key, group_keys in strata.items():
            shuffled[key] = group_keys.copy()
            rng.shuffle(shuffled[key])
        order = []
        depth = 0
        keys = sorted(shuffled)
        while len(order) < len(unk_train_all):
            added = False
            for key in keys:
                if depth < len(shuffled[key]):
                    group_key = shuffled[key][depth]
                    # One labeled row per semantic group. Remaining templates
                    # are augmentation, not additional supervision budget.
                    order.append(rng.choice(grouped_unknown[group_key]))
                    added = True
            if not added:
                break
            depth += 1
        for n in (10, 50, 100):
            subset = known_train + order[:min(n, len(order))]
            X, y = pack(subset, ternary=True)
            clf3, sc3 = _train_lr(X, y)
            Xt, _ = pack(test, ternary=True)
            pred3 = clf3.predict(sc3.transform(Xt))
            pred_s = np.array([{0: "Supported", 1: "Refuted", 2: "Unknown"}[i]
                               for i in pred3])
            curves[n].append({
                "known_acc": float(np.mean(pred_s[known_mask] == gold[known_mask])),
                "unknown_recall": float(np.mean(pred_s[unk_mask] == "Unknown")),
                "UNR": float(np.mean(pred_s[unk_mask] == "Refuted")),
            })

    curve_summary = {}
    for n, records in curves.items():
        curve_summary[n] = {
            metric: {
                "mean": float(np.mean([r[metric] for r in records])),
                "std": float(np.std([r[metric] for r in records])),
            }
            for metric in ("known_acc", "unknown_recall", "UNR")
        }

    return {
        "protocol": protocol,
        "oracle_gate": {
            "known_acc": known_acc,
            "unknown_recall": float(np.mean(gate_pred[unk_mask] == "Unknown")),
            "unsafe_auto_clearance": 0.0,
            "note": "Rule-based upper bound; no learned gold assertion features.",
        },
        "mention_mask_identifiable_from_binary_train": False,
        "fewshot_unknown_10_seed": curve_summary,
    }


def _binary_metrics(y_known: np.ndarray, pred_known: np.ndarray,
                    p_known: np.ndarray, p_ref_unm: np.ndarray,
                    maxp_unm: np.ndarray, tau: float) -> dict:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    conf = np.maximum(p_known, 1 - p_known)
    correct = (pred_known == y_known).astype(float)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    known_unknown = np.r_[np.zeros(len(conf)), np.ones(len(maxp_unm))]
    ood_score = np.r_[1 - conf, 1 - maxp_unm]
    return {
        "known_acc": float(np.mean(pred_known == y_known)),
        "unmentioned_UNR": float(np.mean(p_ref_unm >= 0.5)),
        f"unmentioned_HCUNR@{tau}": float(np.mean(p_ref_unm > tau)),
        "unmentioned_auroc": float(roc_auc_score(known_unknown, ood_score)),
        "known_ECE": float(ece),
        "known_Brier": float(brier_score_loss(y_known, p_known)),
    }


def classifier_robustness(rows: list[dict], E: dict[str, np.ndarray],
                          emb: Embedder, tau: float = 0.9) -> dict:
    """Check whether extrapolation survives heads, regularization and calibration."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    train_all, test_all = protocol_split(rows, "new_background")
    val_all = [r for r in rows if r["concept_split"] == "train"
               and r["background_split"] == "val"
               and r["template_split"] in ("train", "na")]
    train = [r for r in train_all if r["y3"] != "Unknown"]
    val = [r for r in val_all if r["y3"] != "Unknown"]
    test_known = [r for r in test_all if r["y3"] != "Unknown"]
    test_unm = [r for r in test_all if r["state"] == "unmentioned"]

    props = {r["proposition"] for r in rows}
    P = {p: emb.encode([p])[0] for p in props}
    F = {r["id"]: pair_features(E[r["id"]], P[r["proposition"]])
         for r in rows}

    def xy(subset):
        return (np.asarray([F[r["id"]] for r in subset]),
                np.asarray([1 if r["y3"] == "Supported" else 0 for r in subset]))

    Xtr, ytr = xy(train)
    Xv, yv = xy(val)
    Xk, yk = xy(test_known)
    Xu, _ = xy(test_unm)
    results = {}

    def evaluate(name, clf, transform=lambda x: x):
        pk = clf.predict_proba(transform(Xk))[:, list(clf.classes_).index(1)]
        pu = clf.predict_proba(transform(Xu))[:, list(clf.classes_).index(1)]
        results[name] = _binary_metrics(
            yk, (pk >= 0.5).astype(int), pk, 1 - pu,
            np.maximum(pu, 1 - pu), tau)
        return pk, pu

    # C curve with and without scaling.
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    for c in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        clf = LogisticRegression(
            C=c, max_iter=3000, class_weight="balanced", random_state=0)
        clf.fit(Xtr_s, ytr)
        evaluate(f"lr_scaled_C={c:g}", clf, scaler.transform)
    raw_lr = LogisticRegression(
        C=1.0, max_iter=3000, class_weight="balanced", random_state=0)
    raw_lr.fit(Xtr, ytr)
    evaluate("lr_no_scaler_C=1", raw_lr)

    # PCA after scaling; cap dimensions by available samples.
    for dim in (64, 128, 256):
        if dim >= min(Xtr_s.shape):
            continue
        pca = PCA(n_components=dim, random_state=0).fit(Xtr_s)
        clf = LogisticRegression(
            C=1.0, max_iter=3000, class_weight="balanced", random_state=0)
        clf.fit(pca.transform(Xtr_s), ytr)
        evaluate(f"pca{dim}_lr", clf,
                 lambda x, pca=pca: pca.transform(scaler.transform(x)))

    # Temperature scaling on held-out known validation examples.
    base = LogisticRegression(
        C=1.0, max_iter=3000, class_weight="balanced", random_state=0)
    base.fit(Xtr_s, ytr)
    logits_v = base.decision_function(scaler.transform(Xv))
    temperatures = np.logspace(-1, 1.5, 80)
    losses = []
    for t in temperatures:
        pv = 1 / (1 + np.exp(-np.clip(logits_v / t, -30, 30)))
        losses.append(-np.mean(yv * np.log(pv + 1e-9)
                               + (1 - yv) * np.log(1 - pv + 1e-9)))
    best_t = float(temperatures[int(np.argmin(losses))])
    lk = base.decision_function(scaler.transform(Xk)) / best_t
    lu = base.decision_function(scaler.transform(Xu)) / best_t
    pk = 1 / (1 + np.exp(-np.clip(lk, -30, 30)))
    pu = 1 / (1 + np.exp(-np.clip(lu, -30, 30)))
    results["temperature_scaled"] = {
        **_binary_metrics(yk, (pk >= 0.5).astype(int), pk, 1 - pu,
                          np.maximum(pu, 1 - pu), tau),
        "temperature": best_t,
    }

    # Non-parametric heads on normalized pair features.
    norm = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    Xtr_n, Xk_n, Xu_n = norm(Xtr), norm(Xk), norm(Xu)
    centroids = np.vstack([Xtr_n[ytr == c].mean(0) for c in (0, 1)])
    centroids = norm(centroids)
    sk, su = Xk_n @ centroids.T, Xu_n @ centroids.T
    pk = 1 / (1 + np.exp(-(sk[:, 1] - sk[:, 0]) * 10))
    pu = 1 / (1 + np.exp(-(su[:, 1] - su[:, 0]) * 10))
    results["nearest_centroid"] = _binary_metrics(
        yk, (pk >= 0.5).astype(int), pk, 1 - pu,
        np.maximum(pu, 1 - pu), tau)

    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="cosine")
    knn.fit(Xtr_n, ytr)
    evaluate("knn5", knn, norm)
    return results


def result_suffix(data_path: Path) -> str:
    if data_path == FULL_DATA:
        return "_full"
    for tier, path in ARTIFACT_TIERS.items():
        if data_path == path:
            return f"_{tier}"
    if data_path == DATA:
        return ""
    return f"_{data_path.stem}"


def run(model: str = "hash", device: str | None = None, tau: float = 0.9,
        robustness: bool = False, data_path: Path = DATA):
    if not data_path.exists():
        make_pilot(out=data_path, full=data_path == FULL_DATA)
    rows = load_rows(data_path)
    emb = Embedder(model, device=device)
    if model == "tfidf":
        train_rows, _ = protocol_split(rows, "new_background")
        fit_texts = [r["profile"] for r in train_rows]
        fit_texts += sorted({r["proposition"] for r in train_rows})
        emb.fit(fit_texts)
    print(f"encoding {len(rows)} profiles with {model} ...")
    texts = [r["profile"] for r in rows]
    cache_dir = OUT / "embedding_cache"
    fingerprint = hashlib.blake2b(
        (model + "\0" + "\0".join(texts)).encode("utf-8"), digest_size=12
    ).hexdigest()
    cache_path = cache_dir / f"{model.replace('/', '_')}_{fingerprint}.npy"
    if model != "tfidf" and cache_path.exists():
        Z = np.load(cache_path)
        print(f"loaded embedding cache {cache_path}")
    else:
        Z = emb.encode(texts)
        if model != "tfidf":
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, Z)
            print(f"saved embedding cache {cache_path}")
    E = {r["id"]: Z[i] for i, r in enumerate(rows)}

    print("geometry ...")
    geo = margins(rows, E, emb)
    traps, mitigations = {}, {}
    arts = None
    for protocol in ("iid", "new_background", "new_template", "new_concept"):
        print(f"binary trap [{protocol}] ...")
        trap_i = binary_trap(rows, E, emb, tau=tau, protocol=protocol)
        artifact_i = trap_i.pop("_artifacts")
        traps[protocol] = trap_i
        if protocol in ("new_background", "new_concept"):
            if protocol == "new_background":
                arts = artifact_i
            print(f"mitigation [{protocol}] ...")
            mitigations[protocol] = capr_oracle(
                rows, E, emb, tau=tau, protocol=protocol)
    trap = traps["new_background"]
    mit = mitigations["new_background"]

    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "model": model,
        "n": len(rows),
        "geometry": geo,
        "binary_trap_by_protocol": traps,
        "mitigation_by_protocol": mitigations,
        "classifier_robustness": (
            classifier_robustness(rows, E, emb, tau=tau)
            if robustness else None
        ),
        "go_criteria": {
            "known_acc_ge_0.80": trap["known_acc"] >= 0.80,
            "hcunr_ge_0.30": trap[f"HCUNR@{tau}"] >= 0.30,
            "ordinary_abstention_unknown_auroc_lt_0.80":
                trap["unknown_auroc"] < 0.80,
        },
    }
    dataset_suffix = result_suffix(data_path)
    outp = OUT / f"results_{model.replace('/', '_')}{dataset_suffix}.json"
    # risk_coverage is large; keep it
    with outp.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    # confidence hist data
    hist = {
        "refuted_p_ref": [
            float(arts["p_ref"][i]) for i, m in enumerate(arts["meta"])
            if m["state"] == "refuted"
        ],
        "unmentioned_p_ref": [
            float(arts["p_ref"][i]) for i, m in enumerate(arts["meta"])
            if m["state"] == "unmentioned"
        ],
        "not_assessed_p_ref": [
            float(arts["p_ref"][i]) for i, m in enumerate(arts["meta"])
            if m["state"] == "not_assessed"
        ],
    }
    with (OUT / f"conf_hist_{model.replace('/', '_')}{dataset_suffix}.json").open("w") as f:
        json.dump(hist, f)

    # compact print
    print("\n=== Table 1 style ===")
    print(f"Model: {model}")
    print(f"Known Acc: {trap['known_acc']:.3f}")
    print(f"UNR:       {trap['UNR']:.3f}")
    print(f"HCUNR@{tau}: {trap[f'HCUNR@{tau}']:.3f}")
    print(f"Unmentioned UNR: {trap['UNR_by_state']['unmentioned']:.3f}")
    print(f"Unmentioned HCUNR@{tau}: "
          f"{trap[f'HCUNR_by_state@{tau}']['unmentioned']:.3f}")
    print(f"Unk AUROC: {trap['unknown_auroc']:.3f}")
    print(f"ECI:       {geo['ECI']['mean']:.3f} (matched n={geo['ECI']['n']})")
    print(f"Margin means: " + ", ".join(
        f"{s}={geo[s]['mean']:.3f}" for s in STATES if s in geo))
    print(f"UNR by state: {trap['UNR_by_state']}")
    print(f"Oracle gate: unknown recall={mit['oracle_gate']['unknown_recall']:.3f}, "
          f"known acc={mit['oracle_gate']['known_acc']:.3f}")
    print(f"UACR (unsafe clearance): {trap['UACR_unsafe_clearance']:.3f}")
    print(f"Go criteria: {result['go_criteria']}")
    print(f"saved {outp}")
    return result


def outline():
    text = """
SilentEHR — paper outline (EACL/COLING style)

Title
  When Silence Becomes Negative:
  Closed-World Extrapolation with Frozen Patient Embeddings

One-sentence claim
  Closed-world supervision over frozen patient representations can convert
  non-observation into confident clinical absence; geometry, supervision and
  the closed-set decision rule are evaluated separately.

Contributions
  1. Separate representation geometry from closed-world extrapolation.
  2. SilentEHR: matched controlled benchmark (Supported / Refuted / Unknown).
  3. Binary Training Trap: UNR / HCUNR across frozen embedders.
  4. MIMIC structured validation: unmentioned-positive vs unmentioned-negative.
  5. Trial eligibility harms: unsafe auto-clearance on unknowns.
  6. Observation gating: oracle upper bound, predicted gate, and three-way learning.

RQs
  RQ1 Geometry: is Unmentioned closer to Refuted than to Supported?
  RQ2 Trap: do binary classifiers confidently map Unknown → Refuted?
  RQ3 MIMIC: does omission mix true positives and negatives?
  RQ4 Fix: does explicit observation status reduce false certainty?

Main task
  y(x,h) ∈ {Supported, Refuted, Unknown}
  Fine states (Unmentioned, Not assessed, Pending, …) are analysis-only.

Pilot (this repo)
  6 concepts × 20 clean backgrounds × valid family-specific states/templates
  Splits: IID, new-background, new-template, and new-concept reported separately
  Models: hash (smoke) | clinicalbert | qwen | jina
  Expts: margin/ECI, binary trap, abstention AUROC, oracle mention mask

Go / no-go
  Known Acc ≥ 0.80; HCUNR@0.9 ≥ 0.30–0.40 on ≥2 embedders;
  confidence thresholding fails on Unknown; oracle mask drops HCUNR ≥ 0.15
  with Known Acc drop ≤ 0.02.

What the paper is NOT
  Not “embeddings cannot do negation.”
  Not “just add Unknown as a class.”
  It is: silence ≠ evidence of absence under frozen closed-world pipelines.
""".strip()
    print(text)
    p = ROOT / "OUTLINE.md"
    p.write_text(text + "\n", encoding="utf-8")
    print(f"\nwrote {p}")


# ---------------------------------------------------------------------------
# MIMIC structured-grounded validation (RQ3) — discharge notes + hosp labs
# ---------------------------------------------------------------------------
DEFAULT_DISCHARGE = (
    ROOT / "mimic-iv-note-deidentified-free-text-clinical-notes-2.2"
    / "note" / "discharge.csv" / "discharge.csv"
)

MIMIC_CONCEPTS = {
    "hyperkalemia": {
        "itemids": [50971, 50822],
        "agg": "max", "thresh": 5.5, "op": ">",
        "pos_cues": [
            "hyperkalemia", "hyperkalaemia", "hyperkalemic",
            "elevated potassium", "high potassium", "k elevated",
        ],
        "neg_cues": [
            "no hyperkalemia", "without hyperkalemia", "denies hyperkalemia",
            "normal potassium", "potassium normal", "k within normal",
            "normokalemia",
        ],
    },
    "anemia": {
        "itemids": [51222, 50811],
        "agg": "min", "thresh": 10.0, "op": "<",
        "pos_cues": [
            "anemia", "anaemia", "anemic", "anaemic",
            "low hemoglobin", "low haemoglobin", "low hgb",
        ],
        "neg_cues": [
            "no anemia", "no anaemia", "without anemia", "denies anemia",
            "not anemic", "hemoglobin normal", "normal hemoglobin",
        ],
    },
    "thrombocytopenia": {
        "itemids": [51265],
        "agg": "min", "thresh": 100.0, "op": "<",
        "pos_cues": [
            "thrombocytopenia", "thrombocytopenic", "low platelets",
            "low platelet", "platelet count low",
        ],
        "neg_cues": [
            "no thrombocytopenia", "without thrombocytopenia",
            "normal platelets", "platelets normal", "platelet count normal",
        ],
    },
}


def default_notes_path() -> Path | None:
    if DEFAULT_DISCHARGE.exists():
        return DEFAULT_DISCHARGE
    return None


def _read_csv_gz(path: Path, usecols=None, nrows=None):
    import pandas as pd
    return pd.read_csv(path, compression="gzip", usecols=usecols, nrows=nrows,
                       low_memory=False)


def _stream_discharge(notes_path: Path, max_hadm: int):
    """Load complete notes for observation extraction."""
    import pandas as pd
    note_map = {}
    cols = ["hadm_id", "text"]
    for chunk in pd.read_csv(notes_path, usecols=cols, chunksize=5000,
                             low_memory=False):
        for hadm, text in zip(chunk["hadm_id"].tolist(), chunk["text"].tolist()):
            if pd.isna(hadm) or pd.isna(text):
                continue
            hadm = int(hadm)
            if hadm in note_map:
                continue
            t = str(text).strip()
            if len(t) < 200:
                continue
            note_map[hadm] = t
            if len(note_map) >= max_hadm:
                return note_map
    return note_map


def _labevents_filtered(path: Path, hadm_ids: set, itemids: set, chunksize: int = 500_000):
    import pandas as pd
    cols = ["hadm_id", "itemid", "charttime", "valuenum"]
    parts, n, ci = [], 0, 0
    for chunk in pd.read_csv(path, compression="gzip", usecols=cols,
                             chunksize=chunksize, low_memory=False):
        n += len(chunk)
        ci += 1
        sub = chunk[chunk["hadm_id"].isin(hadm_ids) & chunk["itemid"].isin(itemids)]
        if len(sub):
            parts.append(sub)
        if ci % 20 == 0:
            print(f"  scanned ~{n:,} lab rows, kept {sum(len(p) for p in parts):,}")
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)


def _struct_positive(concept: str, value: float) -> bool:
    cfg = MIMIC_CONCEPTS[concept]
    t, op = cfg["thresh"], cfg["op"]
    if op == ">":
        return value > t
    if op == ">=":
        return value >= t
    if op == "<":
        return value < t
    return value <= t


NUMERIC_PATTERNS = {
    "hyperkalemia": {
        "names": r"(?:potassium|K)",
        "patterns": [
            r"(?<!\w)(?:potassium|K)(?!\w)\s*(?:was|is|of|:|=)?\s*"
            r"(\d+(?:\.\d+)?)\s*(?:mEq/?L|mmol/?L)?",
        ],
        "range": (1.0, 10.0),
    },
    "anemia": {
        "names": r"(?:hemoglobin|haemoglobin|Hgb|Hb)",
        "patterns": [
            r"(?<!\w)(?:hemoglobin|haemoglobin|Hgb|Hb)(?!\w)\s*"
            r"(?:was|is|of|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:g/?dL)?",
        ],
        "range": (2.0, 25.0),
    },
    "thrombocytopenia": {
        "names": r"(?:platelets?|platelet count|Plt)",
        "patterns": [
            r"(?<!\w)(?:platelets?|platelet count|Plt)(?!\w)\s*"
            r"(?:was|is|of|:|=)?\s*(\d+(?:\.\d+)?)\s*"
            r"(?:K/?uL|K|x10\^?3/?uL)?",
        ],
        "range": (1.0, 2000.0),
    },
}


def _textual_numeric_evidence(text: str, concept: str) -> dict | None:
    """Extract the most discharge-relevant numeric value from the full note."""
    cfg = NUMERIC_PATTERNS[concept]
    candidates = []
    flags = re.I

    # "improved from 6.1 to 4.2": the destination is the relevant value.
    trend = (
        rf"(?<!\w){cfg['names']}(?!\w).{{0,100}}?"
        r"from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)"
    )
    for m in re.finditer(trend, text, flags=flags):
        value = float(m.group(2))
        if cfg["range"][0] <= value <= cfg["range"][1]:
            candidates.append((4, m.end(), value, m.group(0)))

    arrow = (
        rf"(?<!\w){cfg['names']}(?!\w)\s*(?:was|is|of|:|=)?\s*"
        r"\d+(?:\.\d+)?\s*(?:->|→|to)\s*(\d+(?:\.\d+)?)"
    )
    for m in re.finditer(arrow, text, flags=flags):
        value = float(m.group(1))
        if cfg["range"][0] <= value <= cfg["range"][1]:
            candidates.append((4, m.end(), value, m.group(0)))

    for pattern in cfg["patterns"]:
        for m in re.finditer(pattern, text, flags=flags):
            value = float(m.group(1))
            if not (cfg["range"][0] <= value <= cfg["range"][1]):
                continue
            context = text[max(0, m.start() - 80):m.end() + 80]
            if concept == "hyperkalemia" and re.search(
                r"\b(vitamin\s+K|K/L\s+ratio|potassium\s+\d+(?:\.\d+)?\s*mEq"
                r"|potassium supplement|potassium tablet|hemoly[sz]ed)\b",
                context,
                re.I,
            ):
                continue
            priority = 3 if re.search(
                r"\b(discharge|final|latest|last|on discharge)\b", context, re.I
            ) else 1
            candidates.append((priority, m.end(), value, context))
    if not candidates:
        return None
    # Prefer explicitly discharge/final values; otherwise the last textual value.
    priority = max(c[0] for c in candidates)
    _, pos, value, context = max(
        (c for c in candidates if c[0] == priority), key=lambda c: c[1])
    return {
        "value": value,
        "status": "observed_pos" if _struct_positive(concept, value)
                  else "observed_neg",
        "position": pos,
        "context": re.sub(r"\s+", " ", context).strip()[:240],
    }


MIMIC_STRUCT = ROOT / "data" / "mimic_struct.jsonl"
MIMIC_STRUCT_QWEN3 = ROOT / "data" / "mimic_struct_qwen3.jsonl"
QWEN3_LABELS = ROOT / "data" / "mimic_qwen3_labels.jsonl"
# Backward-compatible aliases (deprecated)
MIMIC_STRUCT_DEEPSEEK = MIMIC_STRUCT_QWEN3
DEEPSEEK_LABELS = QWEN3_LABELS
OBSERVATION_LABEL_MODEL = "qwen3-12b"

QWEN3_TO_MENTION = {
    "observed_positive": "observed_pos",
    "observed_negative": "observed_neg",
    "unmentioned": "unmentioned",
    "historical_only": "observed_historical",
    "admission_only": "observed_historical",
    "uncertain": "observed_uncertain",
    "temporal_conflict": "observed_mixed",
}
DEEPSEEK_TO_MENTION = QWEN3_TO_MENTION


def default_mimic_struct_path() -> Path:
    if not MIMIC_STRUCT_QWEN3.exists():
        raise FileNotFoundError(
            f"{MIMIC_STRUCT_QWEN3} missing; run qwen-label then "
            "build_mimic_qwen3_struct() first.")
    return MIMIC_STRUCT_QWEN3


def observation_cell(structured_positive: bool, mention_status: str) -> str:
    clinical = "C+" if structured_positive else "C-"
    observation = "O-" if mention_status == "unmentioned" else "O+"
    return clinical + observation


def llm_observation_cell(structured_positive: bool, raw_status: str) -> str:
    clinical = "C+" if structured_positive else "C-"
    if raw_status == "unmentioned":
        return clinical + "O-"
    if raw_status in ("observed_positive", "observed_negative"):
        return clinical + "O+"
    return clinical + "O+hist"


def build_mimic_qwen3_struct(
        source: Path | None = None,
        labels_path: Path | None = None,
        output: Path | None = None) -> Path:
    """Overlay Qwen3-12B observation labels onto encoder-visible MIMIC rows."""
    source = source or MIMIC_STRUCT
    labels_path = labels_path or QWEN3_LABELS
    output = output or MIMIC_STRUCT_QWEN3
    if not source.exists():
        raise FileNotFoundError(source)
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    qwen_labels = {}
    for line in labels_path.open(encoding="utf-8"):
        obj = json.loads(line)
        for lab in obj["labels"]:
            qwen_labels[(obj["hadm_id"], lab["concept"])] = lab

    rows_out = []
    missing = 0
    for row in load_rows(source):
        lab = qwen_labels.get((row["hadm_id"], row["concept"]))
        if lab is None:
            missing += 1
            continue
        status = lab["observation_status"]
        if status not in QWEN3_TO_MENTION:
            raise ValueError(f"unknown Qwen3 status: {status}")
        updated = dict(row)
        updated["mention_status"] = QWEN3_TO_MENTION[status]
        updated["qwen3_observation_status"] = status
        updated["qwen3_evidence_span"] = lab.get("evidence_span", "")
        updated["observation_label_source"] = OBSERVATION_LABEL_MODEL
        # Deprecated field names kept for downstream scripts
        updated["deepseek_observation_status"] = status
        updated["deepseek_evidence_span"] = lab.get("evidence_span", "")
        rows_out.append(updated)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows_out:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    cohort = Counter()
    for row in rows_out:
        if row["mention_status"] in (
                "observed_mixed", "observed_uncertain", "observed_historical"):
            continue
        cohort[(row["concept"], row["mention_status"],
                "struct_pos" if row["structured_positive"] else "struct_neg")] += 1
    print(f"wrote {len(rows_out)} Qwen3-labeled rows -> {output}")
    if missing:
        print(f"  skipped {missing} rows without Qwen3 labels")
    print("Qwen3 cohorts (concept, mention, struct):")
    for key, count in sorted(cohort.items()):
        print(f"  {key}: {count}")
    return output


def build_mimic_deepseek_struct(*args, **kwargs) -> Path:
    """Deprecated alias for build_mimic_qwen3_struct."""
    return build_mimic_qwen3_struct(*args, **kwargs)


def _mention_status(text: str, concept: str) -> tuple[str, dict | None]:
    """Span/context heuristic; avoids substring and overlapping negation bugs."""
    cfg = MIMIC_CONCEPTS[concept]
    statuses = []
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.lower())

    def bounded(phrase: str) -> str:
        return rf"(?<!\w){re.escape(phrase)}(?!\w)"

    for sentence in sentences:
        if not sentence:
            continue
        # Explicit normal/negative phrases are classified first and cannot also
        # become affirmative merely because they contain the concept string.
        if any(re.search(bounded(c), sentence) for c in cfg["neg_cues"]):
            statuses.append("observed_neg")
            continue
        for cue in cfg["pos_cues"]:
            for match in re.finditer(bounded(cue), sentence):
                left = sentence[max(0, match.start() - 45):match.start()]
                context = sentence[max(0, match.start() - 80):match.end() + 80]
                if re.search(r"\b(no|not|denies?|denied|without|negative for)\b", left):
                    statuses.append("observed_neg")
                elif re.search(r"\b(possible|possibly|suspected|unclear|question of)\b", context):
                    statuses.append("observed_uncertain")
                elif re.search(
                    r"\b(history of|historical|resolved|corrected|improved|previously|prior)\b",
                    context,
                ):
                    statuses.append("observed_historical")
                else:
                    statuses.append("observed_pos")

    clinical = {s for s in statuses if s in ("observed_pos", "observed_neg")}
    if len(clinical) > 1:
        lexical = "observed_mixed"
    elif clinical:
        lexical = next(iter(clinical))
    elif statuses:
        lexical = statuses[0]
    else:
        lexical = "unmentioned"

    numeric = _textual_numeric_evidence(text, concept)
    if numeric is None:
        return lexical, None
    if lexical in ("observed_pos", "observed_neg") and lexical != numeric["status"]:
        return "observed_mixed", numeric
    # Numeric evidence is an observed assessment even when no diagnostic term
    # appears, and dominates historical/uncertain lexical mentions.
    return numeric["status"], numeric


def mimic_build(mimic_dir: Path, max_hadm: int = 2000,
                notes_path: Path | None = None,
                text_chars: int = 0, window_h: float = 24.0,
                cohort_only: bool = False) -> Path:
    """Join discharge text to the last lab in the final pre-discharge window."""
    import pandas as pd

    notes_path = notes_path or default_notes_path()
    if notes_path is None or not notes_path.exists():
        raise FileNotFoundError("discharge notes not found; pass --notes")

    hosp = mimic_dir / "hosp"
    lab_p = hosp / "labevents.csv.gz"
    adm_p = hosp / "admissions.csv.gz"
    if not lab_p.exists() or not adm_p.exists():
        raise FileNotFoundError(f"need {lab_p} and {adm_p}")

    print(f"streaming discharge notes from {notes_path} (cap {max_hadm}) ...")
    note_map = _stream_discharge(notes_path, max_hadm=max_hadm)
    hadm_ids = set(note_map)
    print(f"  got {len(note_map)} discharge notes")

    print("loading admissions for those hadm ...")
    adm = _read_csv_gz(
        adm_p, usecols=["subject_id", "hadm_id", "admittime", "dischtime"])
    adm["admittime"] = pd.to_datetime(adm["admittime"])
    adm["dischtime"] = pd.to_datetime(adm["dischtime"])
    adm = adm[adm["hadm_id"].isin(hadm_ids)].dropna(subset=["admittime", "dischtime"])
    print(f"  admissions matched: {len(adm)}")

    lab_ids = sorted({i for c in MIMIC_CONCEPTS.values() for i in c["itemids"]})
    print(f"streaming labevents for {len(adm)} hadm / itemids={lab_ids} ...")
    labs = _labevents_filtered(lab_p, set(adm["hadm_id"]), set(lab_ids))
    labs["charttime"] = pd.to_datetime(labs["charttime"])
    labs = labs.dropna(subset=["valuenum", "charttime"])
    labs = labs.merge(adm, on="hadm_id", how="inner")
    window_start = labs["dischtime"] - pd.to_timedelta(window_h, unit="h")
    labs = labs[(labs["charttime"] >= window_start) &
                (labs["charttime"] <= labs["dischtime"])]
    print(f"  final-{window_h:g}h lab rows: {len(labs):,}")

    rows = []
    for concept, cfg in MIMIC_CONCEPTS.items():
        sub = labs[labs["itemid"].isin(cfg["itemids"])]
        if sub.empty:
            print(f"  {concept}: no labs")
            continue
        # Last available value defines near-discharge current status. Using an
        # admission-wide extreme would label resolved transient events positive.
        latest = sub.sort_values("charttime").groupby("hadm_id", as_index=False).tail(1)
        n_c = 0
        for _, r in latest.iterrows():
            hadm = int(r["hadm_id"])
            text = note_map.get(hadm)
            if not text:
                continue
            val = float(r["valuenum"])
            pos = _struct_positive(concept, val)
            # Store the full discharge for encoder chunk-pooling. Long notes are
            # encoded as mean-pooled chunk embeddings rather than truncated.
            visible_text = text if text_chars <= 0 else text[:text_chars]
            rows.append({
                "subject_id": int(r["subject_id"]),
                "hadm_id": hadm,
                "concept": concept,
                "structured_positive": bool(pos),
                "struct_value": float(val),
                "struct_charttime": str(r["charttime"]),
                "dischtime": str(r["dischtime"]),
                "window_h": window_h,
                "full_note_chars": len(text),
                "text": None if cohort_only else visible_text,
                "proposition": {
                    "hyperkalemia": "The patient had hyperkalemia near discharge.",
                    "anemia": "The patient had anemia near discharge.",
                    "thrombocytopenia": "The patient had thrombocytopenia near discharge.",
                }[concept],
            })
            n_c += 1
        print(f"  {concept}: {n_c} pairs")

    out = ROOT / "data" / "mimic_struct.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {out}")

    from collections import Counter
    cohort = Counter()
    for r in rows:
        key = (
            r["concept"],
            "struct_pos" if r["structured_positive"] else "struct_neg",
        )
        cohort[key] += 1
    print("cohorts (concept, structured state):")
    for k, v in sorted(cohort.items()):
        print(f"  {k}: {v}")
    return out


def mimic_delta_recall(struct_path: Path | None = None, model: str = "hash",
                       seed: int = 0):
    """Train on observed mention cases; test unmentioned-pos / unmentioned-neg."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    struct_path = struct_path or default_mimic_struct_path()
    if not struct_path.exists():
        raise FileNotFoundError(
            f"{struct_path} missing; run build_mimic_deepseek_struct() first.")
    if not struct_path.exists():
        raise FileNotFoundError(struct_path)
    rows = [r for r in load_rows(struct_path)
            if r.get("text") and r.get("mention_status") not in (
                None, "observed_mixed", "observed_uncertain", "observed_historical")]
    if len(rows) < 50:
        raise RuntimeError(f"only {len(rows)} labeled rows; rebuild with discharge notes")

    # Subject-level split: all admissions and concepts for one patient stay
    # together. This prevents longitudinal patient leakage.
    subjects = sorted({r["subject_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n_tr = int(0.7 * len(subjects))
    tr_subjects, te_subjects = set(subjects[:n_tr]), set(subjects[n_tr:])

    emb = Embedder(model)
    # cache embedding per hadm (same discharge text reused across concepts)
    uniq = {}
    for r in rows:
        uniq.setdefault(r["hadm_id"], r["text"])

    def is_obs(r):
        return r["mention_status"] in ("observed_pos", "observed_neg")

    def is_clean_obs(r):
        # observed mention agrees with structured label (reduces lexicon noise)
        if r["mention_status"] == "observed_pos" and r["structured_positive"]:
            return True
        if r["mention_status"] == "observed_neg" and not r["structured_positive"]:
            return True
        return False

    # Clean 2×2 design: train only C+O+ and C-O+ agreement cells.
    # Conflicting text/structured cases are excluded rather than relabeled.
    train = [r for r in rows if r["subject_id"] in tr_subjects and is_clean_obs(r)]
    test = [r for r in rows if r["subject_id"] in te_subjects]
    n_pos = sum(r["structured_positive"] for r in train)
    n_neg = len(train) - n_pos
    print(f"train clean C×O: n={len(train)} pos={n_pos} neg={n_neg}")
    if min(n_pos, n_neg) < 20:
        result = {
            "mode": "mimic_discharge_near_discharge",
            "model": model,
            "status": "insufficient_clean_observed_training_data",
            "n_train": len(train),
            "n_train_positive": n_pos,
            "n_train_negative": n_neg,
            "minimum_required_per_class": 20,
            "reason": (
                "Explicit negative mentions are too sparse for a valid binary "
                "training comparison. No classifier metrics were produced."
            ),
        }
        OUT.mkdir(parents=True, exist_ok=True)
        result_path = OUT / f"mimic_delta_recall_{model.replace('/', '_')}.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return result

    # fit TF-IDF on train notes only (avoid leakage)
    if model == "tfidf":
        train_hadm = {r["hadm_id"] for r in rows
                      if r["subject_id"] in tr_subjects}
        emb.fit([uniq[h] for h in train_hadm if h in uniq])

    print(f"encoding {len(uniq)} discharge notes with {model} ...")
    ids = list(uniq.keys())
    Z = emb.encode([uniq[i] for i in ids], batch_size=16)
    Ez = {i: Z[j] for j, i in enumerate(ids)}

    # concept-conditioned pair features when proposition available
    props = {r["proposition"] for r in rows if r.get("proposition")}
    P = {p: emb.encode([p])[0] for p in props} if props else {}

    def feat(r):
        z = Ez[r["hadm_id"]]
        if r.get("proposition") and P:
            return pair_features(z, P[r["proposition"]])
        return z

    Xtr = np.asarray([feat(r) for r in train])
    ytr = np.asarray([1 if r["structured_positive"] else 0 for r in train])

    sc = StandardScaler(with_mean=model != "tfidf")
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    clf.fit(sc.fit_transform(Xtr), ytr)

    def pred_rows(subset):
        if not subset:
            return np.array([])
        return clf.predict(sc.transform(np.asarray([feat(r) for r in subset])))

    def pos_recall(subset):
        pos = [r for r in subset if r["structured_positive"]]
        if not pos:
            return float("nan"), 0
        pr = pred_rows(pos)
        return float(np.mean(pr == 1)), len(pos)

    def neg_rate_as_neg(subset):
        neg = [r for r in subset if not r["structured_positive"]]
        if not neg:
            return float("nan"), 0
        pr = pred_rows(neg)
        return float(np.mean(pr == 0)), len(neg)

    by_c = {}
    for c in MIMIC_CONCEPTS:
        te_c = [r for r in test if r["concept"] == c]
        obs_te = [r for r in te_c if is_obs(r)]
        unm_te = [r for r in te_c if r["mention_status"] == "unmentioned"]
        ro, n_op = pos_recall(obs_te)
        ru, n_up = pos_recall(unm_te)
        rn, n_un = neg_rate_as_neg(unm_te)
        # collapse signal: fraction of ALL unmentioned predicted negative
        if unm_te:
            punm_neg = float(np.mean(pred_rows(unm_te) == 0))
        else:
            punm_neg = float("nan")
        by_c[c] = {
            "observed_pos_recall": ro,
            "unmentioned_pos_recall": ru,
            "gap": (None if (math.isnan(ro) or math.isnan(ru)) else ro - ru),
            "unmentioned_neg_specificity": rn,
            "unmentioned_pred_neg_rate": punm_neg,
            "n_train_clean_obs": sum(1 for r in train if r["concept"] == c),
            "n_test_obs": len(obs_te),
            "n_test_unm": len(unm_te),
            "n_unm_pos": n_up,
            "n_unm_neg": n_un,
        }

    # overall
    obs_te = [r for r in test if is_obs(r)]
    unm_te = [r for r in test if r["mention_status"] == "unmentioned"]
    ro, _ = pos_recall(obs_te)
    ru, n_up = pos_recall(unm_te)
    overall = {
        "observed_pos_recall": ro,
        "unmentioned_pos_recall": ru,
        "gap": (None if (math.isnan(ro) or math.isnan(ru)) else ro - ru),
        "unmentioned_pred_neg_rate": (
            float(np.mean(pred_rows(unm_te) == 0)) if unm_te else float("nan")
        ),
        "n_unmentioned_positive": n_up,
        "n_train": len(train),
        "n_test": len(test),
    }

    # also: among unmentioned, does model ignore structured truth?
    if unm_te and n_up > 0:
        y_unm = np.asarray([1 if r["structured_positive"] else 0 for r in unm_te])
        p_unm = pred_rows(unm_te)
        overall["unmentioned_balanced_acc"] = float(
            0.5 * (np.mean(p_unm[y_unm == 1] == 1) if (y_unm == 1).any() else 0)
            + 0.5 * (np.mean(p_unm[y_unm == 0] == 0) if (y_unm == 0).any() else 0)
        )

    out = {
        "mode": "mimic_discharge",
        "model": model,
        "n_notes": len(uniq),
        "overall": overall,
        "by_concept": by_c,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"mimic_delta_recall_{model.replace('/', '_')}.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


def mimic_embedding_batch(model: str, device: str | None) -> int:
    """Conservative batch sizes for long discharge-note encoding on laptop GPUs."""
    if not device or not str(device).startswith("cuda"):
        return 16
    if model in ("qwen", "qwen3"):
        return 8
    if model in ("jina", "jina-v5", "biolord", "med-gte"):
        return 16
    return 32


def mimic_cxo_experiment(struct_path: Path | None = None, model: str = "hash",
                         seed: int = 0, cap_per_cell: int = 0,
                         device: str | None = None) -> dict:
    """Subject-split C×O experiment: per-concept, joint, and prior controls."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    struct_path = struct_path or default_mimic_struct_path()
    if not struct_path.exists():
        raise FileNotFoundError(
            f"{struct_path} missing; run build_mimic_deepseek_struct() first.")
    rows = [r for r in load_rows(struct_path)
            if r.get("text") and "subject_id" in r
            and r.get("mention_status") not in (
                None, "observed_mixed", "observed_uncertain",
                "observed_historical")]
    if not rows:
        raise RuntimeError(
            "No encoder-visible subject-labeled rows. Rebuild without --cohort-only.")

    subjects = sorted({r["subject_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(subjects)
    cut = int(0.7 * len(subjects))
    train_subjects, test_subjects = set(subjects[:cut]), set(subjects[cut:])

    def clean_observed(r):
        return (
            r["mention_status"] == "observed_pos" and r["structured_positive"]
        ) or (
            r["mention_status"] == "observed_neg"
            and not r["structured_positive"]
        )

    train = [r for r in rows
             if r["subject_id"] in train_subjects and clean_observed(r)]
    test = [r for r in rows if r["subject_id"] in test_subjects
            and (clean_observed(r) or r["mention_status"] == "unmentioned")]
    eligible = []
    for concept in sorted({r["concept"] for r in rows}):
        tr = [r for r in train if r["concept"] == concept]
        n_pos = sum(r["structured_positive"] for r in tr)
        if min(n_pos, len(tr) - n_pos) >= 20:
            eligible.append(concept)
    if not eligible:
        raise RuntimeError("No concept has >=20 clean observed cases per class.")
    train = [r for r in train if r["concept"] in eligible]
    test = [r for r in test if r["concept"] in eligible]

    if cap_per_cell > 0:
        def cap(rows_in, is_train):
            cells = defaultdict(list)
            for r in rows_in:
                observation = "observed" if clean_observed(r) else "unmentioned"
                key = (r["concept"], observation, r["structured_positive"])
                cells[key].append(r)
            sampled = []
            local_rng = random.Random(seed + (0 if is_train else 1))
            for values in cells.values():
                local_rng.shuffle(values)
                sampled.extend(values[:cap_per_cell])
            return sampled
        train = cap(train, True)
        test = cap(test, False)

    notes = {}
    for r in train + test:
        notes.setdefault(r["hadm_id"], r["text"])
    emb = Embedder(model, device=device)
    if model == "tfidf":
        train_hadm = {r["hadm_id"] for r in train}
        emb.fit([notes[h] for h in train_hadm])
    ids = list(notes)
    print(f"encoding {len(ids)} encoder-visible MIMIC notes with {model} ...")
    embedding_batch = mimic_embedding_batch(model, device)
    mimic_texts = [notes[i] for i in ids]
    cache_dir = OUT / "embedding_cache"
    fingerprint = hashlib.blake2b(
        (f"mimic\0{model}\0chunk{MIMIC_CHUNK_CHARS}\0"
         + "\0".join(mimic_texts)).encode("utf-8"),
        digest_size=12,
    ).hexdigest()
    cache_path = cache_dir / (
        f"mimic_{model.replace('/', '_')}_{fingerprint}.npy")
    if model != "tfidf" and cache_path.exists():
        Z = np.load(cache_path)
        print(f"loaded embedding cache {cache_path}")
    else:
        print(f"chunk-pooled encoding (chunk_chars={MIMIC_CHUNK_CHARS}) ...",
              flush=True)
        Z = emb.encode(
            mimic_texts, batch_size=embedding_batch,
            chunk_chars=MIMIC_CHUNK_CHARS)
        if model != "tfidf":
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, Z)
            print(f"saved embedding cache {cache_path}")
    Ez = {hadm: Z[i] for i, hadm in enumerate(ids)}
    props = {r["proposition"] for r in train + test}
    P = {p: emb.encode([p])[0] for p in props}

    def features(subset, proposition_only=False):
        if proposition_only:
            return np.asarray([P[r["proposition"]] for r in subset])
        return np.asarray([
            pair_features(Ez[r["hadm_id"]], P[r["proposition"]])
            for r in subset
        ])

    def summarize_core(subset, p_pos):
        y = np.asarray([int(r["structured_positive"]) for r in subset])
        observed_pos = np.asarray([
            r["structured_positive"] and r["mention_status"] == "observed_pos"
            for r in subset])
        undocumented = np.asarray([
            r["mention_status"] == "unmentioned" for r in subset])
        undocumented_pos = undocumented & (y == 1)
        recall_obs = float(np.mean(p_pos[observed_pos] >= 0.5)) \
            if observed_pos.any() else None
        recall_unm = float(np.mean(p_pos[undocumented_pos] >= 0.5)) \
            if undocumented_pos.any() else None
        unm_y = y[undocumented]
        unm_p = p_pos[undocumented]
        auroc = float(roc_auc_score(unm_y, unm_p)) \
            if len(np.unique(unm_y)) == 2 else None
        observed = np.asarray([
            r["mention_status"] in ("observed_pos", "observed_neg")
            for r in subset])
        known_or_unm = observed | undocumented
        epistemic_y = undocumented[known_or_unm].astype(int)
        epistemic_score = 1 - np.maximum(
            p_pos[known_or_unm], 1 - p_pos[known_or_unm])
        epistemic_auroc = float(roc_auc_score(
            epistemic_y, epistemic_score
        )) if len(np.unique(epistemic_y)) == 2 else None
        return {
            "observed_positive_recall": recall_obs,
            "undocumented_positive_recall": recall_unm,
            "UPFNR": None if recall_unm is None else 1 - recall_unm,
            "HCWE@0.9": (
                float(np.mean(p_pos[undocumented_pos] < 0.1))
                if undocumented_pos.any() else None
            ),
            "documentation_gap": (
                recall_obs - recall_unm
                if recall_obs is not None and recall_unm is not None else None
            ),
            "unmentioned_clinical_state_AUROC": auroc,
            "unmentioned_detection_AUROC": epistemic_auroc,
            "unmentioned_predicted_negative_rate": float(np.mean(
                unm_p < 0.5)) if undocumented.any() else None,
            "n_unmentioned_positive": int(undocumented_pos.sum()),
            "n_unmentioned_negative": int((undocumented & (y == 0)).sum()),
        }

    def summarize(subset, p_pos, n_boot=500):
        result = summarize_core(subset, p_pos)
        subject_indices = defaultdict(list)
        for i, row in enumerate(subset):
            subject_indices[row["subject_id"]].append(i)
        subjects = list(subject_indices)
        rng = np.random.default_rng(seed)
        boot_values = defaultdict(list)
        metrics = (
            "UPFNR", "HCWE@0.9", "documentation_gap",
            "unmentioned_clinical_state_AUROC",
            "unmentioned_detection_AUROC",
            "unmentioned_predicted_negative_rate",
        )
        if len(subjects) >= 2:
            for _ in range(n_boot):
                sampled = rng.choice(subjects, size=len(subjects), replace=True)
                idx = np.asarray([
                    i for subject in sampled
                    for i in subject_indices[subject]
                ], dtype=int)
                boot = summarize_core(
                    [subset[i] for i in idx], p_pos[idx])
                for metric in metrics:
                    if boot[metric] is not None:
                        boot_values[metric].append(boot[metric])
        result["subject_bootstrap_95CI"] = {
            metric: {
                "lower": float(np.quantile(values, 0.025)),
                "upper": float(np.quantile(values, 0.975)),
                "n_boot": len(values),
            }
            for metric, values in boot_values.items()
            if values
        }
        result["n_test_subjects"] = len(subjects)
        return result

    def fit_predict(train_rows, test_rows, proposition_only=False):
        Xtr = features(train_rows, proposition_only)
        Xte = features(test_rows, proposition_only)
        ytr = np.asarray([int(r["structured_positive"]) for r in train_rows])
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=seed)
        clf.fit(scaler.transform(Xtr), ytr)
        return clf.predict_proba(scaler.transform(Xte))[
            :, list(clf.classes_).index(1)]

    by_concept = {}
    for concept in eligible:
        tr = [r for r in train if r["concept"] == concept]
        te = [r for r in test if r["concept"] == concept]
        p = fit_predict(tr, te)
        by_concept[concept] = {
            **summarize(te, p),
            "n_train_positive": sum(r["structured_positive"] for r in tr),
            "n_train_negative": sum(not r["structured_positive"] for r in tr),
        }

    joint_p = fit_predict(train, test)
    prop_p = fit_predict(train, test, proposition_only=True)
    prevalence = {}
    for concept in eligible:
        tr = [r for r in train if r["concept"] == concept]
        prevalence[concept] = float(np.mean(
            [r["structured_positive"] for r in tr]))
    prior_p = np.asarray([prevalence[r["concept"]] for r in test])
    out = {
        "mode": "mimic_subject_split_encoder_visible_CxO",
        "encoder_input": "full_discharge_chunk_pooled",
        "chunk_chars": MIMIC_CHUNK_CHARS,
        "observation_label_source": rows[0].get(
            "observation_label_source", OBSERVATION_LABEL_MODEL),
        "model": model,
        "cap_per_cell": cap_per_cell,
        "eligible_concepts": eligible,
        "by_concept": by_concept,
        "joint_query_conditioned": summarize(test, joint_p),
        "proposition_only_control": summarize(test, prop_p),
        "concept_prior_control": summarize(test, prior_p),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    sample_suffix = (
        "_matched" if model == "tfidf" and cap_per_cell > 0 else "")
    path = OUT / (
        f"mimic_cxo_{model.replace('/', '_')}{sample_suffix}.json")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


def make_mimic_audit_sample(n_per_cell: int = 25, seed: int = 42) -> Path:
    """Create a blinded, stratified sheet for validating MIMIC observation labels."""
    source = default_mimic_struct_path()
    rows = load_rows(source)
    cells = defaultdict(list)
    for row in rows:
        if row["concept"] not in ("anemia", "thrombocytopenia"):
            continue
        if row["mention_status"] not in (
                "observed_pos", "observed_neg", "unmentioned"):
            continue
        key = (
            row["concept"], row["mention_status"],
            bool(row["structured_positive"]),
        )
        cells[key].append(row)

    rng = random.Random(seed)
    sampled = []
    for key in sorted(cells):
        candidates = cells[key][:]
        rng.shuffle(candidates)
        sampled.extend(candidates[:n_per_cell])
    rng.shuffle(sampled)

    output = ROOT / "data" / "mimic_audit_300.csv"
    key_output = ROOT / "data" / "mimic_audit_300_key.csv"
    fields = [
        "audit_id", "concept", "proposition",
        "struct_value", "struct_charttime", "dischtime",
        "full_note_chars", "encoder_visible_text",
        "annotator_1_cell", "annotator_2_cell", "adjudicated_cell",
        "implicit_evidence_present", "temporal_conflict_present",
        "error_category", "reviewer_comments",
    ]
    key_fields = [
        "audit_id", "automatic_cell", "structured_positive",
        "automatic_mention_status", "observation_label_model",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as handle, \
            key_output.open(
                "w", encoding="utf-8-sig", newline="") as key_handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        key_writer = csv.DictWriter(key_handle, fieldnames=key_fields)
        writer.writeheader()
        key_writer.writeheader()
        for i, row in enumerate(sampled, 1):
            audit_id = f"A{i:03d}"
            automatic_cell = observation_cell(
                row["structured_positive"], row["mention_status"])
            writer.writerow({
                "audit_id": audit_id,
                "concept": row["concept"],
                "proposition": row["proposition"],
                "struct_value": row["struct_value"],
                "struct_charttime": row["struct_charttime"],
                "dischtime": row["dischtime"],
                "full_note_chars": row["full_note_chars"],
                "encoder_visible_text": row["text"],
                "annotator_1_cell": "",
                "annotator_2_cell": "",
                "adjudicated_cell": "",
                "implicit_evidence_present": "",
                "temporal_conflict_present": "",
                "error_category": "",
                "reviewer_comments": "",
            })
            key_writer.writerow({
                "audit_id": audit_id,
                "automatic_cell": automatic_cell,
                "structured_positive": row["structured_positive"],
                "automatic_mention_status": row["mention_status"],
                "observation_label_model": OBSERVATION_LABEL_MODEL,
            })
    print(f"wrote {len(sampled)} blinded audit cases -> {output}")
    print(f"wrote hidden scoring key -> {key_output}")
    return output


def score_mimic_audit() -> dict:
    """Score adjudicated human audit cells against LLM observation labels."""
    from sklearn.metrics import cohen_kappa_score

    source = ROOT / "data" / "mimic_audit_300.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    human_path = ROOT / "data" / "human_annotator1.xlsx"
    if human_path.exists():
        import pandas as pd
        human = pd.read_excel(
            human_path, sheet_name="mimic_audit_300_annotated")
        human_by_id = {
            str(row["audit_id"]).strip(): {
                "adjudicated_cell": str(row["adjudicated_cell"]).strip(),
                "error_category": str(row.get("error_category", "")).strip(),
            }
            for _, row in human.iterrows()
        }
        for row in rows:
            if row["audit_id"] in human_by_id:
                info = human_by_id[row["audit_id"]]
                row["adjudicated_cell"] = info["adjudicated_cell"]
                row["error_category"] = info["error_category"]
                if not row.get("annotator_1_cell"):
                    row["annotator_1_cell"] = info["adjudicated_cell"]

    key_path = ROOT / "data" / "mimic_audit_300_key.csv"
    key = {}
    if key_path.exists():
        key = {
            item["audit_id"]: item
            for item in csv.DictReader(key_path.open(encoding="utf-8-sig"))
        }

    struct_index = defaultdict(list)
    for struct_row in load_rows(default_mimic_struct_path()):
        if struct_row["concept"] not in ("anemia", "thrombocytopenia"):
            continue
        struct_float = (
            float(struct_row["struct_value"])
            if struct_row.get("struct_value") not in (None, "") else None
        )
        match_key = (
            struct_row["concept"],
            struct_float,
            struct_row.get("struct_charttime"),
            struct_row.get("dischtime"),
            bool(struct_row["structured_positive"]),
        )
        struct_index[match_key].append(struct_row)

    valid = {"C+O+", "C-O+", "C+O-", "C-O-", "exclude"}
    for row in rows:
        struct_float = (
            float(row["struct_value"]) if row.get("struct_value") else None
        )
        if row["audit_id"] not in key:
            raise RuntimeError(f"{row['audit_id']}: missing audit key row")
        match_key = (
            row["concept"],
            struct_float,
            row.get("struct_charttime"),
            row.get("dischtime"),
            key[row["audit_id"]]["structured_positive"] == "True",
        )
        candidates = struct_index.get(match_key, [])
        if len(candidates) != 1:
            raise RuntimeError(
                f"{row['audit_id']}: expected 1 struct match, "
                f"got {len(candidates)}")
        struct_row = candidates[0]
        raw_status = struct_row.get("deepseek_observation_status", "unmentioned")
        row["automatic_cell"] = llm_observation_cell(
            struct_row["structured_positive"], raw_status)
        row["llm_observation_status"] = raw_status
        row["automatic_mention_status"] = struct_row["mention_status"]
        row["structured_positive"] = struct_row["structured_positive"]

    complete = [
        row for row in rows
        if row.get("adjudicated_cell") in valid
    ]
    if not complete:
        raise RuntimeError(
            "No adjudicated rows found. Fill adjudicated_cell first.")
    for row in complete:
        if row["adjudicated_cell"] not in valid:
            raise ValueError(
                f"{row['audit_id']}: invalid adjudicated_cell="
                f"{row['adjudicated_cell']!r}")

    auto_valid = [row for row in complete if row["adjudicated_cell"] != "exclude"]
    auto_correct = [
        row["automatic_cell"] == row["adjudicated_cell"]
        for row in auto_valid
    ]
    auto_unmentioned = [
        row for row in auto_valid if row["automatic_cell"].endswith("O-")
    ]
    clinical_correct = [
        row["automatic_cell"][:2] == row["adjudicated_cell"][:2]
        for row in auto_valid
    ]
    error_counts = defaultdict(int)
    for row in complete:
        category = (row.get("error_category") or "").strip() or "none"
        error_counts[category] += 1

    by_concept = {}
    for concept in ("anemia", "thrombocytopenia"):
        sub = [row for row in auto_valid if row["concept"] == concept]
        if sub:
            by_concept[concept] = float(np.mean([
                row["automatic_cell"] == row["adjudicated_cell"]
                for row in sub
            ]))

    result = {
        "observation_label_model": OBSERVATION_LABEL_MODEL,
        "n_total": len(rows),
        "n_fully_annotated": len(complete),
        "n_evaluated": len(auto_valid),
        "n_excluded": sum(
            row["adjudicated_cell"] == "exclude" for row in complete),
        "cell_assignment_accuracy": float(np.mean(auto_correct)),
        "cohen_kappa": float(cohen_kappa_score(
            [row["automatic_cell"] for row in auto_valid],
            [row["adjudicated_cell"] for row in auto_valid],
        )) if len(auto_valid) >= 2 else None,
        "O_only_accuracy": float(np.mean([
            row["automatic_cell"][2:] == row["adjudicated_cell"][2:]
            for row in auto_valid
        ])),
        "O_minus_precision": float(np.mean([
            row["adjudicated_cell"].endswith("O-")
            for row in auto_unmentioned
        ])) if auto_unmentioned else None,
        "structured_C_accuracy": float(np.mean(clinical_correct)),
        "by_concept_accuracy": by_concept,
        "excluded_after_adjudication": sum(
            row["adjudicated_cell"] == "exclude" for row in complete),
        "error_categories": dict(sorted(error_counts.items())),
    }
    output = OUT / "mimic_audit_metrics.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved {output}")
    return result


DEEPSEEK_OBSERVATION_PROMPT = """You label observation status in de-identified
clinical discharge-note text. Use only the supplied TEXT; do not infer from
outside knowledge or assume that silence is negative.

The proposition concerns the patient's state near discharge. For each requested
concept, return exactly one label:
- observed_positive: near-discharge/current text supports the abnormal state.
- observed_negative: near-discharge/current text supports the normal/absent state.
- uncertain: current evidence is explicitly uncertain or incomplete.
- historical_only: evidence is only historical or clearly resolved, without a
  usable near-discharge state.
- admission_only: evidence is explicitly limited to admission/early stay.
- temporal_conflict: incompatible current/near-discharge evidence cannot be resolved.
- unmentioned: no usable textual evidence for the proposition.

Numeric thresholds: anemia = hemoglobin/Hgb < 10; thrombocytopenia =
platelets/Plt < 100; hyperkalemia = potassium/K > 5.5. A value in a
"discharge labs", "latest", or "final" section is near-discharge evidence.
Admission values alone are not near-discharge evidence. Explicit normalization,
correction, persistence, or resolution is usable evidence. Recognize tables,
abbreviations, negation, and numeric evidence even when the diagnosis word is
absent.

Return a JSON object only:
{"labels":[{"concept":"anemia","observation_status":"unmentioned",
"temporal_scope":"none","evidence_span":"","implicit_evidence":false}]}
Evidence span must be a short exact quote from TEXT, or empty when no evidence.
Return one item for every requested concept and no other concepts."""

OBSERVATION_PROMPT = DEEPSEEK_OBSERVATION_PROMPT


def qwen_relabel_mimic(limit: int = 0, workers: int = 1, device: str | None = None) -> Path:
    """Relabel encoder-visible MIMIC observation status with Qwen3-12B-Instruct."""
    from supplementary.data_generation.qwen_observation_labeler import relabel_mimic_notes

    return relabel_mimic_notes(limit=limit, device=device)


def deepseek_relabel_mimic(limit: int = 0, workers: int = 32) -> Path:
    """Deprecated alias — uses Qwen3-12B local labeling."""
    return qwen_relabel_mimic(limit=limit, workers=workers)


SUITE_MODELS = [
    "tfidf", "clinicalbert", "qwen", "jina",
    "biolord", "medcpt", "med-gte",
]

MIMIC_EMBED_ORDER = [
    "tfidf", "clinicalbert", "jina", "biolord", "medcpt", "med-gte", "qwen",
]


def mimic_output_path(model: str, cap_per_cell: int) -> Path:
    suffix = "_matched" if model == "tfidf" and cap_per_cell > 0 else ""
    return OUT / f"mimic_cxo_{model.replace('/', '_')}{suffix}.json"


def _mimic_result_row(model: str, result: dict | None, output: Path,
                      status: str, error: str | None = None) -> dict:
    joint = (result or {}).get("joint_query_conditioned", {})
    boot = joint.get("subject_bootstrap_95CI", {})
    return {
        "model": model,
        "status": status,
        "finished_at": datetime.now().isoformat(),
        "output": str(output),
        "eligible_concepts": (result or {}).get("eligible_concepts"),
        "observation_label_source": (result or {}).get(
            "observation_label_source"),
        "encoder_input": (result or {}).get("encoder_input"),
        "chunk_chars": (result or {}).get("chunk_chars"),
        "UPFNR": joint.get("UPFNR"),
        "HCWE@0.9": joint.get("HCWE@0.9"),
        "documentation_gap": joint.get("documentation_gap"),
        "unmentioned_detection_AUROC": joint.get("unmentioned_detection_AUROC"),
        "UPFNR_CI": boot.get("UPFNR"),
        "error": error,
    }


def run_mimic_pipeline(
        mimic_dir: Path | None = None,
        max_hadm: int = 20000,
        mimic_cap_per_cell: int = 500,
        relabel: bool = True,
        skip_rebuild: bool = False,
        skip_relabel: bool = False,
        force_models: bool = False,
        workers: int = 32,
        device: str | None = "cuda") -> dict:
    """Rebuild full-note MIMIC cohort, Qwen3-12B labels, then all embedders."""
    mimic_dir = mimic_dir or Path(r"D:\mimic-iv-3.1")
    OUT.mkdir(parents=True, exist_ok=True)
    logs = OUT / "pipeline_logs"
    logs.mkdir(exist_ok=True)
    status_path = OUT / "mimic_pipeline_status.json"
    results_path = OUT / "mimic_pipeline_results.jsonl"

    completed_models = set()
    if results_path.exists() and not force_models:
        with results_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("status") == "completed":
                    completed_models.add(row["model"])

    status = {
        "started_at": datetime.now().isoformat(),
        "phase": "init",
        "encoder_order": MIMIC_EMBED_ORDER,
        "observation_labels": OBSERVATION_LABEL_MODEL,
        "encoder_input": "full_discharge_chunk_pooled",
        "chunk_chars": MIMIC_CHUNK_CHARS,
        "cap_per_cell": mimic_cap_per_cell,
        "jobs": [],
    }

    def save_status(phase: str | None = None):
        if phase:
            status["phase"] = phase
        status["updated_at"] = datetime.now().isoformat()
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    def append_result(row: dict):
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        status["jobs"].append(row)
        save_status()

    save_status("init")

    if not skip_rebuild:
        save_status("rebuild_struct")
        notes = default_notes_path()
        if notes is None or not notes.exists():
            raise FileNotFoundError("discharge notes not found")
        mimic_build(
            mimic_dir, max_hadm=max_hadm, notes_path=notes,
            text_chars=0, cohort_only=False)
    else:
        print(f"skip rebuild; reusing {MIMIC_STRUCT}", flush=True)

    if relabel and not skip_relabel:
        save_status("qwen_relabel")
        if QWEN3_LABELS.exists():
            backup = OUT / (
                f"mimic_qwen3_labels_{datetime.now():%Y%m%d_%H%M%S}.jsonl.bak")
            shutil.move(QWEN3_LABELS, backup)
            print(f"archived prior Qwen3 labels -> {backup}", flush=True)
        qwen_relabel_mimic(workers=workers)
    elif skip_relabel:
        print("skip Qwen3 relabel", flush=True)

    save_status("build_qwen3_struct")
    build_mimic_qwen3_struct()

    for model in MIMIC_EMBED_ORDER:
        if model in completed_models and not force_models:
            print(f"skip completed model: {model}", flush=True)
            continue
        cap = mimic_cap_per_cell
        output = mimic_output_path(model, cap)
        log_path = logs / f"mimic_cxo_{model}.log"
        record = {
            "model": model,
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "output": str(output),
        }
        save_status(f"mimic_cxo/{model}")
        print(f"start mimic_cxo / {model}", flush=True)
        dev = None if model == "tfidf" else device
        cmd = [
            sys.executable, str(Path(__file__)), "mimic",
            "--reuse", "--model", model,
            "--mimic-cap-per-cell", str(cap),
        ]
        if dev:
            cmd.extend(["--device", dev])
        try:
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"exit code {completed.returncode}; see {log_path}")
            result = json.loads(output.read_text(encoding="utf-8"))
            row = _mimic_result_row(model, result, output, "completed")
            append_result(row)
            print(f"completed mimic_cxo / {model}", flush=True)
        except Exception as exc:
            row = _mimic_result_row(
                model, None, output, "failed", error=str(exc))
            append_result(row)
            print(f"failed mimic_cxo / {model}: {exc}", flush=True)

    status["finished_at"] = datetime.now().isoformat()
    save_status("done")
    print(json.dumps(status, indent=2))
    return status


def run_suite(mimic_cap_per_cell: int = 500, force: bool = False,
              mimic_only: bool = False) -> dict:
    """Run the paper model matrix serially, resuming from result files."""
    OUT.mkdir(parents=True, exist_ok=True)
    build_mimic_deepseek_struct()
    logs = OUT / "suite_logs"
    logs.mkdir(exist_ok=True)
    status_path = OUT / "suite_status.json"
    status = {
        "started_at": datetime.now().isoformat(),
        "models": SUITE_MODELS,
        "observation_labels": OBSERVATION_LABEL_MODEL,
        "jobs": [],
    }

    jobs = []
    if not mimic_only:
        for model in SUITE_MODELS:
            jobs.append({
                "stage": "controlled_full",
                "model": model,
                "output": OUT / f"results_{model}_full.json",
                "command": [
                    sys.executable, str(Path(__file__)), "run",
                    "--model", model, "--full",
                    *(["--device", "cuda"] if model != "tfidf" else []),
                ],
            })
    for model in SUITE_MODELS:
        cap = 0 if model == "tfidf" else mimic_cap_per_cell
        out_name = f"mimic_cxo_{model}.json"
        if model == "tfidf" and cap > 0:
            out_name = f"mimic_cxo_{model}_matched.json"
        jobs.append({
            "stage": "mimic_cxo",
            "model": model,
            "output": OUT / out_name,
            "command": [
                sys.executable, str(Path(__file__)), "mimic",
                "--reuse", "--model", model,
                "--mimic-cap-per-cell", str(cap),
                *(["--device", "cuda"] if model != "tfidf" else []),
            ],
        })

    for job in jobs:
        record = {
            "stage": job["stage"],
            "model": job["model"],
            "output": str(job["output"]),
        }
        if job["output"].exists() and not force:
            record["status"] = "skipped_existing"
            status["jobs"].append(record)
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            print(f"skip existing: {job['stage']} / {job['model']}",
                  flush=True)
            continue

        record["started_at"] = datetime.now().isoformat()
        log_path = logs / f"{job['stage']}_{job['model']}.log"
        print(f"start: {job['stage']} / {job['model']}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                job["command"], cwd=ROOT, stdout=log,
                stderr=subprocess.STDOUT, check=False)
        record["return_code"] = completed.returncode
        record["status"] = "completed" if completed.returncode == 0 else "failed"
        record["finished_at"] = datetime.now().isoformat()
        record["log"] = str(log_path)
        status["jobs"].append(record)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"{record['status']}: {job['stage']} / {job['model']}",
              flush=True)

    status["finished_at"] = datetime.now().isoformat()
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def run_artifact_suite(models: list[str] | None = None, force: bool = False,
                       device: str = "cuda") -> dict:
    """Run controlled artifact-resistant tiers for the headline model set."""
    OUT.mkdir(parents=True, exist_ok=True)
    make_artifact_tiers()
    logs = OUT / "artifact_logs"
    logs.mkdir(exist_ok=True)
    status_path = OUT / "artifact_suite_status.json"
    models = models or SUITE_MODELS
    status = {
        "started_at": datetime.now().isoformat(),
        "models": models,
        "tiers": list(ARTIFACT_TIERS),
        "jobs": [],
    }
    jobs = []
    for tier, data_path in ARTIFACT_TIERS.items():
        for model in models:
            suffix = result_suffix(data_path)
            jobs.append({
                "tier": tier,
                "model": model,
                "output": OUT / f"results_{model}{suffix}.json",
                "command": [
                    sys.executable, str(Path(__file__)), "run",
                    "--model", model,
                    "--data-path", str(data_path),
                    *(["--device", device] if model != "tfidf" else []),
                ],
            })

    for job in jobs:
        record = {
            "tier": job["tier"],
            "model": job["model"],
            "output": str(job["output"]),
        }
        if job["output"].exists() and not force:
            record["status"] = "skipped_existing"
            status["jobs"].append(record)
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            print(f"skip existing: {job['tier']} / {job['model']}", flush=True)
            continue

        record["started_at"] = datetime.now().isoformat()
        log_path = logs / f"{job['tier']}_{job['model']}.log"
        print(f"start: {job['tier']} / {job['model']}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                job["command"], cwd=ROOT, stdout=log,
                stderr=subprocess.STDOUT, check=False)
        record["return_code"] = completed.returncode
        record["status"] = "completed" if completed.returncode == 0 else "failed"
        record["finished_at"] = datetime.now().isoformat()
        record["log"] = str(log_path)
        status["jobs"].append(record)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"{record['status']}: {job['tier']} / {job['model']}", flush=True)

    status["finished_at"] = datetime.now().isoformat()
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "cmd", choices=[
            "make", "make-artifacts", "run", "outline", "mimic", "audit",
            "audit-score", "qwen-label", "deepseek-label", "mimic-pipeline", "suite",
            "artifact-suite", "all"])
    ap.add_argument("--model", default="hash",
                    help="hash | tfidf | clinicalbert | qwen | jina")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--robustness", action="store_true",
                    help="run LR/PCA/calibration/centroid/kNN checks")
    ap.add_argument("--full", action="store_true",
                    help="use 80-background/16-concept controlled benchmark")
    ap.add_argument("--data-path", default=None,
                    help="controlled JSONL path for run/artifact experiments")
    ap.add_argument("--mimic-dir", default=r"D:\mimic-iv-3.1")
    ap.add_argument("--notes", default=None,
                    help="discharge csv; default: local MIMIC-IV-Note path")
    ap.add_argument("--max-hadm", type=int, default=2000)
    ap.add_argument("--text-chars", type=int, default=0,
                    help="optional char cap for stored note text; 0=full discharge")
    ap.add_argument("--window-hours", type=float, default=24.0,
                    help="pre-discharge structured-label window")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse data/mimic_struct.jsonl; skip rebuild")
    ap.add_argument("--cohort-only", action="store_true",
                    help="build/count C×O cells without embedding or classifier")
    ap.add_argument("--mimic-cap-per-cell", type=int, default=0,
                    help="optional balanced C×O cell cap for expensive embedders")
    ap.add_argument("--workers", type=int, default=1,
                    help="unused (local Qwen3-12B); kept for CLI compatibility")
    ap.add_argument("--limit", type=int, default=0,
                    help="maximum new notes for qwen-label (0 = all)")
    ap.add_argument("--force", action="store_true",
                    help="rerun suite jobs even if outputs exist")
    ap.add_argument("--mimic-only", action="store_true",
                    help="suite: rerun only Qwen3-labeled MIMIC experiments")
    ap.add_argument("--skip-rebuild", action="store_true",
                    help="mimic-pipeline: reuse existing mimic_struct.jsonl")
    ap.add_argument("--skip-relabel", action="store_true",
                    help="mimic-pipeline: skip Qwen3 relabeling")
    ap.add_argument("--force-models", action="store_true",
                    help="mimic-pipeline: rerun embedders even if recorded done")
    args = ap.parse_args()
    if args.cmd == "make":
        make_pilot(full=args.full)
    elif args.cmd == "make-artifacts":
        make_artifact_tiers()
    elif args.cmd == "outline":
        outline()
    elif args.cmd == "run":
        if args.data_path:
            data_path = Path(args.data_path)
        else:
            data_path = FULL_DATA if args.full else DATA
        run(args.model, device=args.device, tau=args.tau,
            robustness=args.robustness, data_path=data_path)
    elif args.cmd == "mimic":
        struct = MIMIC_STRUCT
        if not (args.reuse and struct.exists()):
            notes = Path(args.notes) if args.notes else default_notes_path()
            mimic_build(Path(args.mimic_dir), max_hadm=args.max_hadm,
                        notes_path=notes, window_h=args.window_hours,
                        text_chars=args.text_chars,
                        cohort_only=args.cohort_only)
        else:
            print(f"reusing {struct}")
        build_mimic_qwen3_struct()
        if not args.cohort_only:
            mimic_cxo_experiment(
                struct_path=MIMIC_STRUCT_QWEN3,
                model=args.model, cap_per_cell=args.mimic_cap_per_cell,
                device=args.device)
    elif args.cmd == "suite":
        run_suite(mimic_cap_per_cell=args.mimic_cap_per_cell or 500,
                  force=args.force, mimic_only=args.mimic_only)
    elif args.cmd == "artifact-suite":
        run_artifact_suite(force=args.force, device=args.device or "cuda")
    elif args.cmd == "audit":
        make_mimic_audit_sample()
    elif args.cmd == "audit-score":
        score_mimic_audit()
    elif args.cmd in ("qwen-label", "deepseek-label"):
        qwen_relabel_mimic(limit=args.limit, workers=args.workers, device=args.device)
    elif args.cmd == "mimic-pipeline":
        if args.force_models and (OUT / "mimic_pipeline_results.jsonl").exists():
            (OUT / "mimic_pipeline_results.jsonl").unlink()
        run_mimic_pipeline(
            mimic_dir=Path(args.mimic_dir),
            max_hadm=max(args.max_hadm, 20000),
            mimic_cap_per_cell=args.mimic_cap_per_cell or 500,
            relabel=not args.skip_relabel,
            skip_rebuild=args.skip_rebuild,
            skip_relabel=args.skip_relabel,
            force_models=args.force_models,
            workers=args.workers,
            device=args.device or "cuda",
        )
    else:
        make_pilot(full=args.full)
        outline()
        run(args.model, device=args.device, tau=args.tau,
            robustness=args.robustness,
            data_path=FULL_DATA if args.full else DATA)
        notes = Path(args.notes) if args.notes else default_notes_path()
        if notes and notes.exists():
            mimic_build(Path(args.mimic_dir), max_hadm=args.max_hadm,
                        notes_path=notes, window_h=args.window_hours,
                        text_chars=args.text_chars)
            mimic_cxo_experiment(
                model=args.model, cap_per_cell=args.mimic_cap_per_cell,
                device=args.device)


if __name__ == "__main__":
    main()

