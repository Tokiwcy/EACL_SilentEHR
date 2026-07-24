#!/usr/bin/env python3
"""Stratified O- precision on human audit vs Qwen labels + cohort weighting."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_dual_human_deepseek import (  # noqa: E402
    ALLOWED,
    load_audit_lookup,
    load_deepseek_by_hadm,
    load_human1,
    load_human2,
)
from mimic_eval_utils import mimic_split  # noqa: E402
from silentehr import OBSERVATION_LABEL_MODEL  # noqa: E402

AUDITED = {"anemia", "thrombocytopenia"}


def collapse(status: str) -> str:
    return "unmentioned" if status == "unmentioned" else "mentioned"


def build_audit_rows():
    h1 = load_human1()
    h2 = load_human2()
    by_audit, _ = load_audit_lookup()
    deepseek = load_deepseek_by_hadm()
    h1_map = {
        (r["audit_id"], r["concept"]): r["label"].strip()
        for _, r in h1.iterrows() if r["label"] in ALLOWED
    }
    h2_map = {
        (r["audit_id"], r["concept"]): r["label"].strip()
        for _, r in h2.iterrows() if r["label"] in ALLOWED
    }
    rows = []
    for audit_id, concept in sorted(set(h1_map) & set(h2_map)):
        if by_audit[audit_id]["audit_row"]["concept"] != concept:
            continue
        struct_row = by_audit[audit_id]["struct_row"]
        hadm_id = struct_row["hadm_id"]
        llm = deepseek.get((hadm_id, concept))
        if llm is None:
            continue
        h1_label = h1_map[(audit_id, concept)]
        h2_label = h2_map[(audit_id, concept)]
        consensus = h1_label if h1_label == h2_label else None
        rows.append({
            "audit_id": audit_id,
            "concept": concept,
            "structured_positive": bool(struct_row["structured_positive"]),
            "human1": h1_label,
            "human2": h2_label,
            "consensus": consensus,
            "llm": llm,
        })
    return rows


def qwen_ominus_rows(rows):
    return [r for r in rows if collapse(r["llm"]) == "unmentioned"]


def precision_on_subset(rows, human_key):
    """P(human O-| Qwen O-) on rows already filtered to Qwen O-."""
    if not rows:
        return {"n": 0, "precision": None, "tp": 0, "fp": 0}
    tp = fp = 0
    for r in rows:
        if human_key == "consensus":
            if r["consensus"] is None:
                continue
            human = r["consensus"]
        else:
            human = r[human_key]
        if collapse(human) == "unmentioned":
            tp += 1
        else:
            fp += 1
    denom = tp + fp
    return {
        "n": denom,
        "precision": tp / denom if denom else None,
        "tp": tp,
        "fp": fp,
    }


def stratum_key(r):
    return (r["concept"], r["structured_positive"])


def cohort_qwen_ominus_counts():
    _, test, eligible = mimic_split()
    counts = defaultdict(int)
    for r in test:
        if r["concept"] not in AUDITED:
            continue
        if r.get("mention_status") != "unmentioned":
            continue
        counts[stratum_key_from_test(r)] += 1
    return dict(counts)


def stratum_key_from_test(r):
    return (r["concept"], bool(r["structured_positive"]))


def stratified_audit_precisions(qwen_rows, human_key):
    by_stratum = defaultdict(list)
    for r in qwen_rows:
        by_stratum[stratum_key(r)].append(r)
    out = {}
    for sk, sub in sorted(by_stratum.items()):
        out[f"{sk[0]}_{'C+' if sk[1] else 'C-'}"] = precision_on_subset(sub, human_key)
    return out


def cohort_weighted_precision(stratum_precisions, cohort_counts):
    """Weight stratum audit precisions by full-cohort Qwen-O- stratum shares."""
    total = sum(cohort_counts.values())
    if not total:
        return None
    weighted = 0.0
    used = 0
    detail = {}
    for sk, n_cohort in sorted(cohort_counts.items()):
        label = f"{sk[0]}_{'C+' if sk[1] else 'C-'}"
        sp = stratum_precisions.get(label)
        if not sp or sp["n"] == 0 or sp["precision"] is None:
            detail[label] = {
                "cohort_n": n_cohort,
                "audit_n": sp["n"] if sp else 0,
                "precision": None,
                "weight": n_cohort / total,
            }
            continue
        w = n_cohort / total
        weighted += sp["precision"] * w
        used += w
        detail[label] = {
            "cohort_n": n_cohort,
            "audit_n": sp["n"],
            "precision": sp["precision"],
            "weight": w,
        }
    return {
        "precision": weighted,
        "coverage": used,
        "strata": detail,
    }


def stratified_bootstrap_ci(qwen_rows, human_key, cohort_counts, n_reps=10000, seed=0):
    """Bootstrap cohort-weighted O- precision within audit strata."""
    by_stratum = defaultdict(list)
    for r in qwen_rows:
        if human_key == "consensus" and r["consensus"] is None:
            continue
        human = r["consensus"] if human_key == "consensus" else r[human_key]
        tp = int(collapse(human) == "unmentioned")
        by_stratum[stratum_key(r)].append(tp)

    total = sum(cohort_counts.values())
    weights = {sk: n / total for sk, n in cohort_counts.items()}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_reps):
        weighted = 0.0
        for sk, w in weights.items():
            vals = by_stratum.get(sk, [])
            if not vals:
                continue
            draw = rng.choice(vals, size=len(vals), replace=True)
            weighted += w * float(np.mean(draw))
        samples.append(weighted)
    arr = np.asarray(samples)
    return {
        "n_reps": n_reps,
        "p025": float(np.quantile(arr, 0.025)),
        "p50": float(np.quantile(arr, 0.5)),
        "p975": float(np.quantile(arr, 0.975)),
    }


def main():
    rows = build_audit_rows()
    qwen_rows = qwen_ominus_rows(rows)
    qwen_cplus = [r for r in qwen_rows if r["structured_positive"]]
    cohort_counts = cohort_qwen_ominus_counts()

    out = {
        "observation_label_model": OBSERVATION_LABEL_MODEL,
        "audit_n": len(rows),
        "qwen_ominus_n": len(qwen_rows),
        "cohort_qwen_ominus_counts": {
            f"{k[0]}_{'C+' if k[1] else 'C-'}": v
            for k, v in sorted(cohort_counts.items())
        },
        "cohort_qwen_ominus_total_audited": sum(cohort_counts.values()),
    }

    for human_key in ("human1", "human2", "consensus"):
        pool = precision_on_subset(qwen_rows, human_key)
        cplus = precision_on_subset(qwen_cplus, human_key)
        strata = stratified_audit_precisions(qwen_rows, human_key)
        weighted = cohort_weighted_precision(strata, cohort_counts)
        boot = stratified_bootstrap_ci(qwen_rows, human_key, cohort_counts)
        out[human_key] = {
            "pooled_audit_sample": pool,
            "given_cplus": cplus,
            "by_stratum": strata,
            "cohort_weighted": {**weighted, "bootstrap_95CI": boot},
        }

    # Consensus-only audit sample estimate (255 cases) for comparison
    consensus_rows = [r for r in rows if r["consensus"] is not None]
    out["consensus_subset_note"] = {
        "n": len(consensus_rows),
        "qwen_ominus_pooled": precision_on_subset(
            qwen_ominus_rows(consensus_rows), "consensus"),
    }

    path = ROOT / "outputs" / "mimic_audit_ominus_precision.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved {path}\n")

    c = out["consensus"]
    print("=== Consensus O- precision (Qwen predicts unmentioned) ===")
    print(f"Pooled audit sample (n={c['pooled_audit_sample']['n']}): "
          f"{c['pooled_audit_sample']['precision']:.3f}")
    print(f"P(human O-| Qwen O-, C+): "
          f"{c['given_cplus']['precision']:.3f} "
          f"(n={c['given_cplus']['n']})")
    print(f"Cohort-weighted (audited concepts): "
          f"{c['cohort_weighted']['precision']:.3f} "
          f"[{c['cohort_weighted']['bootstrap_95CI']['p025']:.3f}"
          f"--{c['cohort_weighted']['bootstrap_95CI']['p975']:.3f}]")
    for label, sp in c["by_stratum"].items():
        p = sp["precision"]
        print(f"  {label}: n={sp['n']} prec={p:.3f}" if p is not None
              else f"  {label}: n={sp['n']} prec=NA")


if __name__ == "__main__":
    main()
