#!/usr/bin/env python3
"""Human audit binary metrics: kappa, O- precision/recall/F1."""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

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
from silentehr import OBSERVATION_LABEL_MODEL  # noqa: E402


def collapse(status: str) -> str:
    return "unmentioned" if status == "unmentioned" else "mentioned"


def o_minus_metrics(human, llm):
    """O- = unmentioned; positive class for detection."""
    h = [collapse(x) == "unmentioned" for x in human]
    l = [collapse(x) == "unmentioned" for x in llm]
    tn, fp, fn, tp = confusion_matrix(h, l, labels=[False, True]).ravel()
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    acc = sum(a == b for a, b in zip(h, l)) / len(h)
    kappa = float(cohen_kappa_score(h, l))
    return {
        "n": len(human),
        "binary_accuracy": acc,
        "binary_kappa": kappa,
        "o_minus_precision": prec,
        "o_minus_recall": rec,
        "o_minus_f1": f1,
        "confusion": {
            "human_unm_llm_unm": int(tp),
            "human_unm_llm_mention": int(fn),
            "human_mention_llm_unm": int(fp),
            "human_mention_llm_mention": int(tn),
        },
    }


def build_rows(h1_map, h2_map, by_audit, deepseek):
    rows = []
    for audit_id, concept in sorted(set(h1_map) & set(h2_map)):
        if by_audit[audit_id]["audit_row"]["concept"] != concept:
            continue
        hadm_id = by_audit[audit_id]["hadm_id"]
        llm = deepseek.get((hadm_id, concept))
        if llm is None:
            continue
        rows.append({
            "audit_id": audit_id,
            "concept": concept,
            "human1": h1_map[(audit_id, concept)],
            "human2": h2_map[(audit_id, concept)],
            "llm": llm,
            "consensus": (
                h1_map[(audit_id, concept)]
                if h1_map[(audit_id, concept)] == h2_map[(audit_id, concept)]
                else None
            ),
        })
    return rows


def main():
    h1 = load_human1()
    h2 = load_human2()
    by_audit, _ = load_audit_lookup()
    deepseek = load_deepseek_by_hadm()
    h1_map = {
        (r["audit_id"], r["concept"]): r["label"]
        for _, r in h1.iterrows() if r["label"] in ALLOWED
    }
    h2_map = {
        (r["audit_id"], r["concept"]): r["label"]
        for _, r in h2.iterrows() if r["label"] in ALLOWED
    }
    rows = build_rows(h1_map, h2_map, by_audit, deepseek)

    results = {"observation_label_model": OBSERVATION_LABEL_MODEL}
    for name, key in (
        ("human1", "human1"),
        ("human2", "human2"),
        ("mean_human", None),
    ):
        if key:
            human = [r[key] for r in rows]
        else:
            continue
        results[f"{name}_vs_llm"] = o_minus_metrics(human, [r["llm"] for r in rows])

    consensus_rows = [r for r in rows if r["consensus"] is not None]
    results["consensus_vs_llm"] = o_minus_metrics(
        [r["consensus"] for r in consensus_rows],
        [r["llm"] for r in consensus_rows],
    )
    results["consensus_n"] = len(consensus_rows)
    results["disagreement_n"] = len(rows) - len(consensus_rows)

    # seven-class metrics for consensus
    cons_h = [r["consensus"] for r in consensus_rows]
    cons_l = [r["llm"] for r in consensus_rows]
    results["consensus_seven_class"] = {
        "accuracy": sum(a == b for a, b in zip(cons_h, cons_l)) / len(cons_h),
        "cohen_kappa": float(cohen_kappa_score(cons_h, cons_l)),
    }

    print(json.dumps(results, indent=2))
    out = ROOT / "outputs" / "human_audit_binary_metrics.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
