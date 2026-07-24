#!/usr/bin/env python3
"""Human1/Human2 consensus and average metrics vs LLM labels."""
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

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


def metrics(human, llm):
    n = len(human)
    acc = sum(a == b for a, b in zip(human, llm)) / n
    kappa = float(cohen_kappa_score(human, llm))
    b_acc = sum(
        collapse(a) == collapse(b) for a, b in zip(human, llm)
    ) / n
    return {"n": n, "accuracy": acc, "cohen_kappa": kappa, "binary_mention": b_acc}


def main():
    h1 = load_human1()
    h2 = load_human2()
    by_audit, _ = load_audit_lookup()
    deepseek = load_deepseek_by_hadm()

    h1_map = {
        (r["audit_id"], r["concept"]): r["label"]
        for _, r in h1.iterrows()
        if r["label"] in ALLOWED
    }
    h2_map = {
        (r["audit_id"], r["concept"]): r["label"]
        for _, r in h2.iterrows()
        if r["label"] in ALLOWED
    }

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
        })

    h1_labels = [r["human1"] for r in rows]
    h2_labels = [r["human2"] for r in rows]
    llm_labels = [r["llm"] for r in rows]

    inter = metrics(h1_labels, h2_labels)
    inter["binary_kappa"] = float(cohen_kappa_score(
        [collapse(x) for x in h1_labels],
        [collapse(x) for x in h2_labels],
    ))
    m1 = metrics(h1_labels, llm_labels)
    m2 = metrics(h2_labels, llm_labels)

    agree_rows = [r for r in rows if r["human1"] == r["human2"]]
    consensus_labels = [r["human1"] for r in agree_rows]
    consensus_llm = [r["llm"] for r in agree_rows]
    consensus = metrics(consensus_labels, consensus_llm)

    avg = {
        "accuracy": (m1["accuracy"] + m2["accuracy"]) / 2,
        "cohen_kappa": (m1["cohen_kappa"] + m2["cohen_kappa"]) / 2,
        "binary_mention": (m1["binary_mention"] + m2["binary_mention"]) / 2,
    }

    print(f"=== Matched audit concepts (n={len(rows)}) ===\n")
    print(f"Human1 vs Human2: {inter['accuracy']:.1%}, kappa={inter['cohen_kappa']:.3f}")
    print(f"Human1 vs {OBSERVATION_LABEL_MODEL}: {m1['accuracy']:.1%}, kappa={m1['cohen_kappa']:.3f}, binary={m1['binary_mention']:.1%}")
    print(f"Human2 vs {OBSERVATION_LABEL_MODEL}: {m2['accuracy']:.1%}, kappa={m2['cohen_kappa']:.3f}, binary={m2['binary_mention']:.1%}")
    print(f"\nSimple average (H1+H2)/2 vs LLM:")
    print(f"  accuracy={avg['accuracy']:.1%}, kappa={avg['cohen_kappa']:.3f}, binary={avg['binary_mention']:.1%}")
    print(f"\nConsensus (both agree, n={consensus['n']}) vs LLM:")
    print(f"  accuracy={consensus['accuracy']:.1%}, kappa={consensus['cohen_kappa']:.3f}, binary={consensus['binary_mention']:.1%}")
    print(f"  disagreement rate: {1 - len(agree_rows)/len(rows):.1%} ({len(rows)-len(agree_rows)} cases)")

    disagree = [r for r in rows if r["human1"] != r["human2"]]
    if disagree:
        print("\nTop Human1 vs Human2 disagreements:")
        for pair, count in Counter(
                (r["human1"], r["human2"]) for r in disagree).most_common(8):
            print(f"  {pair[0]} vs {pair[1]}: {count}")

    summary = {
        "observation_label_model": OBSERVATION_LABEL_MODEL,
        "n": len(rows),
        "human1_vs_human2": inter,
        "human1_vs_llm": m1,
        "human2_vs_llm": m2,
        "average_human_vs_llm": avg,
        "consensus_vs_llm": consensus,
        "consensus_distribution": dict(Counter(consensus_labels)),
    }
    out = ROOT / "outputs" / "human_consensus_comparison.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
