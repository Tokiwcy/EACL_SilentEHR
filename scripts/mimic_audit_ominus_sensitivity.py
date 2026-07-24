#!/usr/bin/env python3
"""MIMIC audit sensitivity under multiple human O- operational definitions."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
from mimic_eval_utils import (  # noqa: E402
    encode_extra_notes,
    encode_propositions,
    load_mimic_embeddings,
    mimic_split,
    notes_dict,
)
from silentehr import pair_features  # noqa: E402

MODELS = ["tfidf", "qwen", "jina", "biolord"]
AUDITED = {"anemia", "thrombocytopenia"}


def subset_metrics(rows, p, tau=0.9):
    if not rows:
        return {"n": 0}
    p = np.asarray(p)
    pred_pos = p >= 0.5
    cplus = np.asarray([r["structured_positive"] for r in rows])
    cminus = ~cplus
    out = {"n": len(rows)}
    if cplus.any():
        recall = float(np.mean(pred_pos[cplus]))
        out["C+_recall"] = recall
        out["UPFNR"] = 1 - recall
        out["HCWE@0.9"] = float(np.mean(p[cplus] < (1 - tau)))
        out["n_C+"] = int(cplus.sum())
    if cminus.any():
        out["C-_specificity"] = float(np.mean(~pred_pos[cminus]))
        out["n_C-"] = int(cminus.sum())
    out["predicted_negative_rate"] = float(np.mean(~pred_pos))
    return out


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
    for audit_id, concept in sorted(h1_map):
        if concept != by_audit[audit_id]["audit_row"]["concept"]:
            continue
        if (audit_id, concept) not in h2_map:
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
            **struct_row,
            "audit_id": audit_id,
            "human1": h1_label,
            "human2": h2_label,
            "consensus": consensus,
            "llm_status": llm,
        })
    return rows


def ominus_flags(r):
    h1u = r["human1"] == "unmentioned"
    h2u = r["human2"] == "unmentioned"
    cons = r["consensus"] == "unmentioned"
    return {
        "human1_O-": h1u,
        "human2_O-": h2u,
        "consensus_O-": cons,
        "union_O-": h1u or h2u,
        "intersection_O-": h1u and h2u,
        "llm_O-": r["llm_status"] == "unmentioned",
    }


def evaluate_model(model, device="cuda", concept_filter=None):
    train, test, eligible = mimic_split()
    audit_rows = build_audit_rows()
    if concept_filter:
        audit_rows = [r for r in audit_rows if r["concept"] in concept_filter]
    else:
        audit_rows = [r for r in audit_rows if r["concept"] in eligible]

    core_notes = notes_dict(train, test)
    audit_notes = notes_dict(audit_rows)
    notes = {**core_notes, **audit_notes}
    emb, Ez = load_mimic_embeddings(model, core_notes, train, device=device)
    Ez = encode_extra_notes(
        emb, Ez, notes, {r["hadm_id"] for r in audit_rows}, device=device)
    P = encode_propositions(emb, train + test + audit_rows)

    Xtr = np.asarray([
        pair_features(Ez[r["hadm_id"]], P[r["proposition"]]) for r in train])
    ytr = np.asarray([int(r["structured_positive"]) for r in train])
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0)
    clf.fit(scaler.transform(Xtr), ytr)

    def predict(rows):
        X = np.asarray([
            pair_features(Ez[r["hadm_id"]], P[r["proposition"]]) for r in rows])
        return clf.predict_proba(scaler.transform(X))[
            :, list(clf.classes_).index(1)]

    audit_p = predict(audit_rows)
    p_map = {id(r): audit_p[i] for i, r in enumerate(audit_rows)}

    subset_defs = {}
    for r in audit_rows:
        flags = ominus_flags(r)
        for name, ok in flags.items():
            if ok:
                subset_defs.setdefault(name, []).append(r)
    for name in list(subset_defs):
        subset_defs.setdefault(f"{name}_C+", [
            r for r in subset_defs[name] if r["structured_positive"]])

    result = {
        "model": model,
        "concept_filter": sorted(concept_filter) if concept_filter else sorted(eligible),
        "subsets": {},
    }
    for name, sub in subset_defs.items():
        p = np.asarray([p_map[id(r)] for r in sub]) if sub else np.array([])
        result["subsets"][name] = subset_metrics(sub, p)
    return result


def main():
    out = ROOT / "outputs" / "mimic_audit_ominus_sensitivity.json"
    all_results = {"all_eligible": {}, "audited_only": {}}
    for model in MODELS:
        print(f"evaluating {model} (all eligible) ...", flush=True)
        device = None if model == "tfidf" else "cuda"
        all_results["all_eligible"][model] = evaluate_model(model, device=device)
        print(f"evaluating {model} (audited concepts) ...", flush=True)
        all_results["audited_only"][model] = evaluate_model(
            model, device=device, concept_filter=AUDITED)
        out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"checkpoint {model}", flush=True)

    print(f"saved {out}")
    for scope in ("all_eligible", "audited_only"):
        print(f"\n=== {scope} C+ subsets ===")
        for model in MODELS:
            subs = all_results[scope][model]["subsets"]
            parts = []
            for key in (
                "human1_O-_C+", "human2_O-_C+", "consensus_O-_C+",
                "union_O-_C+", "intersection_O-_C+", "llm_O-_C+",
            ):
                m = subs.get(key, {})
                if m.get("n", 0):
                    parts.append(f"{key}: n={m['n_C+']} UPFNR={m['UPFNR']:.2f}")
            print(model, "; ".join(parts))


if __name__ == "__main__":
    main()
