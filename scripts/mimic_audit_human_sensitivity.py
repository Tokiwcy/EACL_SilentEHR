#!/usr/bin/env python3
"""MIMIC sensitivity on human-audit subsets (human-consensus O-)."""
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
            "human_consensus_unmentioned": consensus == "unmentioned",
            "llm_unmentioned": llm == "unmentioned",
            "human_llm_agree_unmentioned": (
                consensus == "unmentioned" and llm == "unmentioned"),
        })
    return rows


def evaluate_model(model, device="cuda"):
    train, test, eligible = mimic_split()
    audit_rows = [r for r in build_audit_rows() if r["concept"] in eligible]
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
    subsets = {
        "all_audit": audit_rows,
        "llm_unmentioned": [r for r in audit_rows if r["llm_unmentioned"]],
        "human_consensus_unmentioned": [
            r for r in audit_rows if r["human_consensus_unmentioned"]],
        "human_llm_agree_unmentioned": [
            r for r in audit_rows if r["human_llm_agree_unmentioned"]],
        "human_consensus_unmentioned_C+": [
            r for r in audit_rows
            if r["human_consensus_unmentioned"] and r["structured_positive"]],
        "llm_unmentioned_C+": [
            r for r in audit_rows
            if r["llm_unmentioned"] and r["structured_positive"]],
        "human_llm_agree_unmentioned_C+": [
            r for r in audit_rows
            if r["human_llm_agree_unmentioned"] and r["structured_positive"]],
    }
    result = {"model": model, "eligible_concepts": eligible, "subsets": {}}
    p_map = {id(r): audit_p[i] for i, r in enumerate(audit_rows)}
    for name, sub in subsets.items():
        p = np.asarray([p_map[id(r)] for r in sub]) if sub else np.array([])
        result["subsets"][name] = subset_metrics(sub, p)
    return result


def main():
    models = ["tfidf", "qwen", "jina", "biolord"]
    out = ROOT / "outputs" / "mimic_audit_human_sensitivity.json"
    all_results = {}
    if out.exists():
        all_results = json.loads(out.read_text(encoding="utf-8"))
    for model in models:
        if model in all_results:
            print(f"skip {model} (already saved)", flush=True)
            continue
        print(f"evaluating {model} ...", flush=True)
        device = None if model == "tfidf" else "cuda"
        all_results[model] = evaluate_model(model, device=device)
        out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"checkpoint saved ({model})", flush=True)
    print(f"saved {out}")
    for model, res in all_results.items():
        print(f"\n=== {model} ===")
        for name, metrics in res["subsets"].items():
            if metrics.get("n", 0) == 0:
                continue
            print(name, metrics)


if __name__ == "__main__":
    main()
