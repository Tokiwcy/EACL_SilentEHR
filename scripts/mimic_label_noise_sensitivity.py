#!/usr/bin/env python3
"""Label-noise sensitivity aligned with main-table mimic_cxo per-concept protocol."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mimic_audited_cohort import (  # noqa: E402
    HCWE_THRESH,
    audited_cplus_ominus_predictions,
    assert_matches_main_table,
    pooled_audited_metrics,
)

MODELS = [
    "tfidf", "qwen", "jina", "biolord", "clinicalbert", "medcpt", "med-gte",
]
MC_MODELS = MODELS
N_REPS = 1000
Q_VALUES = [0.55, 0.60, 0.65]

CALIBRATORS = {
    "per_annotator_conservative": {
        "p_keep": 0.55,
        "note": "per-annotator audit O- precision (conservative)",
    },
    "cohort_weighted_primary": {
        "p_keep": 0.60,
        "note": "cohort-weighted audit O- precision (primary)",
    },
    "consensus_sample_optimistic": {
        "p_keep": 0.65,
        "note": "consensus audit-sample O- precision (optimistic)",
    },
}


def partial_identification_bounds(n, e, q):
    qn = q * n
    lb = max(0.0, (e - (1 - q) * n) / qn) if qn > 0 else 0.0
    ub = min(1.0, e / qn) if qn > 0 else 1.0
    return {"LB": lb, "UB": ub, "q": q}


def mc_stratified(by_concept_probs, p_keep, rng):
    upfnr, hcwe = [], []
    for _ in range(N_REPS):
        kept = []
        for concept, probs in by_concept_probs.items():
            for prob in probs:
                if rng.random() < p_keep:
                    kept.append(prob)
        if not kept:
            continue
        kept = np.asarray(kept)
        upfnr.append(float(np.mean(kept < 0.5)))
        hcwe.append(float(np.mean(kept < HCWE_THRESH)))
    arr_u = np.asarray(upfnr)
    arr_h = np.asarray(hcwe)
    return {
        "UPFNR": {
            "mean": float(arr_u.mean()),
            "p025": float(np.quantile(arr_u, 0.025)),
            "p50": float(np.quantile(arr_u, 0.5)),
            "p975": float(np.quantile(arr_u, 0.975)),
        },
        "HCWE@0.9": {
            "mean": float(arr_h.mean()),
            "p025": float(np.quantile(arr_h, 0.025)),
            "p50": float(np.quantile(arr_h, 0.5)),
            "p975": float(np.quantile(arr_h, 0.975)),
        },
        "n_reps": len(upfnr),
    }


def stratify_probs(rows, probs):
    by_concept = defaultdict(list)
    for r, p in zip(rows, probs):
        by_concept[r["concept"]].append(float(p))
    return dict(by_concept)


def main():
    rng = np.random.default_rng(0)
    all_out = {
        "cohort": "audited_concepts",
        "protocol": "per_concept_fit_predict (matches mimic_cxo by_concept pool)",
        "N": 75,
        "q_values": Q_VALUES,
        "q_roles": {
            "0.55": "per-annotator conservative",
            "0.60": "cohort-weighted primary",
            "0.65": "consensus-sample optimistic",
        },
        "calibrators": CALIBRATORS,
        "mar_scenario_note": (
            "Missing-at-random (MAR) label-noise: each Qwen O- label independently "
            "retained with probability p_keep; false labels independent of model "
            "prediction. Median UPFNR should track observed UPFNR."
        ),
        "partial_identification": {},
        "mar_scenario": {},
    }

    for model in MODELS:
        print(f"model {model} ...", flush=True)
        device = None if model == "tfidf" else "cuda"
        rows, probs, _ = audited_cplus_ominus_predictions(
            model, device=device)
        metrics = pooled_audited_metrics(probs, _)
        assert_matches_main_table(model, metrics)
        n, e = metrics["N"], metrics["E"]
        obs = metrics["observed_UPFNR"]
        print(
            f"  N={n} E={e} UPFNR={obs:.6f} HCWE={metrics['HCWE@0.9']:.6f} "
            f"[assert OK]",
            flush=True,
        )

        all_out["partial_identification"][model] = {
            "N": n,
            "E": e,
            "observed_UPFNR": obs,
            "HCWE@0.9": metrics["HCWE@0.9"],
            "by_q": {
                str(q): partial_identification_bounds(n, e, q) for q in Q_VALUES
            },
        }

        by_concept_probs = stratify_probs(rows, probs)
        model_mar = {}
        for name, cfg in CALIBRATORS.items():
            p_keep = cfg["p_keep"]
            model_mar[name] = {
                **mc_stratified(by_concept_probs, p_keep, rng),
                "p_keep": p_keep,
                "n_base": n,
            }
            u = model_mar[name]["UPFNR"]
            print(
                f"  MAR {name} (p={p_keep}): "
                f"UPFNR {u['p025']:.3f}–{u['p975']:.3f} (med {u['p50']:.3f})",
                flush=True,
            )
        all_out["mar_scenario"][model] = model_mar

    out = ROOT / "outputs" / "mimic_label_noise_sensitivity.json"
    out.write_text(json.dumps(all_out, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
