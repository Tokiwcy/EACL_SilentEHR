#!/usr/bin/env python3
"""Pair-feature composition ablation on SilentEHR new-background protocol."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mechanism_analysis_utils import FULL_DATA, MODELS, encode_propositions, load_embeddings, load_rows  # noqa: E402
from silentehr import OUT  # noqa: E402
from silentehr_trap_utils import run_binary_trap  # noqa: E402

PAIR_MODES = [
    ("ex", "e_x"),
    ("concat", "[e_x;e_h]"),
    ("full", "[e_x;e_h;|Δ|;⊙]"),
]


def main():
    rows = load_rows(FULL_DATA)
    all_out = {"protocol": "new_background", "models": {}}

    for label, model in MODELS:
        print(f"\n=== {label} ===", flush=True)
        device = None if model == "tfidf" else "cuda"
        emb, E = load_embeddings(model, rows, device=device)
        P = encode_propositions(emb, rows)
        model_out = {}
        for mode, desc in PAIR_MODES:
            res = run_binary_trap(rows, E, P, feature_mode=mode)
            model_out[mode] = {"desc": desc, **res}
            print(
                f"  {desc:20s} K={res['known_acc']:.3f} "
                f"UNR={res['UNR_unmentioned']:.3f} "
                f"HCUNR={res['HCUNR@0.9']:.3f} "
                f"U/K={res['unmentioned_auroc']:.3f}",
                flush=True,
            )
        all_out["models"][model] = model_out

    out_path = OUT / "pair_feature_ablation.json"
    out_path.write_text(json.dumps(all_out, indent=2), encoding="utf-8")
    print(f"\nsaved {out_path}")

    header = (
        f"{'Encoder':<14} {'Mode':<8} {'Known':>6} {'UNR':>6} "
        f"{'HCUNR':>6} {'U/K':>6}"
    )
    print(f"\n{header}")
    for label, model in MODELS:
        for mode, desc in PAIR_MODES:
            r = all_out["models"][model][mode]
            print(
                f"{label:<14} {desc:<8} {r['known_acc']:6.3f} "
                f"{r['UNR_unmentioned']:6.3f} {r['HCUNR@0.9']:6.3f} "
                f"{r['unmentioned_auroc']:6.3f}"
            )


if __name__ == "__main__":
    main()
