#!/usr/bin/env python3
"""HCUNR threshold sensitivity at tau in {0.8, 0.9, 0.95}."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mechanism_analysis_utils import FULL_DATA, MODELS, encode_propositions, load_embeddings, load_rows  # noqa: E402
from silentehr import OUT  # noqa: E402
from silentehr_trap_utils import hcunr_curve  # noqa: E402


def main():
    rows = load_rows(FULL_DATA)
    all_out = {"protocol": "new_background", "models": {}}

    for label, model in MODELS:
        print(f"{label} ...", flush=True)
        device = None if model == "tfidf" else "cuda"
        emb, E = load_embeddings(model, rows, device=device)
        P = encode_propositions(emb, rows)
        res = hcunr_curve(rows, E, P)
        all_out["models"][model] = res
        print(
            f"  HCUNR@.8={res['HCUNR@0.8']:.3f} "
            f"@.9={res['HCUNR@0.9']:.3f} @.95={res['HCUNR@0.95']:.3f}",
            flush=True,
        )

    out_path = OUT / "hcunr_threshold_sensitivity.json"
    out_path.write_text(json.dumps(all_out, indent=2), encoding="utf-8")
    print(f"\nsaved {out_path}")

    print(f"\n{'Encoder':<14} {'@0.8':>6} {'@0.9':>6} {'@0.95':>6} {'UNR':>6}")
    for label, model in MODELS:
        r = all_out["models"][model]
        print(
            f"{label:<14} {r['HCUNR@0.8']:6.3f} {r['HCUNR@0.9']:6.3f} "
            f"{r['HCUNR@0.95']:6.3f} {r['UNR_unmentioned']:6.3f}"
        )


if __name__ == "__main__":
    main()
