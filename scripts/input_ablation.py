#!/usr/bin/env python3
"""SilentEHR input ablation: full vs patient-only vs prop-only vs shuffled prop."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mechanism_analysis_utils import FULL_DATA, encode_propositions, load_embeddings, load_rows  # noqa: E402
from silentehr import OUT  # noqa: E402
from silentehr_trap_utils import run_binary_trap  # noqa: E402

MODELS = [
    ("Qwen3-Emb", "qwen"),
    ("Jina v5", "jina"),
    ("BioLORD", "biolord"),
    ("med-gte", "med-gte"),
    ("TF-IDF", "tfidf"),
]

MODES = [
    ("full", "Full"),
    ("patient", "Patient-only"),
    ("prop", "Prop-only"),
    ("shuffled", "Shuffled"),
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
        for mode, _ in MODES:
            res = run_binary_trap(rows, E, P, feature_mode=mode)
            model_out[mode] = res
            print(
                f"  {mode:8s} known={res['known_acc']:.3f} "
                f"UNR={res['UNR_unmentioned']:.3f}",
                flush=True,
            )
        all_out["models"][model] = model_out

    out_path = OUT / "input_ablation.json"
    out_path.write_text(json.dumps(all_out, indent=2), encoding="utf-8")
    print(f"\nsaved {out_path}")

    header = (
        f"{'Encoder':<12} {'Full K':>7} {'Full U':>7} "
        f"{'Pat K':>7} {'Pat U':>7} {'Prop K':>7} {'Prop U':>7} "
        f"{'Shuf K':>7}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for label, model in MODELS:
        m = all_out["models"][model]
        print(
            f"{label:<12} "
            f"{m['full']['known_acc']:7.3f} {m['full']['UNR_unmentioned']:7.3f} "
            f"{m['patient']['known_acc']:7.3f} {m['patient']['UNR_unmentioned']:7.3f} "
            f"{m['prop']['known_acc']:7.3f} {m['prop']['UNR_unmentioned']:7.3f} "
            f"{m['shuffled']['known_acc']:7.3f}"
        )


if __name__ == "__main__":
    main()
