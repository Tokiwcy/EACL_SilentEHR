#!/usr/bin/env python3
"""MIMIC structured-state bridge with Qwen3-12B observation labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_generation.paths import OUTPUT_DIR, ensure_repo_on_path  # noqa: E402

ensure_repo_on_path()


def main() -> None:
    ap = argparse.ArgumentParser(description="MIMIC C×O bridge (Qwen3-12B labels)")
    ap.add_argument(
        "cmd",
        choices=["build-struct", "label", "cxo", "pipeline"],
    )
    ap.add_argument("--mimic-dir", type=Path, default=Path(r"D:\mimic-iv-3.1"))
    ap.add_argument("--notes", type=Path, default=None)
    ap.add_argument("--max-hadm", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="hash", choices=["hash", "tfidf", "qwen", "jina", "biolord"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--cap-per-cell", type=int, default=500)
    ap.add_argument("--reuse-struct", action="store_true")
    ap.add_argument("--skip-label", action="store_true")
    args = ap.parse_args()

    from data_generation.qwen_observation_labeler import (  # noqa: WPS433
        build_qwen3_struct,
        relabel_mimic_notes,
    )
    from data_generation.silentehr import (  # noqa: WPS433
        MIMIC_STRUCT,
        MIMIC_STRUCT_QWEN3,
        mimic_build,
        mimic_cxo_experiment,
    )

    if args.cmd in ("build-struct", "pipeline") and not args.skip_label:
        if args.cmd == "pipeline" and not (args.reuse_struct and MIMIC_STRUCT.exists()):
            mimic_build(
                args.mimic_dir,
                max_hadm=args.max_hadm,
                notes_path=args.notes,
            )
        elif args.cmd == "pipeline" and args.reuse_struct:
            print(f"reusing {MIMIC_STRUCT}")
        build_qwen3_struct()

    if args.cmd == "label":
        relabel_mimic_notes(limit=args.limit, device=args.device)
        build_qwen3_struct()
    elif args.cmd == "build-struct":
        if not args.skip_label:
            relabel_mimic_notes(limit=args.limit, device=args.device)
        build_qwen3_struct()
    elif args.cmd == "cxo":
        if not MIMIC_STRUCT_QWEN3.exists():
            raise SystemExit(f"missing {MIMIC_STRUCT_QWEN3}; run label or build-struct first")
        result = mimic_cxo_experiment(
            struct_path=MIMIC_STRUCT_QWEN3,
            model=args.model,
            cap_per_cell=args.cap_per_cell,
            device=args.device,
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_DIR / f"mimic_cxo_{args.model}.json"
        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"saved {out}")
    elif args.cmd == "pipeline":
        if not args.skip_label:
            relabel_mimic_notes(limit=args.limit, device=args.device)
            build_qwen3_struct()
        result = mimic_cxo_experiment(
            struct_path=MIMIC_STRUCT_QWEN3,
            model=args.model,
            cap_per_cell=args.cap_per_cell,
            device=args.device,
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_DIR / f"mimic_pipeline_{args.model}.json"
        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
