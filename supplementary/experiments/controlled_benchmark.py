#!/usr/bin/env python3
"""Run SilentEHR controlled benchmark (new-background protocol)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SUP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUP_ROOT.parent))
sys.path.insert(0, str(SUP_ROOT))

from supplementary.paths import DATA_DIR, OUTPUT_DIR, ensure_repo_on_path  # noqa: E402

ensure_repo_on_path()


def main() -> None:
    from silentehr import FULL_DATA, run  # noqa: WPS433

    ap = argparse.ArgumentParser(description="SilentEHR controlled benchmark")
    ap.add_argument(
        "--model", default="hash",
        choices=["hash", "tfidf", "clinicalbert", "qwen", "jina", "biolord", "medcpt", "med-gte"],
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--robustness", action="store_true")
    ap.add_argument("--data", type=Path, default=FULL_DATA)
    ap.add_argument("--full", action="store_true", help="regenerate controlled_full.jsonl first")
    args = ap.parse_args()

    if args.full:
        from supplementary.data_generation.controlled_corpus import generate_controlled  # noqa: WPS433
        generate_controlled(full=True)

    if not args.data.exists():
        raise SystemExit(f"missing {args.data}; run with --full or place controlled JSONL under data/")

    result = run(
        args.model,
        device=args.device,
        tau=args.tau,
        robustness=args.robustness,
        data_path=args.data,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"controlled_{args.model}.json"
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
