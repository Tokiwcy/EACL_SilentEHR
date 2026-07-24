#!/usr/bin/env python3
"""Artifact-control tiers on SilentEHR."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_generation.controlled_corpus import generate_artifact_tiers  # noqa: E402
from data_generation.paths import OUTPUT_DIR, ensure_repo_on_path  # noqa: E402
from evaluation.artifact_baselines import run_all  # noqa: E402

ensure_repo_on_path()


def main() -> None:
    ap = argparse.ArgumentParser(description="SilentEHR artifact controls")
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.regenerate:
        generate_artifact_tiers()

    results = run_all(output=OUTPUT_DIR / "artifact_baselines.json", device=args.device)
    print(json.dumps(results["baselines"], indent=2)[:2000])


if __name__ == "__main__":
    main()
