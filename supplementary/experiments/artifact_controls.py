#!/usr/bin/env python3
"""Artifact-control tiers on SilentEHR."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SUP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUP_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from supplementary.paths import OUTPUT_DIR, ensure_repo_on_path  # noqa: E402

ensure_repo_on_path()


def main() -> None:
    ap = argparse.ArgumentParser(description="SilentEHR artifact controls")
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.regenerate:
        from supplementary.data_generation.controlled_corpus import generate_artifact_tiers  # noqa: WPS433
        generate_artifact_tiers()

    script = REPO_ROOT / "scripts" / "artifact_baselines.py"
    subprocess.run([sys.executable, str(script)], check=True)

    src = REPO_ROOT / "outputs" / "artifact_baselines.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUTPUT_DIR / "artifact_baselines.json"
    if src.exists():
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        data = json.loads(dst.read_text(encoding="utf-8"))
        print(json.dumps(data, indent=2)[:2000])
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
