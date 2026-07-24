#!/usr/bin/env python3
"""Unknown-supervision mitigation curves on SilentEHR."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SUP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUP_ROOT.parent))
sys.path.insert(0, str(SUP_ROOT))

from supplementary.paths import OUTPUT_DIR, ensure_repo_on_path  # noqa: E402

ensure_repo_on_path()


def main() -> None:
    from silentehr import FULL_DATA, run  # noqa: WPS433

    ap = argparse.ArgumentParser(description="SilentEHR mitigation (few-shot Unknown)")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--device", default=None)
    ap.add_argument("--data", type=Path, default=FULL_DATA)
    args = ap.parse_args()

    result = run(args.model, device=args.device, data_path=args.data)
    mit = result.get("mitigation_by_protocol", {}).get("new_background", {})
    curve = mit.get("fewshot_unknown_10_seed", {})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"mitigation_{args.model}.json"
    payload = {"model": args.model, "mitigation": mit, "full_result_keys": list(result)}
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(curve, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
