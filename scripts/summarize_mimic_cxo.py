#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in sorted((ROOT / "outputs").glob("mimic_cxo_*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    joint = data["joint_query_conditioned"]
    boot = joint.get("subject_bootstrap_95CI", {}).get("UPFNR", {})
    print(
        path.stem.replace("mimic_cxo_", ""),
        f"UPFNR={joint['UPFNR']:.3f}",
        f"HCWE={joint.get('HCWE@0.9') or 0:.3f}",
        f"gap={joint.get('documentation_gap') or 0:.3f}",
        f"det={joint.get('unmentioned_detection_AUROC') or 0:.3f}",
        f"CI={boot.get('lower', 0):.3f}-{boot.get('upper', 0):.3f}",
    )
