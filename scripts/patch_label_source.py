#!/usr/bin/env python3
"""Normalize observation-label model names in saved result files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = "deepseek-v4-flash"
OLD_VALUES = {
    "qwen3-14b",
    "deepseek_v4_flash",
    "deepseek_v4_flash_full_note",
    "rule_heuristic",
}


def patch_obj(obj):
    changed = False
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if key in (
                    "observation_label_source",
                    "observation_labels",
                    "observation_label_model",
            ) and value in OLD_VALUES:
                obj[key] = TARGET
                changed = True
            else:
                if patch_obj(value):
                    changed = True
    elif isinstance(obj, list):
        for item in obj:
            if patch_obj(item):
                changed = True
    return changed


def main():
    paths = list(OUT.glob("**/*.json"))
    if (OUT / "mimic_pipeline_results.jsonl").exists():
        lines = []
        for line in (OUT / "mimic_pipeline_results.jsonl").read_text(
                encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            patch_obj(obj)
            lines.append(json.dumps(obj, ensure_ascii=False))
        (OUT / "mimic_pipeline_results.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        print("patched mimic_pipeline_results.jsonl")

    for path in paths:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if patch_obj(obj):
            path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
            print(f"patched {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
