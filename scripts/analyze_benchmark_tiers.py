#!/usr/bin/env python3
"""Quick diagnostic stats for SilentEHR pilot vs full tiers."""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def word_count(text: str) -> int:
    return len(text.split())


def has_concept(row: dict) -> bool:
    text = row["profile"].lower()
    name = row["concept"].lower()
    cid = row["concept_id"].replace("_", " ")
    return name in text or cid in text


def neg_cue(text: str) -> bool:
    t = text.lower()
    return any(x in t for x in ("denies", "negative for", " absent", "no current"))


def analyze(name: str, rows: list[dict]) -> None:
    print(f"\n=== {name} ===")
    print(f"rows={len(rows)}")
    print(f"concepts={len(set(r['concept_id'] for r in rows))}")
    print(f"backgrounds={len(set(r['background_id'] for r in rows))}")
    print(f"states={dict(Counter(r['state'] for r in rows))}")

    by_state = {}
    for state in sorted(set(r["state"] for r in rows)):
        subset = [r for r in rows if r["state"] == state]
        ws = [word_count(r["profile"]) for r in subset]
        mentions = sum(has_concept(r) for r in subset)
        by_state[state] = {
            "n": len(subset),
            "avg_words": sum(ws) / len(ws),
            "concept_present": mentions,
        }
        print(
            f"  {state:12s} n={len(subset):5d} "
            f"avg_words={by_state[state]['avg_words']:.1f} "
            f"concept_in_text={mentions}/{len(subset)}"
        )

    unm = [r for r in rows if r["state"] == "unmentioned"]
    other = [r for r in rows if r["state"] != "unmentioned"]
    if unm and other:
        unm_w = sum(word_count(r["profile"]) for r in unm) / len(unm)
        other_w = sum(word_count(r["profile"]) for r in other) / len(other)
        print(f"  length gap (other-unmentioned words): {other_w - unm_w:.1f}")

    # lexical rules
    unm_pred = sum(not has_concept(r) for r in rows if r["state"] == "unmentioned")
    unm_total = sum(1 for r in rows if r["state"] == "unmentioned")
    other_pred = sum(not has_concept(r) for r in rows if r["state"] != "unmentioned")
    other_total = len(rows) - unm_total
    rule_acc = (unm_pred + (other_total - other_pred)) / len(rows)
    print(f"  rule: no concept phrase -> Unmentioned accuracy={rule_acc:.3f}")

    sr = [r for r in rows if r["state"] in ("supported", "refuted")]
    if sr:
        sr_ok = sum(
            (r["state"] == "supported") != neg_cue(r["profile"]) for r in sr
        )
        print(f"  rule: deny/neg cue -> Refuted on S/R subset={sr_ok/len(sr):.3f}")

    # evidence always appended?
    appended = 0
    for r in rows:
        if r["state"] == "unmentioned":
            continue
        bg = r["profile"].rsplit(".", 1)
        if len(bg) == 2 and len(bg[1].strip()) < 120:
            appended += 1
    print(f"  evidence likely final sentence: {appended}/{len(rows)-len(unm)}")


def main() -> None:
    pilot = ROOT / "data" / "pilot.jsonl"
    full = ROOT / "data" / "controlled_full.jsonl"
    if pilot.exists():
        analyze("pilot.jsonl", load(pilot))
    if full.exists():
        analyze("controlled_full.jsonl", load(full))


if __name__ == "__main__":
    main()
