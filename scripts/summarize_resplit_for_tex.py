#!/usr/bin/env python3
import json
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "outputs" / "resplit_protocol_metrics.json"
d = json.loads(p.read_text())
for m in ["tfidf", "qwen", "jina", "biolord", "medcpt", "clinicalbert", "med-gte"]:
    bg = d[m]["background_splits"]
    cf = d[m]["concept_folds"]
    print(
        m,
        f"bg K={bg['known_acc']['mean']:.2f}±{bg['known_acc']['std']:.2f}",
        f"U={bg['UNR_unmentioned']['mean']:.2f}±{bg['UNR_unmentioned']['std']:.2f}",
        f"({int(bg['UNR_unmentioned']['frac_gt_0_5']*10)}/10)",
        f"cf K={cf['known_acc']['mean']:.2f} U={cf['UNR_unmentioned']['mean']:.2f}",
    )
