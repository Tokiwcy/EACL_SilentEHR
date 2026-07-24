SilentEHR — paper outline (EACL/COLING style)

Title
  SilentEHR: Epistemic Collapse in Frozen Patient Representations

One-sentence claim
  Frozen patient representations encode clinical content without the observation
  process; under closed-world Supported/Refuted training, undocumented concepts
  become high-confidence negatives.

Contributions
  1. Define epistemic state collapse for patient–hypothesis representations.
  2. SilentEHR: matched controlled benchmark (Supported / Refuted / Unknown).
  3. Binary Training Trap: UNR / HCUNR across frozen embedders.
  4. MIMIC structured validation: unmentioned-positive vs unmentioned-negative.
  5. Trial eligibility harms: unsafe auto-clearance on unknowns.
  6. CAPR / observation mask: simple coverage-aware fix (oracle vs predicted).

RQs
  RQ1 Geometry: is Unmentioned closer to Refuted than to Supported?
  RQ2 Trap: do binary classifiers confidently map Unknown → Refuted?
  RQ3 MIMIC: does omission mix true positives and negatives?
  RQ4 Fix: does explicit observation status reduce false certainty?

Main task
  y(x,h) ∈ {Supported, Refuted, Unknown}
  Fine states (Unmentioned, Not assessed, Pending, …) are analysis-only.

Pilot (this repo)
  6 concepts × 20 backgrounds × 6 states × 3 paraphrases ≈ 2160 pairs
  Splits: background-held-out (+ concept-held-out stress on last 2 concepts)
  Models: hash (smoke) | clinicalbert | qwen | jina
  Expts: margin/ECI, binary trap, abstention AUROC, oracle mention mask

Go / no-go
  Known Acc ≥ 0.80; HCUNR@0.9 ≥ 0.30–0.40 on ≥2 embedders;
  confidence thresholding fails on Unknown; oracle mask drops HCUNR ≥ 0.15
  with Known Acc drop ≤ 0.02.

What the paper is NOT
  Not “embeddings cannot do negation.”
  Not “just add Unknown as a class.”
  It is: silence ≠ evidence of absence under frozen closed-world pipelines.
