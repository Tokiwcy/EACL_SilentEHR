# LaTeX Manuscript

The ACL-format paper lives in **`results.tex`** (replaces the former Markdown draft).

## Compile

Requires the [ACL style files](https://github.com/acl-org/acl-style-files):
`acl.sty`, `acl_natbib.bst`.

```bash
pdflatex results.tex
bibtex results
pdflatex results.tex
pdflatex results.tex
```

## Contents

| Section | Description |
|---------|-------------|
| Abstract | CWE novelty + main numbers |
| §3 Datasets | Table 1: SilentEHR vs MIMIC |
| §4 Methods | Problem, encoders, $C\times O$, mitigation |
| §5 Results | Tables 2–6: controlled, artifact controls, MIMIC, audit, mitigation |
| Appendix | Protocol and cell-design tables |

## Tables (all in `results.tex`)

1. **Dataset summary** — sizes, splits, label sources  
2. **Controlled new-background** — Known / UNR / HCUNR / Det / ECI  
3. **Artifact-resistant tiers** — Unmentioned UNR (length / randpos / indirect)  
4. **MIMIC matched cohort** — UPFNR / HCWE / doc gap / detection  
5. **Human audit** — dual-annotator, consensus, per-concept  
6. **Mitigation** — base vs 10/50/100 Unknown groups  
7. **Abstention** — confidence + Mahalanobis (new-background)  
8. **Real-note CV** — grouped 5-fold pooled OOF with Wilson CI  

Observation labels in text: **DeepSeek V4 Flash** (`deepseek-v4-flash`). Archived inference
settings: `configs/observation_labeling.yaml`.

## Regenerate numbers

```bash
python scripts/mimic_audited_cohort.py
python scripts/mimic_audit_ominus_precision.py
python scripts/mimic_audit_ominus_sensitivity.py
python scripts/mimic_label_noise_sensitivity.py
python scripts/local_neighborhood_composition.py
python scripts/unknown_known_separability.py
python scripts/input_ablation.py
python scripts/pair_feature_ablation.py
python scripts/mimic_cell_recalls.py
python scripts/mimic_temporal_observability.py
python scripts/hcunr_threshold_sensitivity.py
python scripts/mechanism_probe_robustness.py
python scripts/resplit_protocol_metrics.py
python scripts/build_real_clinical_sru.py
python scripts/real_clinical_sru_experiment.py
python scripts/real_clinical_sru_cv.py
python scripts/label_withholding_experiment.py
python scripts/artifact_baselines.py
python scripts/summarize_mimic_concept_subsets.py
python scripts/summarize_audit_wilson.py
python scripts/extract_paper_tables.py
python scripts/compare_dual_human_deepseek.py
python scripts/compare_human_consensus.py
python scripts/summarize_artifact_tiers.py
```
