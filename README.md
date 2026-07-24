# SilentEHR / COLING reproduction package

Research code for **SilentEHR**: epistemic collapse in frozen patient
representations under binary clinical verification.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python silentehr.py --help
```

Paper numbers live in `results.tex`. Regeneration commands are listed in
`results.md`.

## Public vs restricted artifacts

### Safe to publish (anonymous GitHub)

- `silentehr.py`, `scripts/`, `configs/`
- `requirements.txt`, ACL style files, `results.tex`, `PAPER.md`
- Controlled benchmark JSONL: `data/controlled_*.jsonl`, `data/pilot.jsonl`
- `data/background_contamination.json`
- Annotation guideline templates **without** note text (if added separately)
- Split manifests that contain **no** MIMIC subject/HADM IDs tied to labels

### Restricted (PhysioNet credentialed access only)

Do **not** upload to a public repo:

- MIMIC subject/HADM IDs with human or LLM labels
- `data/mimic_struct.jsonl`, `data/mimic_struct_deepseek.jsonl`
- `data/mimic_deepseek_labels.jsonl` (contains `evidence_span` quotes)
- `data/real_clinical_sru*.jsonl`, `data/human*.csv`, audit CSVs
- `mimic-iv-note-deidentified-free-text-clinical-notes-2.2/`
- `outputs/` (embeddings, experiment JSON, logs with note-derived content)

Use `scripts/patch_label_source.py` before sharing derived metrics that
reference the labeling model name.

## Key experiments added for revision

| Task | Script | Output |
|------|--------|--------|
| Real-note 5-fold CV | `scripts/real_clinical_sru_cv.py` | `outputs/real_clinical_sru_cv.json` |
| Real-note 70/30 split | `scripts/real_clinical_sru_experiment.py` | `outputs/real_clinical_sru_experiment.json` |
| Abstention + mitigation tables | `scripts/extract_paper_tables.py` | `outputs/paper_table_summaries.json` |
| Observation labeling config | `configs/observation_labeling.yaml` | — |

## IID split note

The SilentEHR **IID** protocol groups all counterfactual variants from the same
`(background_id, concept_id)` pair into the same fold via a deterministic hash
(`silentehr.py::stable_index`). It is **not** a random row split.

## Supplementary package

Organized reproduction entry points live in [`supplementary/`](supplementary/README.md):
controlled benchmark, artifact controls, MIMIC bridge (Qwen3-12B observation labels),
and mitigation curves.

## Observation labels

MIMIC observation labels use **Qwen/Qwen3-12B-Instruct** (local transformers inference).
See `supplementary/configs/observation_labeling.yaml` and
`supplementary/prompts/observation_status.txt`.

## License

MIMIC-derived data remain under PhysioNet data-use agreements. Code and
controlled synthetic data may be released separately under a permissive license
at submission time.
