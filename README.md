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

## Public vs restricted

### Published on GitHub

- `silentehr.py`, `supplementary/` (reproduction entry points)
- `requirements.txt`, paper sources (`results.tex`, `PAPER.md`)
- Controlled benchmark JSONL: `data/controlled_*.jsonl`, `data/pilot.jsonl`
- `data/background_contamination.json`

### Local only (not in public repo)

- **`scripts/`** — internal analysis, table extraction, human-audit tooling
- MIMIC subject/HADM IDs, note text, LLM labels, human audit CSVs
- `outputs/` (embeddings, experiment JSON with note-derived content)

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
