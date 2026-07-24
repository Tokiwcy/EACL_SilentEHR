# SilentEHR — Supplementary Reproduction Package

Reproduction code for **SilentEHR: Auditing Closed-World Extrapolation in Frozen Patient Representations** (EACL submission).

MIMIC observation labels use **Qwen/Qwen3-12B-Instruct** (local Hugging Face inference). Encoder experiments use frozen embedding models (Qwen3-Emb, Jina v5, BioLORD, etc.) via the root `silentehr.py` driver.

## Layout

```
supplementary/
├── README.md
├── requirements.txt
├── LICENSE
├── configs/observation_labeling.yaml
├── prompts/observation_status.txt
├── data_generation/
│   ├── controlled_corpus.py
│   └── qwen_observation_labeler.py
├── experiments/
│   ├── controlled_benchmark.py
│   ├── artifact_controls.py
│   ├── mimic_bridge.py
│   └── mitigation.py
├── evaluation/metrics.py
└── outputs/          # experiment JSON (created at runtime)
```

The repository root contains `silentehr.py` and public controlled JSONL under `data/`.
Internal one-off analysis scripts under `scripts/` stay local and are not published.

## Setup

```bash
cd /path/to/EACL_SilentEHR
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r supplementary/requirements.txt
```

**GPU:** Qwen3-12B labeling and neural encoders expect a CUDA GPU with ≥24 GB VRAM (12B bf16). Controlled benchmark smoke tests run on CPU with `--model hash` or `--model tfidf`.

## Quick smoke test (no MIMIC, no GPU for hash)

```bash
python supplementary/experiments/controlled_benchmark.py --model hash
```

Expected: writes `supplementary/outputs/controlled_hash.json` with UNR metrics on `data/controlled_full.jsonl`.

## Controlled benchmark

```bash
# Regenerate 11,466-pair corpus (optional)
python supplementary/data_generation/controlled_corpus.py full

# New-background protocol, any encoder
python supplementary/experiments/controlled_benchmark.py --model qwen --device cuda
python supplementary/experiments/controlled_benchmark.py --model biolord --device cuda
```

## Artifact controls

```bash
python supplementary/experiments/artifact_controls.py --regenerate
```

## Mitigation (few-shot Unknown supervision)

```bash
python supplementary/experiments/mitigation.py --model qwen --device cuda
```

## MIMIC bridge (credentialed data required)

1. Obtain [MIMIC-IV](https://physionet.org/content/mimiciv/) and [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/) under PhysioNet credentialed access.
2. Build struct cohort (from repo root):

```bash
python silentehr.py mimic --mimic-dir /path/to/mimic-iv-3.1 --reuse
```

3. Label observations with Qwen3-12B (resume-safe):

```bash
python supplementary/data_generation/qwen_observation_labeler.py label --limit 10
python supplementary/data_generation/qwen_observation_labeler.py build-struct
```

Or use the experiment wrapper:

```bash
python supplementary/experiments/mimic_bridge.py label --limit 10
python supplementary/experiments/mimic_bridge.py cxo --model hash
```

**Do not commit** `data/mimic_*.jsonl`, human audit CSVs, or `outputs/` containing note text or subject IDs. See `.gitignore`.

## Observation labeling config

- Model: `Qwen/Qwen3-12B-Instruct`
- Prompt: `supplementary/prompts/observation_status.txt`
- YAML: `supplementary/configs/observation_labeling.yaml`
- Output: `data/mimic_qwen3_labels.jsonl` → overlay `data/mimic_struct_qwen3.jsonl`

## Human audit gold (restricted)

Dual-annotator consensus labels live in `data/real_clinical_sru.jsonl` (`label_tier=human_consensus`). Raw annotator sheets: `data/human1.csv`, `data/human2.csv`. These files are **not** included in the public repository.

## Citation

If you use this code, please cite the SilentEHR paper (EACL 2026, anonymous submission).
