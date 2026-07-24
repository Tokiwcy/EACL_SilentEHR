# SilentEHR — EACL reproduction package

Minimal public release for **SilentEHR: Auditing Closed-World Extrapolation in Frozen Patient Representations**.

## Layout

```
├── README.md
├── requirements.txt
├── LICENSE
├── configs/
├── data_generation/
│   ├── silentehr.py          # core benchmark engine
│   ├── controlled_corpus.py
│   ├── qwen_observation_labeler.py
│   └── corpus/               # public controlled JSONL (11,466 pairs)
├── experiments/
│   ├── controlled_benchmark.py
│   ├── artifact_controls.py
│   ├── mimic_bridge.py
│   └── mitigation.py
├── evaluation/
├── prompts/
└── outputs/                    # created at runtime
```

MIMIC observation labels use **Qwen/Qwen3-12B-Instruct** (local Hugging Face inference).

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Smoke test (CPU, no MIMIC)

```bash
python experiments/controlled_benchmark.py --model hash
```

Writes `outputs/controlled_hash.json`.

## Controlled benchmark

```bash
python experiments/controlled_benchmark.py --model qwen --device cuda
python experiments/artifact_controls.py
python experiments/mitigation.py --model qwen --device cuda
```

## MIMIC (local credentialed data only)

Create a local `data/` folder (gitignored) with PhysioNet MIMIC-IV exports, then:

```bash
python experiments/mimic_bridge.py pipeline --mimic-dir /path/to/mimic-iv-3.1 --limit 10
```

This builds `data/mimic_struct.jsonl`, labels with Qwen3-12B, and runs the C×O bridge.

## License

| Component | License | Location |
|-----------|---------|----------|
| Code, configs, prompts, docs | **MIT** | [`LICENSE`](LICENSE) |
| Synthetic benchmark JSONL | **CC BY 4.0** | [`data_generation/corpus/LICENSE`](data_generation/corpus/LICENSE) |
| MIMIC-derived artifacts | PhysioNet DUA | Local `data/` only (not redistributed) |

When reusing the corpus, please cite the SilentEHR paper and credit the dataset
as indicated in the corpus license file.
