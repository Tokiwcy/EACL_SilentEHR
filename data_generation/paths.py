"""Path helpers for the SilentEHR reproduction package."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "data_generation" / "corpus"
OUTPUT_DIR = REPO_ROOT / "outputs"
MIMIC_DATA_DIR = REPO_ROOT / "data"  # optional local credentialed MIMIC files


def ensure_repo_on_path() -> Path:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return REPO_ROOT
