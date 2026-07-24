"""Repository path helpers for supplementary scripts."""
from __future__ import annotations

import sys
from pathlib import Path

SUP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SUP_ROOT.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = SUP_ROOT / "outputs"


def ensure_repo_on_path() -> Path:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return REPO_ROOT
