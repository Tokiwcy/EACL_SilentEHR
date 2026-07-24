"""Controlled SilentEHR corpus generation."""
from __future__ import annotations

from pathlib import Path

from supplementary.paths import DATA_DIR, ensure_repo_on_path

ensure_repo_on_path()


def generate_controlled(full: bool = True) -> Path:
    from silentehr import FULL_DATA, make_pilot  # noqa: WPS433

    return make_pilot(full=full, out=FULL_DATA if full else DATA_DIR / "pilot.jsonl")


def generate_artifact_tiers(source: Path | None = None) -> dict[str, Path]:
    from silentehr import make_artifact_tiers  # noqa: WPS433

    return make_artifact_tiers(source)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["full", "artifacts"])
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    if args.cmd == "full":
        path = generate_controlled(full=not args.pilot)
        print(f"wrote {path}")
    else:
        paths = generate_artifact_tiers()
        for name, path in paths.items():
            print(f"{name}: {path}")
