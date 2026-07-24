#!/usr/bin/env python3
"""Shared binomial statistics helpers."""
from __future__ import annotations

import math


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def wilson_summary(successes: int, n: int, z: float = 1.96) -> dict | None:
    if n <= 0:
        return None
    lo, hi = wilson_ci(successes, n, z=z)
    return {
        "n": n,
        "k": successes,
        "p": successes / n,
        "ci95": [lo, hi],
    }


def format_wilson_tex(summary: dict | None, digits: int = 2) -> str:
    if not summary:
        return "---"
    p = summary["p"]
    lo, hi = summary["ci95"]
    fmt = f".{digits}f"
    return (
        f"{p:{fmt}} [{lo:{fmt}},{hi:{fmt}}]"
    )
