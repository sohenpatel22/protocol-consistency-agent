"""Merge rule and LLM findings, set severity, and calibrate confidence.

Calibration is Platt scaling fitted on the benchmark (results/calibration.json)."""
from __future__ import annotations

import json
import math

import numpy as np

from ..config import RESULTS
from .linkmap import LINK_BY_TYPE, SEVERITY_RANK, Finding

CAL_FILE = RESULTS / "calibration.json"
SAFETY = ("dose", "discontinuation", "inclusion", "concomitant", "sae_definition")
LEVELS = ["minor", "major", "critical"]


def merge(rule_f: list[Finding], llm_f: list[Finding]) -> list[Finding]:
    """Union; a verifier finding that overlaps a rule finding of the same type confirms it."""
    out = list(rule_f)
    for lf in llm_f:
        twin = next((r for r in out if r.type == lf.type and set(r.fields) & set(lf.fields)), None)
        if twin is None:
            out.append(lf)
        else:
            twin.source = "hybrid"
            twin.confidence = 1 - (1 - twin.confidence) * (1 - lf.confidence)
            twin.explanation = f"{twin.explanation} Verifier: {lf.explanation}".strip()
    return out


def assign_severity(f: Finding) -> str:
    base = LINK_BY_TYPE[f.type].severity if f.type in LINK_BY_TYPE else "minor"
    lvl = LEVELS.index(base)
    if lvl < 2 and any(s in fid for fid in f.fields for s in SAFETY):
        lvl += 1
    return LEVELS[lvl]


def fit_platt(scores: list[float], labels: list[int], iters: int = 2000, lr: float = 0.1) -> dict:
    x = np.array(scores, dtype=float)
    y = np.array(labels, dtype=float)
    a, b = 1.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a * x + b)))
        a -= lr * float(np.mean((p - y) * x))
        b -= lr * float(np.mean(p - y))
    return {"a": a, "b": b, "n": len(x)}


def save_calibration(params: dict) -> None:
    CAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    CAL_FILE.write_text(json.dumps(params, indent=1), encoding="utf-8")


def _cal() -> dict | None:
    try:
        return json.loads(CAL_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def rank(findings: list[Finding]) -> list[Finding]:
    cal = _cal()
    for i, f in enumerate(findings):
        f.severity = assign_severity(f)
        if cal:
            f.calibrated = round(1 / (1 + math.exp(-(cal["a"] * f.confidence + cal["b"]))), 3)
        f.confidence = round(f.confidence, 3)
    findings.sort(key=lambda f: (-SEVERITY_RANK[f.severity], -(f.calibrated if f.calibrated is not None else f.confidence)))
    for i, f in enumerate(findings):
        f.id = f"F{i + 1:03d}"
    return findings
