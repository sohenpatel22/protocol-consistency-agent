"""Precision, recall, F1, false positives, hard-negative rate, bootstrap CIs and ECE."""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

from ..agent.linkmap import LINK_BY_TYPE


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return {"precision": _r(p), "recall": _r(r), "f1": _r(f1), "tp": tp, "fp": fp, "fn": fn}


def _r(x):
    return None if x is None else round(x, 3)


def summarise(results: list[dict], n_boot: int = 1000, seed: int = 0) -> dict:
    """``results``: one row per case with keys type, hard_negative, detected (bool),
    n_false_pos (int), confidences [(score, is_tp)]."""
    pos = [r for r in results if not r["hard_negative"]]
    neg = [r for r in results if r["hard_negative"]]

    def counts(rows):
        tp = sum(r["detected"] for r in rows if not r["hard_negative"])
        fn = sum(not r["detected"] for r in rows if not r["hard_negative"])
        fp = sum(r["n_false_pos"] for r in rows)
        return tp, fp, fn

    overall = prf(*counts(results))
    by_type = {}
    groups = defaultdict(list)
    for r in pos:
        groups[r["type"]].append(r)
    for t, rows in sorted(groups.items()):
        tp = sum(r["detected"] for r in rows)
        by_type[t] = {"n": len(rows), "recall": _r(tp / len(rows)),
                      "severity": LINK_BY_TYPE[t].severity if t in LINK_BY_TYPE else ""}
    crit = [r for r in pos if r["severity"] == "critical"]
    crit_rec = _r(sum(r["detected"] for r in crit) / len(crit)) if crit else None

    rng = random.Random(seed)
    boots = {"precision": [], "recall": [], "f1": []}
    for _ in range(n_boot if results else 0):
        sample = [rng.choice(results) for _ in results]
        m = prf(*counts(sample))
        for k in boots:
            if m[k] is not None:
                boots[k].append(m[k])
    ci = {k: [_r(float(np.percentile(v, 2.5))), _r(float(np.percentile(v, 97.5)))] if v else None
          for k, v in boots.items()}

    confs = [c for r in results for c in r.get("confidences", [])]
    docs = {r["doc_id"] for r in results}
    return {
        "n_cases": len(results), "n_conflicts": len(pos), "n_hard_negatives": len(neg), "n_docs": len(docs),
        "overall": overall, "ci95": ci, "critical_recall": crit_rec, "by_type": by_type,
        "false_pos_per_case": _r(sum(r["n_false_pos"] for r in results) / len(results)) if results else None,
        "hard_negative_fpr": _r(sum(r["n_false_pos"] > 0 for r in neg) / len(neg)) if neg else None,
        "ece": _r(ece(confs)) if confs else None,
    }


def ece(pairs: list[tuple[float, bool]], bins: int = 10) -> float:
    if not pairs:
        return 0.0
    x = np.array([p for p, _ in pairs])
    y = np.array([float(t) for _, t in pairs])
    total = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (x >= lo) & (x < hi) if i < bins - 1 else (x >= lo) & (x <= hi)
        if m.any():
            total += m.mean() * abs(x[m].mean() - y[m].mean())
    return float(total)
