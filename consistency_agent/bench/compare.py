"""Compare extraction systems on the same protocols, the way the paper compares RAG and standalone.

Paired by protocol (every system sees the same 23 protocols, reference answers and judge):
mean accuracy per section, the difference from the baseline with a paired bootstrap 95% CI,
a Wilcoxon signed-rank p-value (normal approximation), and a paired effect size (Hedges' g_z).
"""
from __future__ import annotations

import json
import math
import random
from statistics import mean, stdev

from ..config import JUDGEMENTS, RESULTS

KEYS = ["gen", "ie", "ae", "inter", "site", "soe", "overall"]


def load_scores(system: str) -> dict[str, dict]:
    out = {}
    for p in (JUDGEMENTS / system).glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        out[d["nct_id"]] = d["categories"]
    return out


def paired_bootstrap(diffs: list[float], n: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(mean(rng.choice(diffs) for _ in diffs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def wilcoxon_p(diffs: list[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank test, normal approximation (fine for n >= ~10)."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n < 5:
        return None
    ranked = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:                                   # average ranks for ties
        j = i
        while j + 1 < n and abs(d[ranked[j + 1]]) == abs(d[ranked[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[ranked[k]] = (i + j) / 2 + 1
        i = j + 1
    w = sum(r for r, x in zip(ranks, d) if x > 0)
    mu, sd = n * (n + 1) / 4, math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w - mu) / sd
    return round(math.erfc(abs(z) / math.sqrt(2)), 4)


def hedges_gz(diffs: list[float]) -> float | None:
    if len(diffs) < 3 or stdev(diffs) == 0:
        return None
    n = len(diffs)
    return round(mean(diffs) / stdev(diffs) * (1 - 3 / (4 * (n - 1) - 1)), 3)


def compare(systems: list[str], baseline: str) -> dict:
    scores = {s: load_scores(s) for s in systems}
    report = {"baseline": baseline, "systems": {}}
    for s in systems:
        sec = {}
        for k in KEYS:
            vals = [c[k] for c in scores[s].values() if k in c]
            row = {"n": len(vals), "mean": round(mean(vals), 1) if vals else None}
            if s != baseline:
                common = [n for n in scores[s] if n in scores[baseline] and k in scores[s][n] and k in scores[baseline][n]]
                diffs = [scores[s][n][k] - scores[baseline][n][k] for n in common]
                if len(diffs) >= 3:
                    lo, hi = paired_bootstrap(diffs)
                    row.update({"diff": round(mean(diffs), 1), "ci95": [round(lo, 1), round(hi, 1)],
                                "wilcoxon_p": wilcoxon_p(diffs), "hedges_gz": hedges_gz(diffs), "n_paired": len(diffs)})
            sec[k] = row
        report["systems"][s] = sec
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "compare.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (RESULTS / "compare.md").write_text(to_markdown(report), encoding="utf-8")
    return report


def to_markdown(report: dict) -> str:
    lines = [f"# Accuracy by system (baseline: {report['baseline']})", "",
             "| System | " + " | ".join(KEYS) + " |", "|---|" + "---|" * len(KEYS)]
    for s, sec in report["systems"].items():
        cells = []
        for k in KEYS:
            r = sec[k]
            if r["mean"] is None:
                cells.append("-")
            elif "diff" in r:
                cells.append(f"{r['mean']} ({r['diff']:+}, CI {r['ci95'][0]} to {r['ci95'][1]})")
            else:
                cells.append(str(r["mean"]))
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines += ["", "Differences are paired by protocol against the baseline; CI = paired bootstrap 95%."]
    return "\n".join(lines)
