"""Check the AI judge against a person, with the same statistics as the paper
(Pearson, Spearman, linear/quadratic weighted kappa, MAE, bias, exact and near agreement).

1. ``export_sample`` writes a CSV of extracted vs reference answers, *without* the judge's score.
2. A person fills in ``human_score`` (0, 1, 2, 3, 3.5, 4, 4.5 or 5).
3. ``agreement`` compares those scores with the judge's.
"""
from __future__ import annotations

import csv
import json
import random
from statistics import mean

from ..config import JUDGEMENTS, SAMPLE_SEED
from .annotate import load_gt
from .elements import all_elements
from .extract import load
from .judge import SCALE, _extracted_value


def export_sample(system: str, nct_ids: list[str], n: int = 50) -> str:
    rows = []
    for nct in nct_ids:
        try:
            ext, gt = load(system, nct)["extraction"], load_gt(nct)["items"]
        except FileNotFoundError:
            continue
        for el in all_elements():
            got = ext.get("schedule_of_events", {}) if el.key == "soe" else _extracted_value(ext, el)
            rows.append({"system": system, "nct_id": nct, "element": el.key, "label": el.label,
                         "reference": json.dumps(gt.get(el.key, {}).get("answer", {}), ensure_ascii=False),
                         "extracted": json.dumps(got, ensure_ascii=False), "human_score": ""})
    rows = random.Random(SAMPLE_SEED).sample(rows, min(n, len(rows)))
    path = JUDGEMENTS / f"human_check_{system}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["system"])
        w.writeheader()
        w.writerows(rows)
    return str(path)


def _rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def _pearson(a: list[float], b: list[float]) -> float | None:
    ma, mb = mean(a), mean(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if not va or not vb:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb) ** 0.5


def weighted_kappa(a: list[float], b: list[float], power: int) -> float | None:
    """Cohen's kappa on the paper's 8-point scale; power 1 = linear, 2 = quadratic weights."""
    cats = SCALE
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    obs = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1
    n = len(a)
    ra = [sum(row) for row in obs]
    rb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    w = [[(abs(i - j) / (k - 1)) ** power for j in range(k)] for i in range(k)]
    o = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k)) / n
    e = sum(w[i][j] * ra[i] * rb[j] for i in range(k) for j in range(k)) / (n * n)
    return None if e == 0 else 1 - o / e


def agreement(csv_path: str) -> dict:
    human, ai = [], []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("human_score", "").strip():
                continue
            j = json.loads((JUDGEMENTS / r["system"] / f"{r['nct_id']}.json").read_text(encoding="utf-8"))
            s = j["scores"].get(r["element"], {}).get("score")
            if s is None:
                continue
            human.append(min(SCALE, key=lambda c: abs(c - float(r["human_score"]))))
            ai.append(s)
    if len(human) < 3:
        return {"n": len(human), "note": "need at least 3 human-scored rows"}
    diff = [x - y for x, y in zip(ai, human)]
    pr, sr = _pearson(ai, human), _pearson(_rank(ai), _rank(human))
    res = {"n": len(human),
           "pearson_r": None if pr is None else round(pr, 3),
           "spearman_rho": None if sr is None else round(sr, 3),
           "kappa_linear": weighted_kappa(ai, human, 1), "kappa_quadratic": weighted_kappa(ai, human, 2),
           "mae": round(mean(abs(d) for d in diff), 3), "bias_ai_minus_human": round(mean(diff), 3),
           "exact_agreement": round(mean(d == 0 for d in diff), 3),
           "within_half_point": round(mean(abs(d) <= 0.5 for d in diff), 3),
           "paper": {"n": 95, "pearson_r": 0.923, "spearman_rho": 0.942, "kappa_quadratic": 0.877, "mae": 0.553,
                     "bias_ai_minus_human": -0.468, "exact_agreement": 0.432, "within_half_point": 0.642}}
    for k in ("kappa_linear", "kappa_quadratic"):
        if res[k] is not None:
            res[k] = round(res[k], 3)
    return res
