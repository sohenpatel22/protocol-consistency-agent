"""Score extractions against the reference answers with a GPT-4o judge (0-5 scale).

Section score = Table 7 weighted average, shown as a percentage. Overall = mean of the six
sections. The paper doesn't spell out these two steps, so they are our reading of it."""
from __future__ import annotations

import json
from statistics import mean

from ..config import JUDGEMENTS
from ..llm import LLMError, complete_json
from .annotate import load_gt
from .elements import CATEGORY_CODES, all_elements, get_path
from .extract import load

SCALE = [0, 1, 2, 3, 3.5, 4, 4.5, 5]
SYSTEM = (
    "You grade information extracted from a clinical trial protocol against a reference answer.\n"
    "Score on this scale: 5 = recovers 100% of the reference; 4.5 = at least 90% to a human reader; "
    "4 = about 80%; 3.5 = about 70%; 3 = about 60%; 2 = about 40%; 1 = about 20-40%; 0 = less than 20%.\n"
    "Do not penalise abbreviations, formatting or equivalent expressions (units, synonyms). Extra "
    "information is not penalised unless it contradicts the reference, misrepresents it or adds a "
    "safety risk. Penalise hallucinated content. Reply with JSON: "
    '{"score": <one of 0,1,2,3,3.5,4,4.5,5>, "reason": "..."}'
)


def _snap(x: float) -> float:
    return min(SCALE, key=lambda s: abs(s - x))


def _extracted_value(extraction: dict, el) -> dict:
    cat = extraction.get(el.category, {})
    return {p: get_path(cat, p) for p in el.paths}


def judge_protocol(system: str, nct_id: str, log=print) -> dict:
    ext = load(system, nct_id)["extraction"]
    gt = load_gt(nct_id)["items"]
    scores = {}
    for el in all_elements():
        ref = gt.get(el.key, {}).get("answer", {})
        got = _extracted_value(ext, el)
        if el.key == "soe":
            got = ext.get("schedule_of_events", {})
        user = (f"Data element: {el.label}\nReference answer:\n{json.dumps(ref, ensure_ascii=False)}\n\n"
                f"Extracted answer:\n{json.dumps(got, ensure_ascii=False)}")
        try:
            out = complete_json("judge", SYSTEM, user, max_tokens=1000)
            scores[el.key] = {"score": _snap(float(out.get("score", 0))), "reason": out.get("reason", "")}
        except (LLMError, ValueError, TypeError) as exc:
            log(f"  judge {nct_id} {el.key}: ERROR {exc}")
            scores[el.key] = {"score": None, "reason": f"error: {exc}"}
    res = {"nct_id": nct_id, "system": system, "scores": scores, "categories": category_scores(scores)}
    p = JUDGEMENTS / system / f"{nct_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def category_scores(scores: dict) -> dict:
    """Weighted Table 7 average per category, as a percentage of the 5-point maximum."""
    out = {}
    for code, cat in CATEGORY_CODES.items():
        els = [e for e in all_elements() if e.category == cat and e.weight > 0
               and scores.get(e.key, {}).get("score") is not None]
        wsum = sum(e.weight for e in els)
        if wsum:
            out[code] = round(100 * sum(e.weight * scores[e.key]["score"] for e in els) / wsum / 5, 1)
    if out:
        out["overall"] = round(mean(v for k, v in out.items()), 1)
    return out


def summarise(system: str, nct_ids: list[str]) -> dict:
    per = []
    for nct in nct_ids:
        p = JUDGEMENTS / system / f"{nct}.json"
        if p.exists():
            per.append(json.loads(p.read_text(encoding="utf-8"))["categories"])
    keys = list(CATEGORY_CODES) + ["overall"]
    return {"system": system, "n": len(per),
            "mean": {k: round(mean(c[k] for c in per if k in c), 1) for k in keys if any(k in c for c in per)}}
