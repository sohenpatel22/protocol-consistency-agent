"""Reference answers (ground truth) built the way the paper describes.

Two models answer each element, a third (GPT-5) merges them and rates its confidence, and
the 40 least confident items plus 24 random ones go to a CSV for a person to check.
The paper used Claude 3 as one annotator; it is retired, so Claude Haiku 4.5 stands in."""
from __future__ import annotations

import csv
import json
import random

from ..config import GROUND_TRUTH, SAMPLE_SEED
from ..llm import LLMError, complete_json
from .elements import Element, all_elements
from .extract import SYSTEM as EXTRACT_SYSTEM, element_prompt
from .pdfparse import load_chunks
from .retrieval import Retriever, format_context
from .soe import PROMPT as SOE_PROMPT, SYSTEM as SOE_SYSTEM, candidate_pages, render
from ..config import CACHE, RAW

ADJ_SYSTEM = (
    "You adjudicate candidate answers for a clinical-trial protocol annotation task. Using the "
    "protocol excerpts as the source of truth, produce the single best answer in the same JSON "
    "shape, combining correct parts of the candidates and fixing errors. Then rate your confidence "
    "that the final answer is complete and correct from 0 to 1. Reply with JSON: "
    '{"answer": {...}, "confidence": 0.0, "notes": "..."}'
)


def _candidates(el: Element, retriever: Retriever, nct_id: str) -> tuple[list[dict], str]:
    if el.key == "soe":
        pdf = RAW / f"{nct_id}.pdf"
        pages = candidate_pages(pdf)
        images = render(pdf, pages, CACHE / nct_id / "soe")
        # gpt-4o is text-only in this project, so both SoE annotators are Claude vision models
        cands = [complete_json("annotator_b", SOE_SYSTEM, SOE_PROMPT, images=images),
                 complete_json("soe_vision", SOE_SYSTEM, SOE_PROMPT, images=images)]
        return cands, f"(schedule-of-events pages {[p + 1 for p in pages]})"
    chunks = retriever.retrieve(el.queries, top_k=(el.top_k or 8) + 6, char_budget=45_000)
    ctx = format_context(chunks)
    prompt = element_prompt(el, ctx)
    cands = [complete_json(r, EXTRACT_SYSTEM, prompt, max_tokens=8000) for r in ("annotator_a", "annotator_b")]
    return cands, ctx


def annotate_protocol(nct_id: str, log=print) -> dict:
    """Build the reference answers. If a file from an earlier run has failed items, only those are redone.
    Raises RuntimeError when more than half fail (almost always a key, quota or model-name problem)."""
    retriever = Retriever(nct_id, load_chunks(nct_id))
    try:
        items = load_gt(nct_id)["items"]
    except FileNotFoundError:
        items = {}
    failed_before = {k for k, v in items.items() if str(v.get("notes", "")).startswith("error:")}
    todo = [el for el in all_elements() if el.key not in items or el.key in failed_before]
    for el in todo:
        try:
            cands, ctx = _candidates(el, retriever, nct_id)
            user = (f"Data element: {el.label}\nCandidate A:\n{json.dumps(cands[0])}\n\n"
                    f"Candidate B:\n{json.dumps(cands[1])}\n\nProtocol excerpts:\n{ctx[:60000]}")
            adj = complete_json("adjudicator", ADJ_SYSTEM, user, max_tokens=12000)
            if not isinstance(adj, dict) or "answer" not in adj:
                raise ValueError("the adjudicator's reply had no 'answer'")
            items[el.key] = {"label": el.label, "category": el.category,
                             "answer": adj.get("answer", {}), "confidence": float(adj.get("confidence", 0)),
                             "notes": adj.get("notes", ""), "candidates": cands, "human_reviewed": False}
        except (LLMError, ValueError, AttributeError, TypeError, IndexError) as exc:
            log(f"  {nct_id} {el.key}: ERROR {exc}")
            items[el.key] = {"label": el.label, "category": el.category, "answer": {}, "confidence": 0.0,
                             "notes": f"error: {exc}", "candidates": [], "human_reviewed": False}
    out = {"nct_id": nct_id, "items": items}
    errored = [k for k, v in items.items() if str(v.get("notes", "")).startswith("error:")]
    if len(errored) > len(all_elements()) / 2:
        first = items[errored[0]]["notes"]
        raise RuntimeError(f"{len(errored)} of {len(items)} reference answers failed for {nct_id}. {first[:300]}")
    p = GROUND_TRUTH / f"{nct_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def gt_complete(nct_id: str) -> bool:
    """True when every element has a reference answer that did not fail."""
    try:
        items = load_gt(nct_id)["items"]
    except FileNotFoundError:
        return False
    return len(items) >= len(all_elements()) and not any(str(v.get("notes", "")).startswith("error:") for v in items.values())


def load_gt(nct_id: str) -> dict:
    return json.loads((GROUND_TRUTH / f"{nct_id}.json").read_text(encoding="utf-8"))


def write_review_queue(nct_ids: list[str], n_uncertain: int = 40, n_random: int = 24) -> str:
    rows = []
    for nct in nct_ids:
        try:
            gt = load_gt(nct)
        except FileNotFoundError:
            continue
        for key, it in gt["items"].items():
            rows.append({"nct_id": nct, "element": key, "label": it["label"], "confidence": it["confidence"],
                         "answer": json.dumps(it["answer"], ensure_ascii=False), "corrected_answer": ""})
    rows.sort(key=lambda r: r["confidence"])
    uncertain = rows[:n_uncertain]
    rest = rows[n_uncertain:]
    rng = random.Random(SAMPLE_SEED)
    chosen = uncertain + rng.sample(rest, min(n_random, len(rest)))
    for r in chosen:
        r["why"] = "least confident" if r in uncertain else "random"
    path = GROUND_TRUTH / "review_queue.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["nct_id", "element", "label", "why", "confidence", "answer",
                                          "corrected_answer"])
        w.writeheader()
        w.writerows(chosen)
    return str(path)


def apply_reviews(path: str | None = None) -> int:
    """Fold a filled-in review_queue.csv back into the ground truth (blank = answer accepted)."""
    path = path or str(GROUND_TRUTH / "review_queue.csv")
    n = 0
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gt = load_gt(r["nct_id"])
            it = gt["items"][r["element"]]
            if r.get("corrected_answer", "").strip():
                it["answer"] = json.loads(r["corrected_answer"])
            it["human_reviewed"] = True
            (GROUND_TRUTH / f"{r['nct_id']}.json").write_text(json.dumps(gt, indent=1), encoding="utf-8")
            n += 1
    return n
