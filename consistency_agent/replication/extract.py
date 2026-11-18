"""The paper's RAG pipeline: one retrieval and one LLM call per data element.

Writes data/extractions/rag/<NCT>.json: the six sections in the paper's format, plus
where each answer came from (pages and section headings, and optionally a quote per field).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..agent.normalize import norm_quote
from ..config import EXTRACTIONS, SETTINGS
from ..llm import LLMError, complete_json
from .elements import ELEMENTS, SCHEMAS, Element, empty_category, set_path
from .pdfparse import load_chunks
from .retrieval import Retriever, format_context

SYSTEM = (
    "You are a clinical research coordinator extracting study information from a clinical trial "
    "protocol for site start-up. Use ONLY the protocol excerpts provided. Copy values faithfully "
    "(numbers, units, thresholds, drug names). If a value is not in the excerpts, leave it as an "
    "empty string or empty list; never guess. Reply with a single JSON object and nothing else."
)


CITE = (
    "\n\nAlso add a key \"_evidence\": for each path you filled, give the page number and a short "
    "verbatim quote (under 30 words) from the excerpts, e.g. "
    '{"<path>": {"page": 12, "quote": "..."}}. Each excerpt header shows its section and pages.'
)


CONDITIONS = (
    "\n\nIf a value differs by condition (study part, arm, cohort, cycle, weight, age group, renal "
    "function...), do not pick one: give every value with its condition in the same field, e.g. "
    '"200 mg/m2 (Part A); 175 mg/m2 (Part B)".'
)


def element_prompt(el: Element, context: str, cite: bool = False, conditions: bool = False) -> str:
    return (
        f"Data element: {el.label}\n"
        f"{el.guide}\n\n"
        "Fill this JSON object. Keys are fixed paths in our schema; keep every key, fill the values. "
        "List placeholders show the shape of one item; return as many items as the protocol has.\n"
        f"{json.dumps(el.template(), indent=1)}"
        f"{CONDITIONS if conditions else ''}"
        f"{CITE if cite else ''}\n\n"
        f"Protocol excerpts:\n{context}"
    )


class ExtractionFailed(RuntimeError):
    """Most fields failed: almost always a setup problem (key, quota, model name), not the protocol."""


def _flatten(obj, prefix: str = "") -> dict:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else k))
            out[f"{prefix}.{k}" if prefix else k] = v
    return out


def _coerce(out, el: Element) -> dict:
    """Accept the model's answer if it holds this element's fields, even when nested under another key.

    A reply with none of the expected keys is an error, not an empty answer: otherwise a wrong
    format would silently turn every field blank."""
    if isinstance(out, list) and len(el.paths) == 1:
        out = {el.paths[0]: out}                      # a bare list for a single list-valued field
    if not isinstance(out, dict):
        raise ValueError(f"{el.key}: expected a JSON object, got {type(out).__name__}")
    if any(p in out for p in el.paths):
        return out
    flat = _flatten(out)
    found = {p: flat[p] for p in el.paths if p in flat}
    if not found:                                       # keys given without their section prefix
        short = {p.split(".")[-1]: p for p in el.paths}
        found = {short[k.split(".")[-1]]: v for k, v in flat.items() if k.split(".")[-1] in short}
    if not found:
        raise ValueError(f"{el.key}: the reply held none of the expected fields {el.paths} (keys: {list(out)[:6]})")
    if "_evidence" in out:
        found["_evidence"] = out["_evidence"]
    return found


def run_element(el: Element, retriever: Retriever) -> tuple[dict, dict]:
    chunks = retriever.retrieve(el.queries, top_k=el.top_k)
    context = format_context(chunks)
    cite = SETTINGS.field_citations
    prompt = element_prompt(el, context, cite, SETTINGS.conditional_values)
    out = _coerce(complete_json("extractor", SYSTEM, prompt, max_tokens=16000), el)
    values = {p: out.get(p, el.template()[p]) for p in el.paths}
    # where the answer came from: always the retrieved pages and sections,
    # and with field citations on, a page + quote per field (quote checked against the excerpts)
    prov = {"chunks": [c.id for c in chunks],
            "pages": sorted({p for c in chunks for p in range(c.page_start, c.page_end + 1)}),
            "sections": list(dict.fromkeys(c.header() for c in chunks))}
    if cite and isinstance(out.get("_evidence"), dict):
        fields = {}
        for path, ev in out["_evidence"].items():
            if path in el.paths and isinstance(ev, dict) and ev.get("quote"):
                hit = next((c for c in chunks if norm_quote(ev["quote"]) in norm_quote(c.text)), None)
                fields[path] = {"page": ev.get("page"), "quote": ev["quote"], "verified": hit is not None,
                                "section": hit.header() if hit else None}
        prov["fields"] = fields
    return values, prov


def extract_protocol(nct_id: str, workers: int = 6, include_soe: bool = True, log=print,
                     system: str = "rag", previous: dict | None = None) -> dict:
    """Extract every data element. With ``previous`` (an earlier result that had errors), only the
    elements that failed are run again and everything else is kept, so a resumed run costs little.

    Raises ExtractionFailed when more than half the elements fail, so a setup problem stops the
    run instead of being saved as a finished, empty extraction."""
    chunks = load_chunks(nct_id)
    retriever = Retriever(nct_id, chunks)
    retriever.rank(["warm up"])                            # embed once before threads start
    if previous:
        doc, provenance = previous["extraction"], dict(previous.get("provenance", {}))
        todo_keys = set(previous.get("errors", {}))
    else:
        doc = {cat: empty_category(cat) for cat in SCHEMAS if cat != "schedule_of_events"}
        provenance, todo_keys = {}, None
    errors: dict[str, str] = {}
    todo = [el for el in ELEMENTS if todo_keys is None or el.key in todo_keys]

    def work(el: Element):
        try:
            return el, *run_element(el, retriever), None
        except (LLMError, ValueError) as exc:
            return el, {}, {}, str(exc)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for el, values, prov, err in pool.map(work, todo):
            if err:
                errors[el.key] = err
                log(f"  {nct_id} {el.key}: ERROR {err}")
                continue
            for path, v in values.items():
                set_path(doc[el.category], path, v)
            provenance[el.key] = prov

    if include_soe and (todo_keys is None or "soe" in todo_keys):
        from .soe import extract_soe
        try:
            soe, prov = extract_soe(nct_id)
            doc["schedule_of_events"] = soe
            provenance["soe"] = prov
        except (LLMError, ValueError) as exc:
            errors["soe"] = str(exc)
            log(f"  {nct_id} soe: ERROR {exc}")
            doc.setdefault("schedule_of_events", {"schedule_of_events": []})
    if len(errors) > (len(todo) + 1) / 2:
        first = next(iter(errors.values()))
        raise ExtractionFailed(f"{len(errors)} of {len(todo) + 1} elements failed for {nct_id}. First error: {first}")
    return {"nct_id": nct_id, "system": system, "extraction": doc, "provenance": provenance, "errors": errors,
            "settings": {"retrieval": SETTINGS.retrieval, "rerank": SETTINGS.rerank, "conditional_values":
                         SETTINGS.conditional_values, "field_citations": SETTINGS.field_citations,
                         "extractor": SETTINGS.model("extractor"), "effort": SETTINGS.effort}}


def out_path(system: str, nct_id: str) -> Path:
    return EXTRACTIONS / system / f"{nct_id}.json"


def save(result: dict) -> Path:
    p = out_path(result["system"], result["nct_id"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return p


def load(system: str, nct_id: str) -> dict:
    return json.loads(out_path(system, nct_id).read_text(encoding="utf-8"))


def run_resumable(nct_id: str, system: str, log=print, force: bool = False, **kw) -> tuple[dict, str]:
    """Build an extraction, skip it if it is complete, or re-run only the elements that failed last time."""
    prev = load(system, nct_id) if out_path(system, nct_id).exists() and not force else None
    if prev is not None and not prev.get("errors"):
        return prev, "already built"
    res = extract_protocol(nct_id, system=system, previous=prev, log=log, **kw)
    save(res)
    return res, ("resumed" if prev else "built")
