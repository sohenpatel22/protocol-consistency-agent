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


def element_prompt(el: Element, context: str, cite: bool = False) -> str:
    return (
        f"Data element: {el.label}\n"
        f"{el.guide}\n\n"
        "Fill this JSON object. Keys are fixed paths in our schema; keep every key, fill the values. "
        "List placeholders show the shape of one item; return as many items as the protocol has.\n"
        f"{json.dumps(el.template(), indent=1)}"
        f"{CITE if cite else ''}\n\n"
        f"Protocol excerpts:\n{context}"
    )


def run_element(el: Element, retriever: Retriever) -> tuple[dict, dict]:
    chunks = retriever.retrieve(el.queries, top_k=el.top_k)
    context = format_context(chunks)
    cite = SETTINGS.field_citations
    out = complete_json("extractor", SYSTEM, element_prompt(el, context, cite), max_tokens=8000)
    if not isinstance(out, dict):
        out = {}
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


def extract_protocol(nct_id: str, workers: int = 6, include_soe: bool = True, log=print) -> dict:
    chunks = load_chunks(nct_id)
    retriever = Retriever(nct_id, chunks)
    retriever.retrieve(["warm up"])                       # embed once before threads start
    doc = {cat: empty_category(cat) for cat in SCHEMAS if cat != "schedule_of_events"}
    provenance: dict[str, dict] = {}
    errors: dict[str, str] = {}

    def work(el: Element):
        try:
            return el, *run_element(el, retriever), None
        except (LLMError, ValueError) as exc:
            return el, {}, {}, str(exc)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for el, values, prov, err in pool.map(work, ELEMENTS):
            if err:
                errors[el.key] = err
                log(f"  {nct_id} {el.key}: ERROR {err}")
                continue
            for path, v in values.items():
                set_path(doc[el.category], path, v)
            provenance[el.key] = prov

    if include_soe:
        from .soe import extract_soe
        try:
            soe, prov = extract_soe(nct_id)
            doc["schedule_of_events"] = soe
            provenance["soe"] = prov
        except (LLMError, ValueError) as exc:
            errors["soe"] = str(exc)
            doc["schedule_of_events"] = {"schedule_of_events": []}
    return {"nct_id": nct_id, "system": "rag", "extraction": doc, "provenance": provenance, "errors": errors}


def out_path(system: str, nct_id: str) -> Path:
    return EXTRACTIONS / system / f"{nct_id}.json"


def save(result: dict) -> Path:
    p = out_path(result["system"], result["nct_id"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return p


def load(system: str, nct_id: str) -> dict:
    return json.loads(out_path(system, nct_id).read_text(encoding="utf-8"))
