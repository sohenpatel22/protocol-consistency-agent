"""LLM check for each link in the link map.

The model must quote the fields it relies on. Quotes are checked against the stored values
and a conflict with a made-up quote is dropped."""
from __future__ import annotations

import json

from ..llm import LLMError, complete_json
from . import normalize as N
from .linkmap import LINKS, Evidence, Finding, Link, link_fields

SYSTEM = (
    "You check a clinical-trial protocol extraction for contradictions between sections. You are "
    "given one type of cross-section relationship and the relevant field values, each with a field id. "
    "Report ONLY genuine conflicts of the stated type. Equivalent wordings are NOT conflicts "
    "(unit conversions such as 1.5 x 10^9/L = 1500/uL, 2-8 C = 36-46 F, Cycle 2 Day 1 of a 21-day "
    "cycle = Day 22, 14 weeks = 98 days, Phase II = Phase 2, 0.2 g = 200 mg). Missing information "
    "on one side is a conflict only when the question says so. For every conflict quote the exact "
    "words from each field you rely on. Reply with JSON: "
    '{"conflicts": [{"fields": ["<field id>", ...], "quotes": [{"field": "<field id>", "quote": "..."}], '
    '"explanation": "...", "likely_cause": "extraction_error|missing_condition|protocol_inconsistency", '
    '"confidence": 0.0}]}. Return {"conflicts": []} if there are none.'
)
MAX_CHARS = 2500


def _block(fs) -> str:
    return "\n".join(f"- {f.id}: {f.text[:MAX_CHARS]}" for f in fs)


def verify_link(link: Link, fields: list[N.Field], context: dict | None = None) -> tuple[list[Finding], int]:
    a, b = link_fields(fields, link)
    if not a or not b:
        return [], 0
    by_id = {f.id: f for f in a + b}
    extra = ""
    if link.type == "nct_id_invalid" and context and context.get("registry_nct_id"):
        extra = f"\nRegistry NCT ID: {context['registry_nct_id']}"
    if link.type == "duration_vs_schedule":
        cyc = [f for f in fields if f.id.startswith("interventions.") and N.cycle_length_days(f.text)]
        if cyc:
            extra = f"\nCycle information: {cyc[0].id}: {cyc[0].text[:300]}"
    user = (f"Relationship type: {link.type}\nQuestion: {link.question}\n\nSide A fields:\n{_block(a)}\n\n"
            f"Side B fields:\n{_block(b) if link.a != link.b else '(same as side A)'}{extra}")
    out = complete_json("verifier", SYSTEM, user, max_tokens=8000)
    findings, dropped = [], 0
    for c in (out.get("conflicts", []) if isinstance(out, dict) else []):
        quotes = c.get("quotes") or []
        ev = []
        ok = bool(quotes)
        for q in quotes:
            f = by_id.get(q.get("field", ""))
            if not f or not q.get("quote") or N.norm_quote(q["quote"]) not in N.norm_quote(f.text):
                ok = False
                break
            ev.append(Evidence(f.id, q["quote"]))
        ids = sorted({e.field for e in ev} | {i for i in c.get("fields", []) if i in by_id})
        if link.type == "other_conflict" and len({i.split(".")[0] for i in ids}) < 2:
            ok = False                       # the open sweep only reports conflicts between sections
        if not ok or not ids:
            dropped += 1
            continue
        try:
            conf = float(c.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        findings.append(Finding(type=link.type, fields=ids, evidence=ev, explanation=c.get("explanation", ""),
                                source="llm", confidence=max(0.0, min(conf, 1.0)),
                                likely_cause=c.get("likely_cause", "")))
    return findings, dropped


def run_verifier(extraction: dict, context: dict | None = None, log=print) -> tuple[list[Finding], dict]:
    fields = N.flatten(extraction)
    findings, stats = [], {"dropped_unquoted": 0, "errors": 0, "calls": 0}
    for link in LINKS:
        try:
            fs, dropped = verify_link(link, fields, context)
            stats["calls"] += 1
        except (LLMError, ValueError) as exc:
            log(f"  verifier {link.type}: ERROR {exc}")
            stats["errors"] += 1
            continue
        findings.extend(fs)
        stats["dropped_unquoted"] += dropped
    return findings, stats


def dumps(findings: list[Finding]) -> str:
    return json.dumps([f.to_dict() for f in findings], indent=1)
