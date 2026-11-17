"""Fix-up loop for one finding.

1. Re-query: search the same protocol index again, but only for the fields in conflict.
   The queries are that field's own retrieval queries plus the conflicting text itself.
2. Read blind: for each conflicting field, ask what the protocol says, without showing
   the extracted value (so the model can't just agree with it).
3. Decide: compare the blind answers with the extracted values and pick a cause:
     extraction_error       -> corrected value
     missing_condition      -> value with its condition ("X, except when Y")
     protocol_inconsistency -> suggested protocol edit (sponsor) / draft sponsor query (site)
4. Re-check: apply the proposed value to a copy and run the checks again. If the conflict
   is still there, or a new one appears in linked fields, escalate instead.
Nothing is written to the extraction until a reviewer approves (see review()).
"""
from __future__ import annotations

import copy
import json

from ..llm import LLMError, complete_json
from ..replication.elements import element_for_field, get_path, set_path
from ..replication.pdfparse import load_chunks
from ..replication.retrieval import Retriever, format_context
from . import audit
from . import normalize as N
from .linkmap import Finding
from .rules import run_rules

BLIND_SYSTEM = (
    "You read clinical trial protocol excerpts and report what they say about one data field. "
    "Quote the protocol verbatim. If the excerpts say different things in different places, "
    "report each with its quote. Reply with JSON: "
    '{"answers": [{"value": "...", "quote": "...", "page": <int or null>}], "found": true}'
)
DECIDE_SYSTEM = "\n".join([
    "You resolve a cross-section conflict in a structured extraction of a clinical trial protocol.",
    "You get the conflict, the extracted values, and what the protocol itself says about each field",
    "(read independently, with quotes). Decide the cause:",
    "- extraction_error: the protocol is consistent and an extracted value is wrong -> corrected value;",
    "- missing_condition: both values are right but one needs a condition the extraction left out -> "
    "the value with its condition;",
    "- protocol_inconsistency: the protocol text itself disagrees -> a suggested protocol edit for the "
    "sponsor and a short, polite draft query to the sponsor for a site;",
    "- unresolved: the evidence does not settle it.",
    'Reply with JSON: {"cause": "...", "field": "<field id to change or null>", "proposed_value": <value or null>, '
    '"protocol_quote": "<verbatim quote that supports the decision>", "suggested_protocol_edit": "...", '
    '"draft_sponsor_query": "...", "rationale": "..."}',
])


def requery(f: Finding, retriever: Retriever) -> list:
    """Focused re-query: the conflicting fields' own queries plus the conflicting text."""
    queries = []
    for fid in f.fields:
        el = element_for_field(fid)
        if el:
            queries += el.queries
    queries += [e.quote[:300] for e in f.evidence if e.quote.strip()]
    return retriever.retrieve(list(dict.fromkeys(queries)) or [f.explanation], top_k=6, char_budget=20000)


def read_blind(fid: str, excerpt: str) -> dict:
    el = element_for_field(fid)
    label = f"{el.label} ({fid})" if el else fid
    user = f"Data field: {label}\n\nProtocol excerpts:\n{excerpt}"
    out = complete_json("verifier", BLIND_SYSTEM, user, max_tokens=3000)
    answers = out.get("answers", []) if isinstance(out, dict) else []
    for a in answers:
        a["quote_verified"] = bool(a.get("quote")) and N.norm_quote(a["quote"]) in N.norm_quote(excerpt)
    return {"field": fid, "answers": answers}


CONFIRM_SYSTEM = (
    "An automated sweep of a clinical-trial protocol extraction flagged a possible contradiction between "
    "two sections. It only saw the extracted values, not the protocol. You get the suspect, what the "
    "protocol says about each field (read independently, with quotes) and the protocol excerpts. Decide "
    "whether the conflict is real. It is NOT real if the values are equivalent, apply under different "
    "conditions that the protocol states, or one of them is simply not about the same thing. Reply with "
    'JSON: {"confirmed": true|false, "protocol_quote": "<verbatim quote supporting your decision>", '
    '"likely_cause": "extraction_error|missing_condition|protocol_inconsistency|not_a_conflict", "reason": "..."}'
)


def confirm_suspect(nct_id: str, extraction: dict, f: Finding, retriever: Retriever | None = None) -> dict:
    """Open-sweep findings are only suspects: keep one only if the protocol text backs it up."""
    retriever = retriever or Retriever(nct_id, load_chunks(nct_id))
    chunks = requery(f, retriever)
    excerpt = format_context(chunks)
    blind = [read_blind(fid, excerpt) for fid in f.fields[:3]]
    values = {fid: get_path(extraction, fid) for fid in f.fields}
    user = "\n\n".join([
        f"Suspect: {f.explanation}",
        f"Extracted values:\n{json.dumps(values, indent=1, ensure_ascii=False)}",
        "What the protocol says (read without seeing the extracted values):\n"
        + json.dumps(blind, indent=1, ensure_ascii=False),
        f"Protocol excerpts:\n{excerpt}",
    ])
    out = complete_json("verifier", CONFIRM_SYSTEM, user, max_tokens=3000)
    if not isinstance(out, dict):
        out = {}
    quote = out.get("protocol_quote") or ""
    verified = bool(quote) and N.norm_quote(quote) in N.norm_quote(excerpt)
    return {"confirmed": bool(out.get("confirmed")) and verified, "protocol_quote": quote,
            "quote_verified": verified, "likely_cause": out.get("likely_cause", ""), "reason": out.get("reason", ""),
            "pages": sorted({p for c in chunks for p in range(c.page_start, c.page_end + 1)})}


def propose(nct_id: str, extraction: dict, f: Finding, retriever: Retriever | None = None,
            context: dict | None = None) -> dict:
    retriever = retriever or Retriever(nct_id, load_chunks(nct_id))
    chunks = requery(f, retriever)
    excerpt = format_context(chunks)
    blind = [read_blind(fid, excerpt) for fid in f.fields[:3]]
    values = {fid: get_path(extraction, fid) for fid in f.fields}
    user = "\n\n".join([
        f"Conflict type: {f.type}\nWhy it was flagged: {f.explanation}",
        f"Extracted values:\n{json.dumps(values, indent=1, ensure_ascii=False)}",
        "What the protocol says (read without seeing the extracted values):\n"
        + json.dumps(blind, indent=1, ensure_ascii=False),
        f"Protocol excerpts:\n{excerpt}",
    ])
    out = complete_json("verifier", DECIDE_SYSTEM, user, max_tokens=6000)
    if not isinstance(out, dict):
        out = {"cause": "unresolved"}
    quote = out.get("protocol_quote") or ""
    out["quote_verified"] = bool(quote) and N.norm_quote(quote) in N.norm_quote(excerpt)
    out["blind_reads"] = blind
    out["requery_pages"] = sorted({p for c in chunks for p in range(c.page_start, c.page_end + 1)})
    return out


def recheck(extraction: dict, f: Finding, proposal: dict, context: dict | None = None) -> dict:
    """Apply the proposed value to a copy and re-run the rules on the linked fields."""
    field = proposal.get("field")
    if proposal.get("cause") not in ("extraction_error", "missing_condition") or not field \
            or proposal.get("proposed_value") is None or get_path(extraction, field) is None:
        return {"applied": False}
    trial = copy.deepcopy(extraction)
    set_path(trial, field, proposal["proposed_value"])
    before = {x.key() for x in run_rules(extraction, context)}
    after = run_rules(trial, context)
    linked = set(f.fields) | {field}
    still = any(a.type == f.type and set(a.fields) & linked for a in after)
    new = [a for a in after if a.key() not in before and set(a.fields) & linked]
    return {"applied": True, "original_conflict_remains": still,
            "new_conflicts": [{"type": n.type, "fields": n.fields} for n in new]}


def decide(proposal: dict, check: dict) -> str:
    cause = proposal.get("cause")
    if cause == "unresolved" or not proposal.get("quote_verified"):
        return "escalate"
    if cause == "protocol_inconsistency":
        return "sponsor_query"
    if check.get("applied") and not check.get("new_conflicts") and not check.get("original_conflict_remains"):
        return "suggest_fix"
    return "escalate"


def resolve_all(nct_id: str, extraction: dict, findings: list[Finding], context: dict | None = None,
                log=print) -> list[dict]:
    retriever = Retriever(nct_id, load_chunks(nct_id))
    out = []
    for f in findings:
        try:
            prop = propose(nct_id, extraction, f, retriever, context)
        except (LLMError, ValueError) as exc:
            log(f"  resolve {f.id}: ERROR {exc}")
            prop = {"cause": "unresolved", "error": str(exc), "quote_verified": False}
        chk = recheck(extraction, f, prop, context)
        action = decide(prop, chk)
        rec = {"finding_id": f.id, "type": f.type, "severity": f.severity, "proposal": prop,
               "recheck": chk, "action": action, "review": {"status": "pending"}}
        audit.record(nct_id, "proposal", finding_id=f.id, action=action, cause=prop.get("cause"),
                     field=prop.get("field"), quote_verified=prop.get("quote_verified"), recheck=chk)
        out.append(rec)
    return out


def review(nct_id: str, proposals: list[dict], finding_id: str, decision: str, note: str = "",
           extraction: dict | None = None) -> dict:
    """Human gate. An approved fix is applied to the extraction and logged with old and new values."""
    rec = next(p for p in proposals if p["finding_id"] == finding_id)
    who = audit.reviewer()
    rec["review"] = {"status": decision, "note": note, "by": who}
    applied = None
    if decision == "approve" and extraction is not None and rec["action"] == "suggest_fix":
        field = rec["proposal"]["field"]
        applied = {"field": field, "old": get_path(extraction, field), "new": rec["proposal"]["proposed_value"]}
        set_path(extraction, field, rec["proposal"]["proposed_value"])
    audit.record(nct_id, "review", actor=who, finding_id=finding_id, decision=decision, note=note, applied=applied)
    return rec
