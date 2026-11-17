"""The paper's baseline: the whole protocol text in one GPT-4o prompt per section."""
from __future__ import annotations

import json

from ..config import RAW, SETTINGS
from ..llm import LLMError, complete_json
from .elements import SCHEMAS, elements_for, set_path, empty_category
from .pdfparse import full_text

SYSTEM = (
    "You extract study information from the full text of a clinical trial protocol. Copy values "
    "faithfully. If a value is not present, leave it empty; never guess. Reply with one JSON object."
)


def extract_protocol(nct_id: str, log=print) -> dict:
    text = full_text(RAW / f"{nct_id}.pdf")
    truncated = len(text) > SETTINGS.standalone_char_limit
    if truncated:
        log(f"  {nct_id}: protocol text {len(text):,} chars, truncated to {SETTINGS.standalone_char_limit:,}")
        text = text[:SETTINGS.standalone_char_limit]
    doc, errors = {}, {}
    for cat in SCHEMAS:
        els = elements_for(cat)
        template = {p: v for el in els for p, v in el.template().items()}
        labels = ", ".join(el.label for el in els)
        user = (f"Extract these data elements: {labels}.\nFill this JSON object (keys are fixed schema "
                f"paths; list placeholders show one item's shape):\n{json.dumps(template, indent=1)}\n\n"
                f"Protocol:\n{text}")
        doc[cat] = empty_category(cat)
        try:
            out = complete_json("baseline", SYSTEM, user, max_tokens=12000)
        except (LLMError, ValueError) as exc:
            errors[cat] = str(exc)
            continue
        for p in template:
            if isinstance(out, dict) and p in out:
                set_path(doc[cat], p, out[p])
    return {"nct_id": nct_id, "system": "standalone", "extraction": doc, "provenance": {},
            "errors": errors, "truncated": truncated}
