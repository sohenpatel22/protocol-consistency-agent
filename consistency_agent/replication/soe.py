"""Schedule of events: find the table pages, render them, read them with a vision model.

The paper uses Table Transformer to find the pages. We use it if it is installed
(pip install .[tables]) and fall back to a keyword score otherwise."""
from __future__ import annotations

import re
from pathlib import Path

import pymupdf as fitz

from ..config import CACHE, RAW
from ..llm import complete_json

SOE_TITLE = re.compile(r"schedule of (events|assessments|activities|procedures)|study calendar|"
                       r"time and events|flow ?chart|study flowchart", re.I)
VISIT_WORDS = re.compile(r"\b(screening|baseline|cycle|day|week|visit|follow[- ]up|end of (treatment|study)|EOT|C\dD\d)\b", re.I)
MAX_PAGES = 6

SYSTEM = (
    "You read schedule-of-events tables from clinical trial protocols. Use only what is in the "
    "images. Footnote markers can be dropped but keep the footnote meaning in the procedure name "
    "if it changes when a procedure happens. Reply with a single JSON object."
)
PROMPT = (
    "Extract the schedule of events as JSON:\n"
    '{"schedule_of_events": [{"visit_number": 1, "visit_time": "e.g. Screening (Day -28 to -1)", '
    '"procedures": [{"procedure_name": "..."}]}]}\n'
    "One entry per visit/column in order; list every procedure marked for that visit. If the "
    "table spans several pages, merge them. If there is more than one schedule (e.g. per arm), "
    "include all visits and prefix visit_time with the schedule name."
)


def _keyword_score(text: str) -> float:
    """Heuristic SoE score: table title, visit vocabulary, and stand-alone 'X' cells."""
    title = 2.5 if SOE_TITLE.search(text) else 0.0
    visits = min(len(VISIT_WORDS.findall(text)) / 15, 3.0)
    marks = min(len(re.findall(r"(?m)^\s*(?:X|x|✓|✔|●)\s*$", text)) / 10, 4.0)
    toc = -4.0 if len(re.findall(r"\.{5,}\s*\d+", text)) >= 5 else 0.0   # table-of-contents page
    return title + visits + marks + toc


def candidate_pages(pdf: Path) -> list[int]:
    """0-indexed pages most likely to hold the SoE table, in reading order."""
    with fitz.open(pdf) as doc:
        texts = [doc[i].get_text("text") for i in range(doc.page_count)]
    scores = [_keyword_score(t) for t in texts]
    tt = _table_transformer_pages(pdf, [i for i, s in enumerate(scores) if s >= 1.5])
    if tt is not None:
        scores = [s + (2.0 if i in tt else -1.0) for i, s in enumerate(scores)]
    top = max(scores, default=0)
    if top < 1.5:
        return []
    keep = max(1.5, 0.6 * top)
    best = sorted((i for i, s in enumerate(scores) if s >= keep), key=lambda i: -scores[i])[:MAX_PAGES]
    pages = set()
    for i in best:          # tables continue across pages: add following pages that also score
        pages.add(i)
        j = i + 1
        while j < len(texts) and scores[j] >= 1.5 and len(pages) < MAX_PAGES:
            pages.add(j)
            j += 1
    return sorted(pages)[:MAX_PAGES]


def _table_transformer_pages(pdf: Path, pages: list[int]) -> set[int] | None:
    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection
    except ImportError:
        return None
    name = "microsoft/table-transformer-detection"
    proc = AutoImageProcessor.from_pretrained(name)
    model = TableTransformerForObjectDetection.from_pretrained(name)
    found = set()
    with fitz.open(pdf) as doc:
        for i in pages:
            pix = doc[i].get_pixmap(dpi=100)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            inputs = proc(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            res = proc.post_process_object_detection(outputs, threshold=0.8,
                                                     target_sizes=[(img.height, img.width)])[0]
            if len(res["scores"]):
                found.add(i)
    return found


def render(pdf: Path, pages: list[int], out_dir: Path, dpi: int = 150) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with fitz.open(pdf) as doc:
        for i in pages:
            p = out_dir / f"page_{i + 1:03d}.png"
            if not p.exists():
                doc[i].get_pixmap(dpi=dpi).save(p)
            paths.append(p)
    return paths


def extract_soe(nct_id: str) -> tuple[dict, dict]:
    pdf = RAW / f"{nct_id}.pdf"
    pages = candidate_pages(pdf)
    if not pages:
        return {"schedule_of_events": []}, {"pages": [], "note": "no schedule-of-events table found"}
    images = render(pdf, pages, CACHE / nct_id / "soe")
    out = complete_json("soe_vision", SYSTEM, PROMPT, images=images, max_tokens=16000)
    if not isinstance(out, dict) or "schedule_of_events" not in out:
        out = {"schedule_of_events": out if isinstance(out, list) else []}
    return out, {"pages": [p + 1 for p in pages]}
