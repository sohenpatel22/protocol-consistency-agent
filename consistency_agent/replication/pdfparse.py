"""Split a protocol PDF into chunks that follow its section headings.

Each chunk keeps its heading path and page range. Long sections are split into parts that
share a group id so retrieval can put them back together. Chunk sizes are our choice; the
paper doesn't give them."""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf as fitz

from ..config import CACHE, RAW, SETTINGS

NUMBERED = re.compile(r"^(?:[Ss]ection\s+)?(\d{1,2}(?:\.\d{1,2}){0,4})\.?\s+([A-Z].{1,110})$")
TOC_LINE = re.compile(r"(\.{4,}|…{2,}|(\.\s){4,})\s*\d+\s*$")   # "Inclusion Criteria ........ 23"
APPENDIX = re.compile(r"^(appendix|attachment|annex)\s+[A-Z0-9]+\b.{0,100}$", re.I)


@dataclass
class Chunk:
    id: str
    order: int
    group: str                 # parts of one split section share a group
    part: int
    n_parts: int
    path: list[str]            # heading hierarchy, outermost first
    page_start: int
    page_end: int
    text: str

    def header(self) -> str:
        return " > ".join(self.path) if self.path else "(front matter)"


@dataclass
class _Section:
    path: list[str]
    page_start: int
    page_end: int
    lines: list[str] = field(default_factory=list)


def _lines(doc: fitz.Document) -> list[tuple[int, str, float, bool]]:
    """(page_no, text, max_font_size, is_bold) for every text line, page numbers 1-indexed."""
    out = []
    for pno in range(doc.page_count):
        d = doc[pno].get_text("dict")
        for block in d.get("blocks", []):
            for line in block.get("lines", []):
                spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                if not spans:
                    continue
                text = " ".join(s["text"].strip() for s in spans).strip()
                size = max(s.get("size", 0) for s in spans)
                bold = all(s.get("flags", 0) & 16 for s in spans)
                out.append((pno + 1, text, size, bold))
    return out


def _boilerplate(lines, n_pages: int) -> set[str]:
    """Running headers/footers: short lines repeated on more than half of the pages."""
    if n_pages < 5:
        return set()
    per_page: dict[int, set[str]] = {}
    for p, t, *_ in lines:
        key = re.sub(r"\d+", "#", t.strip().lower())
        if len(key) <= 120:
            per_page.setdefault(p, set()).add(key)
    counts = Counter(k for keys in per_page.values() for k in keys)
    return {k for k, c in counts.items() if c > n_pages * 0.5}


def _heading_level(text: str, size: float, bold: bool, body: float) -> int | None:
    t = text.strip()
    if len(t) < 3 or len(t) > 120 or t.endswith((".", ",", ";")) and not NUMBERED.match(t):
        return None
    if re.fullmatch(r"[\d\W]+", t):
        return None
    m = NUMBERED.match(t)
    if m:
        styled = bold or size >= body + 0.5 or t.isupper()
        # "5.1 Inclusion Criteria" is a heading; "1. Age 18 years or older." is a list item
        multi = "." in m.group(1) and not t.endswith(".") and len(t) < 80
        return m.group(1).count(".") + 1 if styled or multi else None
    if APPENDIX.match(t):
        return 1
    if (bold and size >= body) or size >= body + 1.5:
        return 1 if size >= body + 1.5 else 2
    return None


def parse_pdf(pdf: Path, max_chunk_chars: int | None = None) -> list[Chunk]:
    max_chunk_chars = max_chunk_chars or SETTINGS.max_chunk_chars
    with fitz.open(pdf) as doc:
        lines = _lines(doc)
        n_pages = doc.page_count
    if not lines:
        return []
    body = Counter(round(s * 2) / 2 for _, _, s, _ in lines).most_common(1)[0][0]
    skip = _boilerplate(lines, n_pages)

    sections: list[_Section] = [_Section(path=[], page_start=1, page_end=1)]
    stack: list[tuple[int, str]] = []
    for page, text, size, bold in lines:
        if re.sub(r"\d+", "#", text.strip().lower()) in skip or TOC_LINE.search(text):
            continue
        level = _heading_level(text, size, bold, body)
        if level is not None:
            stack = [(lv, h) for lv, h in stack if lv < level] + [(level, text.strip())]
            sections.append(_Section(path=[h for _, h in stack], page_start=page, page_end=page))
            continue
        cur = sections[-1]
        cur.lines.append(text)
        cur.page_end = page

    chunks: list[Chunk] = []
    order = 0
    for si, sec in enumerate(sections):
        body_text = _join(sec.lines)
        if not body_text.strip():
            continue
        parts = _split(body_text, max_chunk_chars)
        for pi, part in enumerate(parts):
            chunks.append(Chunk(id=f"c{order:04d}", order=order, group=f"s{si:04d}", part=pi,
                                n_parts=len(parts), path=sec.path, page_start=sec.page_start,
                                page_end=sec.page_end, text=part))
            order += 1
    return chunks


def _join(lines: list[str]) -> str:
    """Re-flow lines; keep list items and numbered items on their own lines."""
    out: list[str] = []
    for ln in lines:
        if out and not re.match(r"^(•|-|\*|\(?[a-z0-9]{1,3}[\).]|\d+\.)\s", ln) and not out[-1].endswith((".", ":", ";")):
            out[-1] = out[-1] + " " + ln
        else:
            out.append(ln)
    return "\n".join(out)


def _split(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for para in text.split("\n"):
        if cur and len(cur) + len(para) + 1 > limit:
            parts.append(cur)
            cur = ""
        while len(para) > limit:                     # pathological very long paragraph
            parts.append(para[:limit])
            para = para[limit:]
        cur = f"{cur}\n{para}" if cur else para
    if cur:
        parts.append(cur)
    return parts


def full_text(pdf: Path) -> str:
    with fitz.open(pdf) as doc:
        return "\n".join(doc[i].get_text("text") for i in range(doc.page_count))


def load_chunks(nct_id: str) -> list[Chunk]:
    """Parse once per protocol and cache the chunks as JSON."""
    cache = CACHE / nct_id / "chunks.json"
    if cache.exists():
        return [Chunk(**c) for c in json.loads(cache.read_text(encoding="utf-8"))]
    chunks = parse_pdf(RAW / f"{nct_id}.pdf")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([asdict(c) for c in chunks]), encoding="utf-8")
    return chunks
