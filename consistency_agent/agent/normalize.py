"""Flatten an extraction into fields and parse the values we compare.

All parsers return one unit so equivalent wordings match: 1.5 x 10^9/L = 1500/uL,
2-8 C = 36-46 F, Phase II = Phase 2, 0.2 g = 200 mg."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterator

SECTION_NAMES = {"general_information": "General information", "inclusion_exclusion": "Inclusion/exclusion",
                 "adverse_events": "Adverse events", "interventions": "Interventions",
                 "site_requirements": "Site requirements", "schedule_of_events": "Schedule of events"}


@dataclass(frozen=True)
class Field:
    id: str          # e.g. "interventions.arms[0].interventions[1].dosage"
    section: str     # top-level category
    text: str


def flatten(extraction: dict) -> list[Field]:
    out: list[Field] = []

    def walk(obj: Any, path: str, section: str):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{path}.{k}" if path else k, section)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]", section)
        elif obj not in (None, "") and not isinstance(obj, bool):
            out.append(Field(path, section, str(obj)))

    for section, doc in extraction.items():
        walk(doc, section, section)
    return out


def fields_matching(fields: list[Field], pattern: str) -> list[Field]:
    rx = re.compile(pattern)
    return [f for f in fields if rx.search(f.id)]


def clean(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("×", "x").replace("≤", "<=").replace("≥", ">=").replace("–", "-")
    t = t.replace("−", "-").replace("µ", "u").replace("μ", "u").replace("°", " deg ")
    return re.sub(r"\s+", " ", t).strip()


def norm_quote(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean(text).lower()).strip()


NUM = r"(\d+(?:,\d{3})*(?:\.\d+)?)"


def _num(s: str) -> float:
    return float(s.replace(",", ""))


# ---- durations ------------------------------------------------------------------------
UNIT_DAYS = {"day": 1, "week": 7, "wk": 7, "month": 30.44, "mo": 30.44, "year": 365.25, "yr": 365.25}
DURATION = re.compile(NUM + r"\s*-?\s*(days?|weeks?|wks?|months?|mos?|years?|yrs?)\b", re.I)
WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
           "nine": 9, "ten": 10, "twelve": 12, "eighteen": 18, "twenty-four": 24}


def durations_days(text: str) -> list[float]:
    t = clean(text)
    for w, n in WORDNUM.items():
        t = re.sub(rf"\b{w}\b", str(n), t, flags=re.I)
    out = []
    for n, unit in DURATION.findall(t):
        u = unit.lower().rstrip("s")
        out.append(_num(n) * UNIT_DAYS.get(u, UNIT_DAYS.get(u[:2], 1)))
    return out


# ---- temperatures (deg C ranges) -------------------------------------------------------
TEMP_RANGE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:deg\s*)?([CF])?\s*(?:-|to|and)\s*(-?\d+(?:\.\d+)?)\s*(?:deg\s*)?([CF])\b", re.I)
TEMP_ONE = re.compile(r"(?:at|below|<=?|>=?|or below|not exceeding)?\s*(-?\d+(?:\.\d+)?)\s*deg\s*([CF])\b", re.I)


def _to_c(v: float, unit: str) -> float:
    return round((v - 32) * 5 / 9, 1) if unit.upper() == "F" else v


def temperatures_c(text: str) -> list[tuple[float, float]]:
    """Temperature ranges in deg C. A single value becomes a degenerate range."""
    t = clean(text)
    t = re.sub(r"(\d)\s*(?:deg\s*)?([CF])\b", r"\1 deg \2", t)
    out, spans = [], []
    for m in TEMP_RANGE.finditer(t):
        u = m.group(4)
        lo, hi = _to_c(float(m.group(1)), m.group(2) or u), _to_c(float(m.group(3)), u)
        out.append((min(lo, hi), max(lo, hi)))
        spans.append(m.span())
    for m in TEMP_ONE.finditer(t):
        if any(a <= m.start(1) < b for a, b in spans):
            continue
        v = _to_c(float(m.group(1)), m.group(2))
        out.append((v, v))
    if re.search(r"room temperature|ambient", t, re.I) and not out:
        out.append((15.0, 25.0))
    if re.search(r"refrigerat", t, re.I) and not out:
        out.append((2.0, 8.0))
    return out


# ---- lab thresholds ---------------------------------------------------------------------
ANALYTES = {
    "anc": r"absolute neutrophil count|\banc\b|neutrophils?",
    "platelets": r"platelets?|\bplt\b",
    "hemoglobin": r"ha?emoglobin|\bhgb\b|\bhb\b",
    "creatinine_clearance": r"creatinine clearance|\bcrcl\b|\begfr\b|glomerular filtration",
    "creatinine": r"(?<!clearance )serum creatinine|\bcreatinine\b(?! clearance)",
    "bilirubin": r"bilirubin",
    "ast_alt": r"\bast\b|\balt\b|transaminase|aminotransferase|sgot|sgpt",
}
LOWER_BOUND = {"anc", "platelets", "hemoglobin", "creatinine_clearance"}   # eligibility needs >= X
COMP = r"(>=|<=|>|<|=|greater than or equal to|less than or equal to|at least|no more than|greater than|less than|above|below|of)"


@dataclass(frozen=True)
class Lab:
    analyte: str
    op: str          # ">=", "<=", ">", "<"
    value: float     # canonical: cells/uL, g/dL, mL/min, or multiple of ULN ("xuln" unit)
    unit: str


def _op(s: str) -> str:
    s = s.lower()
    return {"greater than or equal to": ">=", "at least": ">=", "less than or equal to": "<=",
            "no more than": "<=", "greater than": ">", "above": ">", "less than": "<", "below": "<",
            "of": "=", "=": "="}.get(s, s)


def _canon(analyte: str, value: float, unit_text: str) -> tuple[float, str] | None:
    u = unit_text.lower().replace(" ", "")
    if "uln" in u or "upperlimit" in u or "timesthe" in u:
        return value, "xULN"
    if analyte in ("anc", "platelets"):
        if re.search(r"10\^?9|10e9|x109|g/l", u):
            return value * 1000, "/uL"                  # 10^9/L == 10^3/uL
        if re.search(r"10\^?3|10e3|x103|k/ul", u):
            return value * 1000, "/uL"
        if "/ul" in u or "/mm3" in u or "mm³" in u or "/mm^3" in u or u == "" and value >= 100:
            return value, "/uL"
        return None
    if analyte == "hemoglobin":
        if "g/dl" in u:
            return value, "g/dL"
        if "g/l" in u:
            return value / 10, "g/dL"
        if "mmol/l" in u:
            return value * 1.611, "g/dL"
        return None
    if analyte == "creatinine_clearance":
        return value, "mL/min"
    if analyte in ("creatinine", "bilirubin"):
        if "mg/dl" in u:
            return value, "mg/dL"
        if "umol/l" in u:
            return value / (88.4 if analyte == "creatinine" else 17.1), "mg/dL"
    return None


UNIT_TAIL = r"\s*((?:x|times)\s*(?:the\s*)?(?:institutional\s*)?(?:uln|upper limit of normal)|x\s*10\s*\^?\s*[39]\s*/\s*l|x\s*10\s*\^?\s*[39]\s*/\s*ul|/\s*u?l|/\s*mm\^?3|g/dl|g/l|mg/dl|umol/l|mmol/l|ml/min(?:/1\.73\s*m2)?|k/ul)?"


def labs(text: str) -> list[Lab]:
    t = clean(text).lower()
    t = re.sub(r"10\s*(?:\^|\*\*)?\s*([39])(?=\s*/)", r"10^\1", t)
    t = t.replace("10⁹", "10^9").replace("10³", "10^3")
    out = []
    for analyte, rx in ANALYTES.items():
        for m in re.finditer(rf"(?:{rx})[^.;<>=\d]{{0,40}}?{COMP}\s*{NUM}{UNIT_TAIL}", t):
            op, val, unit = _op(m.group(1)), _num(m.group(2)), m.group(3) or ""
            c = _canon(analyte, val, unit)
            if c:
                out.append(Lab(analyte, op, round(c[0], 3), c[1]))
    return out


# ---- doses -----------------------------------------------------------------------------
DOSE = re.compile(NUM + r"\s*(mg/m2|mg/m\^2|mg/kg|mcg/kg|ug/kg|mg|mcg|ug|g|iu|units?|auc)\b", re.I)
AUC = re.compile(r"\bauc\s*(?:of\s*)?(?:=\s*)?" + NUM, re.I)


def doses(text: str) -> list[tuple[float, str]]:
    t = clean(text).replace("mg/m²", "mg/m2")
    out = []
    for n, unit in DOSE.findall(t):
        u = unit.lower().replace("^", "")
        v = _num(n)
        if u == "g":
            v, u = v * 1000, "mg"
        elif u in ("mcg", "ug"):
            v, u = v / 1000, "mg"
        elif u in ("mcg/kg", "ug/kg"):
            v, u = v / 1000, "mg/kg"
        elif u.startswith("unit"):
            u = "iu"
        if u != "auc":
            out.append((round(v, 4), u))
    out += [(_num(n), "auc") for n in AUC.findall(t)]
    return out


# ---- visits ------------------------------------------------------------------------------
CYCLE_LEN = re.compile(r"(\d+)[- ]day cycles?|cycles? (?:of|are|is|=)\s*(\d+) days|every (\d+) days|q(\d+)d\b|every (\d+) weeks|q(\d)w\b", re.I)


def cycle_length_days(text: str) -> int | None:
    m = CYCLE_LEN.search(clean(text))
    if not m:
        return None
    g = [x for x in m.groups() if x]
    v = int(g[0])
    return v * 7 if re.search(r"weeks|q\dw", m.group(0), re.I) else v


def visit_day(text: str, cycle_len: int | None) -> float | None:
    """Study day of a visit label; None when it cannot be placed on the calendar."""
    t = clean(text)
    m = re.search(r"\b(?:cycle|c)\s*(\d+)\s*,?\s*(?:day|d)\s*(-?\d+)", t, re.I)
    if m and cycle_len:
        return (int(m.group(1)) - 1) * cycle_len + int(m.group(2))
    m = re.search(r"\bmonth\s*(\d+)", t, re.I)
    if m:
        return int(m.group(1)) * 30.44
    m = re.search(r"\bweek\s*(\d+)", t, re.I)
    if m:
        return int(m.group(1)) * 7
    m = re.search(r"\bday\s*(-?\d+)", t, re.I)
    if m:
        return int(m.group(1))
    return None


# ---- other -------------------------------------------------------------------------------
ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4"}


def phases(text: str) -> set[str]:
    t = clean(text).lower()
    out = set()
    for m in re.finditer(r"phase\s*([0-4]|iv|iii|ii|i)\b(?:\s*[/-]\s*(?:phase\s*)?([0-4]|iv|iii|ii|i)\b)?", t):
        for g in m.groups():
            if g:
                out.add(ROMAN.get(g, g))
    return out


NCT = re.compile(r"\bnct\s*-?\s*(\d+)\b", re.I)


def nct_ids(text: str) -> list[str]:
    return ["NCT" + d for d in NCT.findall(clean(text))]


GENERIC_DRUGS = {"placebo", "saline", "standard of care", "best supportive care", "matching placebo", "vehicle"}


def drug_key(name: str) -> str:
    t = clean(name).lower()
    t = re.sub(r"[®™]|\(r\)|\(tm\)", "", t)
    t = re.sub(r"\(.*?\)", "", t)
    t = re.sub(r"\b(injection|tablets?|capsules?|for injection|oral|iv|intravenous|solution|hydrochloride|hcl|sodium|mesylate)\b", "", t)
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def iter_strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, dict):
        for v in obj.values():
            yield from iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_strings(v)
    elif isinstance(obj, str) and obj:
        yield obj
