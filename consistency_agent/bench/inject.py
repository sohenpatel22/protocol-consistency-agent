"""Plant one known contradiction in a copy of a real extraction.

Each operator returns the changed copy plus a label (type, fields, original value).
Hard negatives rewrite a value into an equivalent form and should not be flagged."""
from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..agent import normalize as N
from ..replication.elements import get_path, set_path


@dataclass
class Case:
    case_id: str
    operator: str
    type: Optional[str]                     # None for hard negatives
    fields: list[str]
    doc: dict
    original: dict = field(default_factory=dict)   # field id -> original value (for resolution scoring)
    note: str = ""

    @property
    def hard_negative(self) -> bool:
        return self.type is None


Op = Callable[[dict, random.Random], Optional[tuple[dict, list[str], dict, str]]]
OPERATORS: dict[str, tuple[Optional[str], Op]] = {}


def op(name: str, type_: Optional[str]):
    def deco(fn):
        OPERATORS[name] = (type_, fn)
        return fn
    return deco


def _fields(doc, pattern):
    return N.fields_matching(N.flatten(doc), pattern)


def _set(doc, fid, value):
    d = copy.deepcopy(doc)
    set_path(d, fid, value)
    return d


def _arm_tli_pairs(doc):
    """(arm dosage field, tli dose field) for the same drug with a parseable shared dose."""
    from ..agent.rules import _products, _same_drug
    prods = _products(N.flatten(doc))
    arms = [(n, d, k) for n, d, k in prods if ".arms[" in n.id and d and N.doses(d.text)]
    tli = [(n, d, k) for n, d, k in prods if ".treatment_level_information[" in n.id and d and N.doses(d.text)]
    return [(ad, td) for _, ad, ak in arms for _, td, tk in tli
            if _same_drug(ak, tk) and set(N.doses(ad.text)) & set(N.doses(td.text))]


def _bump_number(text: str, factor: float) -> str:
    m = (re.search(N.NUM + r"(?=\s*(?:mg|mcg|ug|g|iu|units?|mg/m2|mg/kg)\b)", text, re.I)
         or re.search(r"\bAUC\s*(?:of\s*)?(?:=\s*)?" + N.NUM, text, re.I))
    if not m:
        return text
    v = float(m.group(1).replace(",", ""))
    nv = v * factor
    nv_s = f"{nv:.0f}" if nv >= 10 else f"{nv:.2g}"
    return text[:m.start(1)] + nv_s + text[m.end(1):]


# ---- conflict operators -------------------------------------------------------------------
@op("dose_change", "dose_mismatch")
def dose_change(doc, rng):
    pairs = _arm_tli_pairs(doc)
    if not pairs:
        return None
    ad, td = rng.choice(pairs)
    new = _bump_number(ad.text, rng.choice([1.25, 1.5, 0.5]))
    if new == ad.text:
        return None
    return _set(doc, ad.id, new), [ad.id, td.id], {ad.id: ad.text}, f"{ad.text!r} -> {new!r}"


ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else "-" + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " and " + words(n % 100))
    return str(n)


@op("dose_change_paraphrase", "dose_mismatch")
def dose_change_paraphrase(doc, rng):
    """Same contradiction, written in words: rules cannot parse it, the verifier should."""
    pairs = _arm_tli_pairs(doc)
    if not pairs:
        return None
    ad, td = rng.choice(pairs)
    m = re.search(N.NUM + r"\s*(mg/m2|mg/kg|mg)\b", N.clean(ad.text), re.I)
    if not m:
        return None
    v = int(float(m.group(1).replace(",", "")) * 1.5)
    unit = {"mg/m2": "milligrams per square metre", "mg/kg": "milligrams per kilogram"}.get(m.group(2).lower(), "milligrams")
    new = N.clean(ad.text)[:m.start()] + f"{words(v)} {unit}" + N.clean(ad.text)[m.end():]
    return _set(doc, ad.id, new), [ad.id, td.id], {ad.id: ad.text}, f"{ad.text!r} -> {new!r}"


@op("lab_threshold", "lab_threshold_conflict")
def lab_threshold(doc, rng):
    inc = [(f, l) for f in _fields(doc, r"^inclusion_exclusion\.inclusion\[\d+\]$") for l in N.labs(f.text)
           if l.analyte in ("anc", "platelets") and l.op in (">=", ">")]
    disc = "adverse_events.safety_monitoring_and_management.discontinuation_criteria"
    if not inc:
        return None
    f, l = rng.choice(inc)
    stop = int(l.value * 1.3)
    name = "absolute neutrophil count" if l.analyte == "anc" else "platelet count"
    old = get_path(doc, disc) or ""
    new = (old.rstrip() + " " if old else "") + f"Permanently discontinue study treatment if {name} < {stop:,}/uL."
    return _set(doc, disc, new), [f.id, disc], {disc: old}, f"appended stop rule {name} < {stop}"


@op("dose_mod_vs_stop", "dose_modification_vs_ae")
def dose_mod_vs_stop(doc, rng):
    """Add an AE stopping rule at the same grade where interventions only reduce the dose."""
    from ..agent.rules import _grade_rules
    disc = "adverse_events.safety_monitoring_and_management.discontinuation_criteria"
    mods = [(f, r) for f in _fields(doc, r"^interventions\.treatment_level_information\[\d+\]\.modifications$")
            for r in _grade_rules(f.text, r"reduc|hold|interrupt|withh|delay|decrease|dose level")]
    if not mods:
        return None
    f, (tox, grade, _) = rng.choice(mods)
    old = get_path(doc, disc) or ""
    new = (old.rstrip() + " " if old else "") + f"Permanently discontinue study treatment for grade {grade} {tox}."
    return _set(doc, disc, new), [f.id, disc], {disc: old}, f"stop at grade {grade} {tox}"


@op("duration_shorten", "duration_vs_schedule")
def duration_shorten(doc, rng):
    from ..agent.rules import _cycle_len
    fid = "general_information.general_information.estimated_duration"
    old = get_path(doc, fid)
    fields = N.flatten(doc)
    cl = _cycle_len(fields)
    visits = [(v, N.visit_day(v.text, cl)) for v in _fields(doc, r"^schedule_of_events\.schedule_of_events\[\d+\]\.visit_time$")]
    visits = [(v, d) for v, d in visits if d]
    if not old or not visits:
        return None
    last_v, last = max(visits, key=lambda t: t[1])
    if last < 28:
        return None
    weeks = max(1, int(last / 7 * 0.4))
    new = f"{weeks} weeks"
    return _set(doc, fid, new), [fid, last_v.id], {fid: old}, f"{old!r} -> {new!r}"


def _equipment(doc):
    return " | ".join(f.text.lower() for f in _fields(doc, r"^site_requirements\.site_requirements\[\d+\]\.equipment\[\d+\]$"))


@op("storage_freezer", "storage_vs_equipment")
def storage_freezer(doc, rng):
    fid = "site_requirements.ip_storage[0].temperature_requirements"
    eq = _equipment(doc)
    if get_path(doc, fid) is None or not eq or re.search(r"freezer|-20|-70|-80|ultra", eq):
        return None
    new = "Store frozen at -20 °C (-4 °F)"
    return _set(doc, fid, new), [fid], {fid: get_path(doc, fid)}, f"-> {new!r}"


@op("procedure_mri", "procedure_vs_equipment")
def procedure_mri(doc, rng):
    soe = get_path(doc, "schedule_of_events.schedule_of_events") or []
    eq = _equipment(doc)
    if not soe or not eq or re.search(r"\bmri\b|magnetic", eq):
        return None
    i = rng.randrange(len(soe))
    procs = soe[i].get("procedures") or []
    fid = f"schedule_of_events.schedule_of_events[{i}].procedures[{len(procs)}].procedure_name"
    d = copy.deepcopy(doc)
    d["schedule_of_events"]["schedule_of_events"][i].setdefault("procedures", []).append({"procedure_name": "Brain MRI"})
    return d, [fid], {}, f"added 'Brain MRI' to visit {i}"


@op("drug_misspell", "drug_name_mismatch")
def drug_misspell(doc, rng):
    from ..agent.rules import _products, _same_drug
    inter = [k for _, _, k in _products(N.flatten(doc))]
    cands = [f for f in _fields(doc, r"^site_requirements\.ip_storage\[\d+\]\.product_name$")
             if len(N.drug_key(f.text)) >= 6 and any(_same_drug(N.drug_key(f.text), k) for k in inter)]
    if not cands:
        return None
    f = rng.choice(cands)
    t = f.text
    letters = [i for i in range(1, len(t) - 2) if t[i].isalpha() and t[i + 1].isalpha() and t[i] != t[i + 1]]
    if not letters:
        return None
    i = rng.choice(letters)
    new = t[:i] + t[i + 1] + t[i] + t[i + 2:]
    new = new[:-1] + ("a" if new[-1] != "a" else "e")
    return _set(doc, f.id, new), [f.id], {f.id: t}, f"{t!r} -> {new!r}"


@op("nct_truncate", "nct_id_invalid")
def nct_truncate(doc, rng):
    fid = "general_information.general_information.study_nct_id"
    old = get_path(doc, fid)
    if not old or not re.fullmatch(r"NCT\d{8}", old.strip()):
        return None
    return _set(doc, fid, old.strip()[:-1]), [fid], {fid: old}, "dropped last digit"


@op("prohibited_med", "prohibited_med_conflict")
def prohibited_med(doc, rng):
    from ..agent.rules import _products
    fid = "adverse_events.specific_ae_information.concomitant_medication_restrictions"
    prods = [(n, k) for n, _, k in _products(N.flatten(doc)) if len(k) >= 4]
    if not prods:
        return None
    n, k = rng.choice(prods)
    old = get_path(doc, fid) or ""
    new = (old.rstrip() + " " if old else "") + f"Use of {n.text.strip()} is prohibited during the study."
    return _set(doc, fid, new), [fid, n.id], {fid: old}, f"prohibited {n.text!r}"


@op("safety_lab_remove", "safety_lab_not_scheduled")
def safety_lab_remove(doc, rng):
    disc = "adverse_events.safety_monitoring_and_management.discontinuation_criteria"
    text = (get_path(doc, disc) or "").lower()
    soe = get_path(doc, "schedule_of_events.schedule_of_events") or []
    rx = re.compile(r"hematolog|haematolog|\bcbc\b|complete blood|blood count|\bfbc\b", re.I)
    if not soe or not re.search(r"neutrophil|\banc\b|platelet|ha?emoglobin", text):
        return None
    if not any(rx.search(p.get("procedure_name", "")) for v in soe for p in v.get("procedures") or []):
        return None
    d = copy.deepcopy(doc)
    for v in d["schedule_of_events"]["schedule_of_events"]:
        v["procedures"] = [p for p in v.get("procedures") or [] if not rx.search(p.get("procedure_name", ""))]
    return d, [disc], {}, "removed haematology from the schedule"


@op("boilerplate_copy", "boilerplate_duplicate")
def boilerplate_copy(doc, rng):
    a = "adverse_events.ae_definitions_and_classifications.ae_definition"
    b = "adverse_events.ae_definitions_and_classifications.sae_definition"
    va, vb = get_path(doc, a), get_path(doc, b)
    if not va or not vb or len(va) < 60 or N.norm_quote(va) == N.norm_quote(vb):
        return None
    return _set(doc, b, va), [a, b], {b: vb}, "copied AE definition into SAE definition"


@op("phase_change", "phase_mismatch")
def phase_change(doc, rng):
    fid = "general_information.general_information.phase"
    title = get_path(doc, "general_information.general_information.study_title") or ""
    old = get_path(doc, fid)
    tp = N.phases(title)
    if not old or not tp:
        return None
    choices = [p for p in ("1", "2", "3") if p not in tp]
    new = f"Phase {rng.choice(choices)}"
    return _set(doc, fid, new), [fid, "general_information.general_information.study_title"], {fid: old}, f"{old!r} -> {new!r}"


# ---- hard negatives (equivalent rewrites) ----------------------------------------------------
@op("hn_lab_units", None)
def hn_lab_units(doc, rng):
    for f in _fields(doc, r"^inclusion_exclusion\.inclusion\[\d+\]$"):
        t = N.clean(f.text)
        m = re.search(r"(1,?500|100,?000)\s*/\s*(?:u?l|mm\^?3|mm3)", t, re.I)
        if m:
            v = float(m.group(1).replace(",", "")) / 1000
            new = t[:m.start()] + f"{v:g} x 10^9/L" + t[m.end():]
            return _set(doc, f.id, new), [f.id], {}, f"{m.group(0)!r} -> '{v:g} x 10^9/L'"
        m = re.search(r"(1\.5|100)\s*x\s*10\^?9\s*/\s*l", t, re.I)
        if m:
            v = int(float(m.group(1)) * 1000)
            new = t[:m.start()] + f"{v:,}/mm3" + t[m.end():]
            return _set(doc, f.id, new), [f.id], {}, f"{m.group(0)!r} -> '{v:,}/mm3'"
    return None


@op("hn_temp_units", None)
def hn_temp_units(doc, rng):
    for f in _fields(doc, r"^site_requirements\.ip_storage\[\d+\]\.temperature_requirements$"):
        t = N.clean(f.text)
        m = re.search(r"2\s*(?:deg\s*)?C?\s*(?:-|to)\s*8\s*(?:deg\s*)?C", t, re.I)
        if m:
            return _set(doc, f.id, t[:m.start()] + "36-46 °F" + t[m.end():]), [f.id], {}, "2-8 C -> 36-46 F"
    return None


@op("hn_duration_units", None)
def hn_duration_units(doc, rng):
    fid = "general_information.general_information.estimated_duration"
    t = get_path(doc, fid) or ""
    m = re.search(r"(\d+)\s*weeks?", t, re.I)
    if not m:
        return None
    return _set(doc, fid, t[:m.start()] + f"{int(m.group(1)) * 7} days" + t[m.end():]), [fid], {}, "weeks -> days"


@op("hn_phase_roman", None)
def hn_phase_roman(doc, rng):
    fid = "general_information.general_information.phase"
    t = get_path(doc, fid) or ""
    rom = {"1": "I", "2": "II", "3": "III", "4": "IV"}
    new = re.sub(r"(?i)(phase\s*)([1-4])\b", lambda m: m.group(1) + rom[m.group(2)], t)
    if new == t:
        return None
    return _set(doc, fid, new), [fid], {}, f"{t!r} -> {new!r}"


@op("hn_dose_units", None)
def hn_dose_units(doc, rng):
    for ad, _ in _arm_tli_pairs(doc):
        t = N.clean(ad.text)
        m = re.search(r"\b(\d{3,4})\s*mg\b(?!/)", t)
        if m:
            g = int(m.group(1)) / 1000
            return _set(doc, ad.id, t[:m.start()] + f"{g:g} g" + t[m.end():]), [ad.id], {}, "mg -> g"
    return None


@op("hn_drug_case", None)
def hn_drug_case(doc, rng):
    fs = _fields(doc, r"^site_requirements\.ip_storage\[\d+\]\.product_name$")
    if not fs:
        return None
    f = fs[0]
    return _set(doc, f.id, f.text.upper() + "®"), [f.id], {}, "upper case + (R)"


def generate(doc: dict, doc_id: str, seed: int = 0, operators: list[str] | None = None) -> list[Case]:
    cases = []
    for name in operators or OPERATORS:
        type_, fn = OPERATORS[name]
        rng = random.Random(f"{seed}:{doc_id}:{name}")
        res = fn(doc, rng)
        if res is None:
            continue
        new_doc, fields, original, note = res
        cases.append(Case(f"{doc_id}:{name}", name, type_, fields, new_doc, original, note))
    return cases
