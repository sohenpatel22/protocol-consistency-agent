"""Rule-based checks. Precise, but only for wording they can parse.

Every finding quotes the field text it is based on."""
from __future__ import annotations

import re
from typing import Callable

from . import normalize as N
from .linkmap import Evidence, Finding
from .normalize import Field

Rule = Callable[[dict, list[Field], dict], list[Finding]]
RULES: dict[str, Rule] = {}


def rule(name: str):
    def deco(fn):
        RULES[name] = fn
        return fn
    return deco


def _f(type_, pairs: list[tuple[Field, str]], explanation: str, conf: float, cause: str, **extra) -> Finding:
    return Finding(type=type_, fields=sorted({f.id for f, _ in pairs}),
                   evidence=[Evidence(f.id, q) for f, q in pairs], explanation=explanation,
                   source="rules", confidence=conf, likely_cause=cause, extra=extra)


def _get(fields: list[Field], pattern: str) -> list[Field]:
    return N.fields_matching(fields, pattern)


def _parent(fid: str) -> str:
    return fid.rsplit(".", 1)[0]


# ---- interventions ----------------------------------------------------------------------
def _products(fields: list[Field]) -> list[tuple[Field, Field | None, str]]:
    """(name field, dose field, drug key) for arm interventions and treatment-level entries."""
    by_parent: dict[str, dict[str, Field]] = {}
    for f in _get(fields, r"^interventions\."):
        by_parent.setdefault(_parent(f.id), {})[f.id.rsplit(".", 1)[1]] = f
    out = []
    for parent, kids in by_parent.items():
        name = kids.get("name") or kids.get("product_name")
        if not name:
            continue
        key = N.drug_key(name.text)
        if not key or key in N.GENERIC_DRUGS or "placebo" in key:
            continue
        out.append((name, kids.get("dosage") or kids.get("dose"), key))
    return out


def _same_drug(a: str, b: str) -> bool:
    return a == b or a in b.split() or b in a.split() or N.similar(a, b) >= 0.9


@rule("dose_mismatch")
def dose_mismatch(ext, fields, ctx):
    prods = _products(fields)
    arms = [p for p in prods if ".arms[" in p[0].id and p[1]]
    tli = [p for p in prods if ".treatment_level_information[" in p[0].id and p[1]]
    out = []
    for tname, tdose, tkey in tli:
        tvals = set(N.doses(tdose.text))
        if not tvals:
            continue
        # every arm dose of this drug must appear among the treatment-level doses
        for n, d, k in arms:
            av = set(N.doses(d.text))
            if _same_drug(k, tkey) and av and not (tvals & av):
                out.append(_f("dose_mismatch", [(tdose, tdose.text), (d, d.text)],
                              f"{tname.text}: arm dose '{d.text}' does not match the treatment-level "
                              f"dose '{tdose.text}'.", 0.9, "extraction_error"))
    return out


@rule("drug_name_mismatch")
def drug_name_mismatch(ext, fields, ctx):
    inter = [(n, k) for n, _, k in _products(fields)]
    out = []
    for f in _get(fields, r"^site_requirements\.ip_storage\[\d+\]\.product_name$"):
        key = N.drug_key(f.text)
        if not key or "placebo" in key or not inter:
            continue
        if any(_same_drug(key, k) for _, k in inter):
            continue
        best = max(inter, key=lambda nk: N.similar(key, nk[1]))
        sim = N.similar(key, best[1])
        if sim >= 0.6:
            out.append(_f("drug_name_mismatch", [(f, f.text), (best[0], best[0].text)],
                          f"IP storage names '{f.text}' but the interventions name '{best[0].text}' "
                          f"(similarity {sim:.2f}).", 0.6 + 0.3 * sim, "extraction_error"))
    return out


@rule("prohibited_med_conflict")
def prohibited_med_conflict(ext, fields, ctx):
    out = []
    conmed = _get(fields, r"^adverse_events\.specific_ae_information\.concomitant_medication_restrictions$")
    names = {k: n for n, _, k in _products(fields) if len(k) >= 4}
    for f in conmed:
        for sent in re.split(r"(?<=[.;])\s+", N.clean(f.text)):
            low = sent.lower()
            if not re.search(r"prohibit|not (be )?(permitted|allowed)|must not|should not be (used|taken|given)|avoid", low):
                continue
            for key, nf in names.items():
                if re.search(rf"\b{re.escape(key)}\b", low) and not re.search(
                        rf"(other than|except|apart from|besides|in addition to)[^.]*\b{re.escape(key)}\b", low):
                    out.append(_f("prohibited_med_conflict", [(f, sent), (nf, nf.text)],
                                  f"'{nf.text}' is a study intervention but the concomitant-medication "
                                  f"section prohibits it.", 0.85, "protocol_inconsistency"))
    return out


TOXICITIES = ["febrile neutropenia", "neutropenia", "thrombocytopenia", "anemia", "anaemia", "diarrhea",
              "diarrhoea", "nausea", "vomiting", "rash", "neuropathy", "fatigue", "mucositis", "stomatitis",
              "hepatotoxicity", "pneumonitis", "hypertension", "qt prolongation", "colitis"]
GRADE = re.compile(r"grade\s*([1-4])(\s*(?:or (?:higher|greater|above)|\+|and above))?|(?:>=|>|at least)\s*grade\s*([1-4])")


def _grade_rules(text: str, actions: str) -> list[tuple[str, int, str]]:
    """(toxicity, grade, sentence) for sentences that apply `actions` at a toxicity grade."""
    out = []
    for sent in re.split(r"(?<=[.;])\s+", N.clean(text)):
        low = sent.lower()
        if not re.search(actions, low):
            continue
        m = GRADE.search(low)
        if not m:
            continue
        grade = int(m.group(1) or m.group(3))
        for tox in TOXICITIES:
            if tox in low:
                out.append((tox.replace("anaemia", "anemia").replace("diarrhoea", "diarrhea"), grade, sent))
                break
    return out


@rule("dose_modification_vs_ae")
def dose_modification_vs_ae(ext, fields, ctx):
    """Interventions says 'reduce the dose at grade N toxicity'; AE section says 'stop at grade <= N'."""
    mods = [(f, r) for f in _get(fields, r"^interventions\.treatment_level_information\[\d+\]\.modifications$")
            for r in _grade_rules(f.text, r"reduc|hold|interrupt|withh|delay|decrease|dose level")]
    stops = [(f, r) for f in _get(fields, r"^adverse_events\.safety_monitoring_and_management\.discontinuation_criteria$")
             for r in _grade_rules(f.text, r"permanently|discontinu")]
    out = []
    for mf, (tox, g_mod, s_mod) in mods:
        for sf, (tox2, g_stop, s_stop) in stops:
            if tox == tox2 and g_stop <= g_mod:
                out.append(_f("dose_modification_vs_ae", [(mf, s_mod), (sf, s_stop)],
                              f"Interventions reduce or hold the dose at grade {g_mod} {tox}, but the adverse-event "
                              f"section stops treatment at grade {g_stop}.", 0.8, "protocol_inconsistency"))
    return out


# ---- eligibility vs stopping rules ------------------------------------------------------
@rule("lab_threshold_conflict")
def lab_threshold_conflict(ext, fields, ctx):
    elig = [(f, lab) for f in _get(fields, r"^inclusion_exclusion\.inclusion\[\d+\]$") for lab in N.labs(f.text)]
    stop = [(f, lab) for f in _get(fields, r"^adverse_events\.safety_monitoring_and_management\.discontinuation_criteria$")
            for lab in N.labs(f.text)]
    out = []
    for ef, el in elig:
        for sf, sl in stop:
            if el.analyte != sl.analyte or el.unit != sl.unit:
                continue
            lower = el.analyte in N.LOWER_BOUND
            if lower and el.op in (">=", ">") and sl.op in ("<", "<=") and sl.value > el.value:
                bad = True
            elif not lower and el.op in ("<=", "<") and sl.op in (">", ">=") and sl.value < el.value:
                bad = True
            else:
                bad = False
            if bad:
                out.append(_f("lab_threshold_conflict", [(ef, ef.text), (sf, sf.text)],
                              f"Eligibility allows {el.analyte} {el.op} {el.value:g} {el.unit}, but treatment "
                              f"stops at {sl.op} {sl.value:g} {sl.unit}: an eligible participant could meet the "
                              f"stopping rule on day 1.", 0.85, "protocol_inconsistency"))
    return out


# ---- duration vs schedule ---------------------------------------------------------------
def _cycle_len(fields):
    for f in _get(fields, r"^interventions\."):
        c = N.cycle_length_days(f.text)
        if c:
            return c
    return None


@rule("duration_vs_schedule")
def duration_vs_schedule(ext, fields, ctx):
    dur = _get(fields, r"^general_information\.general_information\.estimated_duration$")
    visits = _get(fields, r"^schedule_of_events\.schedule_of_events\[\d+\]\.visit_time$")
    if not dur or not visits:
        return []
    d = N.durations_days(dur[0].text)
    if not d or re.search(r"until (disease )?progression|until death|indefinite", dur[0].text, re.I):
        return []
    total = max(d)
    cl = _cycle_len(fields)
    days = [(v, N.visit_day(v.text, cl)) for v in visits]
    days = [(v, x) for v, x in days if x is not None]
    if not days:
        return []
    last_v, last = max(days, key=lambda t: t[1])
    if last > total * 1.10 + 7:
        return [_f("duration_vs_schedule", [(dur[0], dur[0].text), (last_v, last_v.text)],
                   f"Schedule runs to about day {last:.0f} but the stated duration is about {total:.0f} days.",
                   0.8, "extraction_error")]
    return []


# ---- storage / procedures vs equipment ---------------------------------------------------
EQUIP = {
    "refrigerator": r"refrigerat|fridge|2\s*(?:deg\s*c?)?\s*(?:-|to)\s*8",
    "freezer_-20": r"freezer|-20|-25|-70|-80|ultra[- ]?low",
    "freezer_-80": r"-70|-80|-65|ultra[- ]?low|deep freezer",
}


def _equipment_text(fields):
    eq = _get(fields, r"^site_requirements\.site_requirements\[\d+\]\.equipment\[\d+\]$")
    return eq, " | ".join(N.clean(f.text).lower() for f in eq)


@rule("storage_vs_equipment")
def storage_vs_equipment(ext, fields, ctx):
    eq, eq_text = _equipment_text(fields)
    if not eq:
        return []
    out = []
    for f in _get(fields, r"^site_requirements\.(ip_storage\[\d+\]\.temperature_requirements|site_requirements\[\d+\]\.sample_handling\.storage_conditions)$"):
        for lo, hi in N.temperatures_c(f.text):
            need = ("freezer_-80" if hi <= -60 else "freezer_-20" if hi <= -15
                    else "refrigerator" if 0 <= lo and hi <= 10 else None)
            if need and not re.search(EQUIP[need], eq_text):
                out.append(_f("storage_vs_equipment", [(f, f.text), (eq[0], eq[0].text)],
                              f"Storage at {lo:g} to {hi:g} C needs a {need.replace('_', ' ')} C unit, which is "
                              f"not in the site equipment list.", 0.75, "missing_condition", need=need))
                break
    return out


PROC_EQUIP = {
    "ECG machine": (r"\becg\b|\bekg\b|electrocardiogra", r"\becg\b|\bekg\b|electrocardiogra"),
    "MRI": (r"\bmri\b|magnetic resonance", r"\bmri\b|magnetic resonance"),
    "CT scanner": (r"\bct\b|computed tomography|ct scan", r"\bct\b|computed tomography|imaging"),
    "Echocardiography": (r"echocardiogra|\becho\b|muga", r"echocardiogra|\becho\b|muga|ultrasound"),
    "Centrifuge": (r"\bpk\b|pharmacokinetic|plasma sample|serum sample", r"centrifug"),
}


@rule("procedure_vs_equipment")
def procedure_vs_equipment(ext, fields, ctx):
    eq, eq_text = _equipment_text(fields)
    if not eq:
        return []
    procs = _get(fields, r"^schedule_of_events\.schedule_of_events\[\d+\]\.procedures\[\d+\]\.procedure_name$")
    out, seen = [], set()
    for p in procs:
        low = N.clean(p.text).lower()
        for name, (proc_rx, eq_rx) in PROC_EQUIP.items():
            if name in seen or not re.search(proc_rx, low):
                continue
            if not re.search(eq_rx, eq_text):
                seen.add(name)
                out.append(_f("procedure_vs_equipment", [(p, p.text), (eq[0], eq[0].text)],
                              f"Schedule includes '{p.text}' but the equipment list has no {name}.", 0.6,
                              "missing_condition", equipment=name))
    return out


SAFETY_TESTS = {
    "haematology": (r"neutrophil|\banc\b|platelet|ha?emoglobin|white blood", r"hematolog|haematolog|\bcbc\b|complete blood|blood count|\bfbc\b"),
    "chemistry": (r"\bast\b|\balt\b|bilirubin|creatinine|transaminase|liver function|hepatic", r"chemistr|\bcmp\b|\blft|liver function|metabolic panel|biochem|hepatic"),
    "ECG": (r"\bqtc?\b|\becg\b|qt prolong", r"\becg\b|\bekg\b|electrocardiogra"),
}


@rule("safety_lab_not_scheduled")
def safety_lab_not_scheduled(ext, fields, ctx):
    procs = _get(fields, r"^schedule_of_events\.schedule_of_events\[\d+\]\.procedures\[\d+\]\.procedure_name$")
    if not procs:
        return []
    ptext = " | ".join(N.clean(p.text).lower() for p in procs)
    out = []
    for f in _get(fields, r"^adverse_events\.safety_monitoring_and_management\.discontinuation_criteria$"):
        low = N.clean(f.text).lower()
        for test, (trigger, sched) in SAFETY_TESTS.items():
            m = re.search(trigger, low)
            if m and not re.search(sched, ptext):
                out.append(_f("safety_lab_not_scheduled", [(f, f.text)],
                              f"Stopping rules use {test} results ('{m.group(0)}') but no {test} test appears in "
                              f"the schedule of events.", 0.7, "missing_condition", test=test))
    return out


# ---- general information ---------------------------------------------------------------
@rule("nct_id_invalid")
def nct_id_invalid(ext, fields, ctx):
    out = []
    reg = ctx.get("registry_nct_id")
    for f in _get(fields, r"^general_information\.general_information\.study_nct_id$"):
        ids = N.nct_ids(f.text)
        val = ids[0] if ids else N.clean(f.text)
        if not re.fullmatch(r"NCT\d{8}", val):
            out.append(_f("nct_id_invalid", [(f, f.text)], f"'{f.text}' is not a valid NCT ID (NCT + 8 digits).",
                          0.95, "extraction_error"))
        elif reg and val != reg:
            out.append(_f("nct_id_invalid", [(f, f.text)], f"'{val}' differs from the registry record {reg}.",
                          0.9, "extraction_error"))
    return out


@rule("phase_mismatch")
def phase_mismatch(ext, fields, ctx):
    ph = _get(fields, r"^general_information\.general_information\.phase$")
    ti = _get(fields, r"^general_information\.general_information\.study_title$")
    if not ph or not ti:
        return []
    a, b = N.phases(ph[0].text) or N.phases("phase " + ph[0].text), N.phases(ti[0].text)
    if a and b and a != b:
        return [_f("phase_mismatch", [(ph[0], ph[0].text), (ti[0], ti[0].text)],
                   f"Phase field says {'/'.join(sorted(a))} but the title says {'/'.join(sorted(b))}.", 0.85,
                   "extraction_error")]
    return []


@rule("boilerplate_duplicate")
def boilerplate_duplicate(ext, fields, ctx):
    ae = [f for f in _get(fields, r"^adverse_events\.[a-z_]+\.[a-z_]+$") if len(f.text) >= 60]
    out = []
    for i, a in enumerate(ae):
        for b in ae[i + 1:]:
            na, nb = N.norm_quote(a.text), N.norm_quote(b.text)
            if na == nb or (min(len(na), len(nb)) > 80 and N.similar(na, nb) > 0.95):
                out.append(_f("boilerplate_duplicate", [(a, a.text[:200]), (b, b.text[:200])],
                              f"'{a.id.rsplit('.', 1)[1]}' and '{b.id.rsplit('.', 1)[1]}' contain the same text.",
                              0.8, "extraction_error"))
    return out


def run_rules(extraction: dict, ctx: dict | None = None) -> list[Finding]:
    fields = N.flatten(extraction)
    out = []
    for fn in RULES.values():
        out.extend(fn(extraction, fields, ctx or {}))
    return out
