"""Offline tests: no network, no API keys. LLM calls are replaced by fakes."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pymupdf as fitz
import pytest

from consistency_agent import llm
from consistency_agent.agent import normalize as N
from consistency_agent.agent.linkmap import LINK_BY_TYPE
from consistency_agent.agent.pipeline import check
from consistency_agent.agent.resolve import decide, recheck
from consistency_agent.agent.rules import run_rules
from consistency_agent.agent.verifier import verify_link
from consistency_agent.bench.inject import generate
from consistency_agent.bench.metrics import ece, summarise
from consistency_agent.replication import elements as E
from consistency_agent.replication.judge import category_scores
from consistency_agent.replication.pdfparse import parse_pdf
from consistency_agent.replication.retrieval import Retriever

FIX = Path(__file__).parent / "fixtures" / "synthetic_extraction.json"


@pytest.fixture
def doc():
    return json.loads(FIX.read_text(encoding="utf-8"))["extraction"]


@pytest.fixture(autouse=True)
def no_llm():
    def fail(call):
        raise AssertionError(f"unexpected LLM call for role {call.role}")
    llm.set_override(fail)
    yield
    llm.set_override(None)


# ---- normalisation -------------------------------------------------------------------
@pytest.mark.parametrize("a,b", [
    ("ANC >= 1,500/uL", "ANC >= 1.5 x 10^9/L"),
    ("platelets >= 100,000/mm3", "platelet count >= 100 x 10^9/L"),
    ("Hemoglobin >= 9 g/dL", "hemoglobin >= 90 g/L"),
])
def test_lab_units_equivalent(a, b):
    assert [(l.analyte, l.value, l.unit) for l in N.labs(a)] == [(l.analyte, l.value, l.unit) for l in N.labs(b)]


def test_other_parsers():
    assert N.durations_days("14 weeks") == [98]
    assert N.temperatures_c("2-8 °C") == [(2.0, 8.0)]
    lo, hi = N.temperatures_c("36-46 °F")[0]
    assert abs(lo - 2.2) < 0.1 and abs(hi - 7.8) < 0.1
    assert N.visit_day("Cycle 2 Day 1", 21) == 22
    assert N.phases("Phase II") == N.phases("phase 2") == {"2"}
    assert N.doses("0.2 g") == N.doses("200 mg")
    assert N.nct_ids("nct 02187822") == ["NCT02187822"]


def test_paths_roundtrip():
    d = E.empty_category("site_requirements")
    E.set_path(d, "site_requirements[0].equipment", ["ECG"])
    assert E.get_path(d, "site_requirements[0].equipment") == ["ECG"]
    assert all(e.weight >= 0 for e in E.all_elements())
    for code, cat in E.CATEGORY_CODES.items():
        w = sum(e.weight for e in E.all_elements() if e.category == cat)
        assert abs(w - 1.0) < 1e-6, (code, w)       # Table 7 weights sum to 1 per category


# ---- rules + benchmark ------------------------------------------------------------------
def test_fixture_is_consistent(doc):
    assert run_rules(doc, {"registry_nct_id": "NCT09999999"}) == []


def test_every_conflict_operator_is_caught_by_rules_except_paraphrase(doc):
    for case in generate(doc, "NCT09999999"):
        found = run_rules(case.doc, {"registry_nct_id": "NCT09999999"})
        types = {f.type for f in found}
        if case.hard_negative:
            assert not found, (case.operator, [f.explanation for f in found])
        elif case.operator == "dose_change_paraphrase":
            assert case.type not in types
        else:
            assert case.type in types, case.operator


def test_findings_quote_their_fields(doc):
    case = next(c for c in generate(doc, "X") if c.operator == "lab_threshold")
    f = next(f for f in run_rules(case.doc) if f.type == "lab_threshold_conflict")
    flat = {x.id: x.text for x in N.flatten(case.doc)}
    for e in f.evidence:
        assert N.norm_quote(e.quote) in N.norm_quote(flat[e.field])


def test_metrics():
    rows = [{"doc_id": "d", "type": "dose_mismatch", "hard_negative": False, "detected": True, "n_false_pos": 0,
             "severity": "critical", "confidences": [(0.9, True)]},
            {"doc_id": "d", "type": None, "hard_negative": True, "detected": False, "n_false_pos": 1,
             "severity": "", "confidences": [(0.6, False)]}]
    s = summarise(rows, n_boot=50)
    assert s["overall"]["recall"] == 1.0 and s["overall"]["precision"] == 0.5
    assert s["hard_negative_fpr"] == 1.0 and s["critical_recall"] == 1.0
    assert ece([(1.0, True), (0.0, False)]) == 0.0


# ---- verifier guardrail --------------------------------------------------------------------
def test_verifier_drops_unquoted_conflicts(doc):
    case = next(c for c in generate(doc, "X") if c.operator == "dose_change_paraphrase")
    arm = case.fields[0]
    flat = {x.id: x.text for x in N.flatten(case.doc)}

    def fake(call):
        return {"conflicts": [
            {"fields": [arm], "quotes": [{"field": arm, "quote": "three hundred milligrams"}],
             "explanation": "300 mg vs 200 mg", "likely_cause": "extraction_error", "confidence": 0.8},
            {"fields": [arm], "quotes": [{"field": arm, "quote": "450 mg twice daily"}],   # not in the field
             "explanation": "hallucinated", "confidence": 0.9}]}
    llm.set_override(fake)
    found, dropped = verify_link(LINK_BY_TYPE["dose_mismatch"], N.flatten(case.doc))
    assert len(found) == 1 and dropped == 1
    assert "three hundred" in flat[arm]


def test_open_sweep_needs_two_sections(doc):
    fields = N.flatten(doc)
    title = "general_information.general_information.study_title"
    disc = "adverse_events.safety_monitoring_and_management.discontinuation_criteria"
    flat = {f.id: f.text for f in fields}

    def fake(call):
        return {"conflicts": [
            {"fields": [title, disc], "quotes": [{"field": title, "quote": "Double-Blind"},
                                                 {"field": disc, "quote": "QTc > 500 ms"}],
             "explanation": "cross-section", "confidence": 0.6},
            {"fields": [title], "quotes": [{"field": title, "quote": "Phase 2"}],      # one section only
             "explanation": "same section", "confidence": 0.9}]}
    llm.set_override(fake)
    found, dropped = verify_link(LINK_BY_TYPE["other_conflict"], fields)
    assert len(found) == 1 and dropped == 1 and "Double-Blind" in flat[title]


def test_hybrid_check_merges_sources(doc):
    case = next(c for c in generate(doc, "X") if c.operator == "dose_change")
    llm.set_override(lambda call: {"conflicts": []})
    findings, stats = check(case.doc, "hybrid")
    assert any(f.type == "dose_mismatch" for f in findings)
    assert stats["calls"] == len(LINK_BY_TYPE)
    assert all(f.id and f.severity for f in findings)


# ---- resolution re-check -------------------------------------------------------------------
def test_recheck_accepts_restoring_fix_and_rejects_bad_fix(doc):
    case = next(c for c in generate(doc, "X") if c.operator == "dose_change")
    f = next(f for f in run_rules(case.doc) if f.type == "dose_mismatch")
    arm = case.fields[0]
    good = {"cause": "extraction_error", "field": arm, "proposed_value": case.original[arm], "quote_verified": True}
    bad = {"cause": "extraction_error", "field": arm, "proposed_value": "999 mg daily", "quote_verified": True}
    assert decide(good, recheck(case.doc, f, good)) == "suggest_fix"
    assert decide(bad, recheck(case.doc, f, bad)) == "escalate"
    assert decide({"cause": "protocol_inconsistency", "quote_verified": True}, {}) == "sponsor_query"
    assert decide({"cause": "extraction_error", "quote_verified": False}, {}) == "escalate"


def test_resolver_reads_blind_then_decides(doc, tmp_path):
    from consistency_agent.agent import resolve
    from consistency_agent.replication.pdfparse import Chunk
    case = next(c for c in generate(doc, "X") if c.operator == "dose_change")
    f = next(f for f in run_rules(case.doc) if f.type == "dose_mismatch")
    arm = case.fields[0]
    chunk = Chunk("c0", 0, "s0", 0, 1, ["6 Study Treatment"], 31, 31,
                  "Exampletinib 200 mg orally once daily on Days 1-21 of each 21-day cycle.")
    seen = []

    def fake(call):
        seen.append(call.user)
        if "Data field:" in call.user:
            return {"answers": [{"value": "200 mg once daily", "quote": "Exampletinib 200 mg orally once daily",
                                 "page": 31}]}
        return {"cause": "extraction_error", "field": arm, "proposed_value": case.original[arm],
                "protocol_quote": "Exampletinib 200 mg orally once daily"}
    llm.set_override(fake)
    prop = resolve.propose("X", case.doc, f, retriever=Retriever("X", [chunk], backend="bm25"))
    assert "250 mg" not in seen[0]                 # the blind read never sees the extracted value
    assert prop["quote_verified"] and prop["requery_pages"] == [31]
    assert resolve.decide(prop, resolve.recheck(case.doc, f, prop)) == "suggest_fix"


# ---- replication pieces ---------------------------------------------------------------------
def _toy_pdf(path: Path):
    d = fitz.open()
    pg = d.new_page()
    y = 72
    for text, size in [("1 INTRODUCTION", 14), ("Background text about the study drug.", 10),
                       ("5 ELIGIBILITY CRITERIA", 14), ("5.1 Inclusion Criteria", 12),
                       ("1. Absolute neutrophil count >= 1,500/uL.", 10), ("2. Age 18 years or older.", 10),
                       ("8 STUDY DRUG STORAGE", 14), ("Store exampletinib at 2-8 C, protected from light.", 10)]:
        pg.insert_text((72, y), text, fontsize=size, fontname="helv" if size == 10 else "hebo")
        y += 24
    d.save(path)


def test_parse_and_retrieve(tmp_path):
    pdf = tmp_path / "toy.pdf"
    _toy_pdf(pdf)
    chunks = parse_pdf(pdf)
    assert any("Inclusion" in " ".join(c.path) for c in chunks)
    r = Retriever("toy", chunks, backend="bm25")
    top = r.retrieve(["study drug storage temperature"], top_k=1)
    assert "2-8" in top[0].text
    assert [c.order for c in r.retrieve(["inclusion", "storage"], top_k=3)] == sorted(
        c.order for c in r.retrieve(["inclusion", "storage"], top_k=3))


def test_category_scores():
    scores = {e.key: {"score": 5} for e in E.all_elements()}
    cats = category_scores(scores)
    assert cats["overall"] == 100.0
    scores["inclusion"]["score"] = 0
    assert category_scores(scores)["ie"] == 50.0
