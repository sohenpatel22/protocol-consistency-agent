"""Demo app for the study data consistency agent.

    streamlit run app/demo.py

Works offline on a synthetic protocol (rule checks, no API key, no cost). With keys in .env,
switch the checker to "Rules + LLM" and pick a real extraction from data/extractions/.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from consistency_agent.agent import audit  # noqa: E402
from consistency_agent.agent.pipeline import check  # noqa: E402
from consistency_agent.agent.resolve import recheck  # noqa: E402
from consistency_agent.agent.rules import run_rules  # noqa: E402
from consistency_agent.bench import metrics  # noqa: E402
from consistency_agent.bench.inject import OPERATORS, generate  # noqa: E402
from consistency_agent.bench.run import evaluate_case  # noqa: E402
from consistency_agent.config import EXTRACTIONS  # noqa: E402
from consistency_agent.replication.elements import get_path  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_extraction.json"
SECTIONS = {"general_information": "General info", "inclusion_exclusion": "Inclusion / exclusion",
            "adverse_events": "Adverse events", "interventions": "Interventions",
            "site_requirements": "Site requirements", "schedule_of_events": "Schedule of events"}
SEV_COLOR = {"critical": "#D9711F", "major": "#016983", "minor": "#5B6770"}
OP_NAMES = {
    "dose_change": "Change an arm dose (200 → 250 mg)",
    "dose_change_paraphrase": "Change a dose, written in words (needs the LLM)",
    "dose_mod_vs_stop": "Stop treatment at the grade where the dose is only reduced",
    "lab_threshold": "Stopping lab limit stricter than the eligibility limit",
    "duration_shorten": "Shorten the stated study duration",
    "storage_freezer": "Drug needs a freezer the site doesn't list",
    "procedure_mri": "Add an MRI the site has no equipment for",
    "drug_misspell": "Misspell the drug name in IP storage",
    "nct_truncate": "Cut the last digit off the NCT ID",
    "prohibited_med": "Prohibit one of the study drugs",
    "safety_lab_remove": "Remove the blood counts the stopping rules depend on",
    "boilerplate_copy": "Copy the AE definition into the SAE definition",
    "phase_change": "Change the phase so it disagrees with the title",
    "hn_lab_units": "Look-alike: 1,500/µL rewritten as 1.5 × 10⁹/L",
    "hn_temp_units": "Look-alike: 2–8 °C rewritten as 36–46 °F",
    "hn_duration_units": "Look-alike: weeks rewritten as days",
    "hn_phase_roman": "Look-alike: Phase 2 rewritten as Phase II",
    "hn_dose_units": "Look-alike: 200 mg rewritten as 0.2 g",
    "hn_drug_case": "Look-alike: drug name in capitals with ®",
}

st.set_page_config(page_title="Study Data Consistency Agent", page_icon="✅", layout="wide")


# ---------- data ----------
def sources() -> dict[str, Path]:
    out = {"Synthetic protocol (offline demo)": FIXTURE}
    if EXTRACTIONS.exists():
        for p in sorted(EXTRACTIONS.glob("*/*.json")):
            out[f"{p.stem} ({p.parent.name})"] = p
    return out


@st.cache_data
def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_check(extraction: dict, mode: str, nct: str, provenance: dict) -> tuple[list, dict]:
    ctx = {"registry_nct_id": nct, "nct_id": nct, "provenance": provenance}
    return check(extraction, mode, ctx, log=lambda *_: None)


def badge(sev: str) -> str:
    return (f"<span style='background:{SEV_COLOR.get(sev, '#5B6770')};color:white;padding:2px 8px;"
            f"border-radius:10px;font-size:0.8em;font-weight:600'>{sev.upper()}</span>")


def show_finding(f, extraction: dict):
    with st.container(border=True):
        st.markdown(f"{badge(f.severity)} &nbsp; **{f.type.replace('_', ' ')}** &nbsp; "
                    f"<span style='color:#5B6770'>source: {f.source} · confidence {f.confidence:.2f}"
                    f"{' · likely cause: ' + f.likely_cause.replace('_', ' ') if f.likely_cause else ''}</span>",
                    unsafe_allow_html=True)
        st.write(f.explanation)
        for e in f.evidence:
            src = f.extra.get("sources", {}).get(e.field, {})
            where = f" · protocol p. {', '.join(map(str, src['pages']))}" if src.get("pages") else ""
            section = SECTIONS.get(e.field.split(".")[0], e.field.split(".")[0])
            st.markdown(f"- **{section}** `{e.field.split('.', 1)[1]}`{where}  \n  > {e.quote[:300]}")


# ---------- sidebar ----------
st.sidebar.title("Consistency agent")
src_name = st.sidebar.selectbox("Protocol extraction", list(sources()))
mode_label = st.sidebar.radio("Checker", ["Rules only (free, offline)", "Rules + LLM (needs API key)"])
mode = "rules" if mode_label.startswith("Rules only") else "hybrid"
st.sidebar.caption("Rules only runs instantly with no API calls. Rules + LLM adds the LLM verifier and the "
                   "open sweep, whose suspects are confirmed in the protocol before they count.")
data = load(str(sources()[src_name]))
base = data["extraction"]
nct = data.get("nct_id", "demo")
provenance = data.get("provenance", {})

st.title("Study Data Consistency Agent")
st.caption("Checks that the six sections extracted from a clinical trial protocol agree with each other, "
           "explains each conflict with evidence, and proposes a fix that a person approves.")

tabs = st.tabs(["1 · Extraction", "2 · Check", "3 · Plant a contradiction", "4 · Fix and review", "5 · Benchmark"])

# ---------- 1. extraction ----------
with tabs[0]:
    st.subheader(f"Six-section extraction · {nct}")
    if src_name.startswith("Synthetic"):
        st.info("A fictional protocol written for the demo. With API keys, real ClinicalTrials.gov protocols "
                "extracted with Banting's method appear in the sidebar.")
    cols = st.columns(3)
    for i, (key, label) in enumerate(SECTIONS.items()):
        with cols[i % 3].expander(label, expanded=False):
            st.json(base.get(key, {}), expanded=2)

# ---------- 2. check ----------
with tabs[1]:
    st.subheader("Consistency check")
    if st.button("Run the check", type="primary", key="run_check"):
        st.session_state["base_findings"] = run_check(base, mode, nct, provenance)
    if "base_findings" in st.session_state:
        findings, stats = st.session_state["base_findings"]
        c = st.columns(4)
        c[0].metric("Findings", len(findings))
        for i, sev in enumerate(("critical", "major", "minor")):
            c[i + 1].metric(sev.capitalize(), sum(f.severity == sev for f in findings))
        if not findings:
            st.success("No contradictions between sections. Try tab 3 to plant one.")
        for f in findings:
            show_finding(f, base)
        if stats.get("n_suspects_dropped"):
            st.caption(f"{stats['n_suspects_dropped']} open-sweep suspect(s) were not confirmed in the protocol "
                       "and are not reported.")

# ---------- 3. plant ----------
with tabs[2]:
    st.subheader("Plant a known contradiction and watch it get caught")
    st.write("Each option changes one value in a copy of the extraction. Look-alikes rewrite a value into an "
             "equivalent form: a good checker must **not** flag them.")
    cases = {c.operator: c for c in generate(base, nct)}
    available = [op for op in OPERATORS if op in cases]
    op = st.selectbox("What to plant", available, format_func=lambda o: OP_NAMES.get(o, o))
    case = cases[op]
    st.markdown(f"**Change made:** {case.note}")
    if st.button("Plant it and run the check", type="primary", key="plant"):
        base_f, _ = run_check(base, mode, nct, provenance)
        row = evaluate_case(case, {f.key() for f in base_f}, mode, {"registry_nct_id": nct, "nct_id": nct})
        after, _ = run_check(case.doc, mode, nct, provenance)
        new = [f for f in after if f.key() not in {b.key() for b in base_f}]
        st.session_state["planted"] = {"case": case, "findings": new}
        if case.hard_negative:
            (st.success if not new else st.error)(
                "No alert: the checker saw these mean the same thing." if not new else
                f"False alarm: {len(new)} finding(s) on an equivalent rewrite.")
        else:
            (st.success if row["detected"] else st.warning)(
                "Caught it." if row["detected"] else
                "Not caught in this mode. Wording like this needs the LLM verifier (Rules + LLM).")
        for f in new:
            show_finding(f, case.doc)

# ---------- 4. fix and review ----------
with tabs[3]:
    st.subheader("Fix it, re-check it, and record the decision")
    planted = st.session_state.get("planted")
    if not planted or not planted["findings"]:
        st.info("Plant a contradiction in tab 3 first (one that gets caught).")
    else:
        case, findings = planted["case"], planted["findings"]
        f = findings[0]
        show_finding(f, case.doc)
        field = next(iter(case.original), f.fields[0])
        st.markdown(f"**Field to fix:** `{field}`  \n**Current value:** {get_path(case.doc, field)!r}")
        st.caption("With an API key, the fix-up agent re-reads the protocol pages (without seeing this value) "
                   "and proposes the correction. Offline, the original value stands in for its proposal.")
        proposed = st.text_input("Proposed value", value=str(case.original.get(field, "")))
        if st.button("Re-check with this value", key="recheck"):
            prop = {"cause": "extraction_error", "field": field, "proposed_value": proposed, "quote_verified": True}
            chk = recheck(case.doc, f, prop, {"registry_nct_id": nct})
            st.session_state["recheck_result"] = (prop, chk)
        if "recheck_result" in st.session_state:
            prop, chk = st.session_state["recheck_result"]
            if chk.get("applied") and not chk.get("original_conflict_remains") and not chk.get("new_conflicts"):
                st.success("Conflict gone, and no new conflict in the linked fields. Ready for a reviewer.")
            else:
                st.error(f"Not clean: conflict remains = {chk.get('original_conflict_remains')}, "
                         f"new conflicts = {chk.get('new_conflicts')}. This would be escalated to a person.")
            a, b = st.columns(2)
            note = st.text_input("Reviewer note", value="Checked against the protocol")
            if a.button("Approve", type="primary"):
                audit.record(nct, "review", actor="demo-reviewer", finding_type=f.type, field=prop["field"],
                             old=get_path(case.doc, prop["field"]), new=prop["proposed_value"], decision="approve",
                             note=note)
                st.success("Approved and logged.")
            if b.button("Reject"):
                audit.record(nct, "review", actor="demo-reviewer", finding_type=f.type, field=prop["field"],
                             decision="reject", note=note)
                st.warning("Rejected and logged.")
            log = [r for r in audit.read(nct) if r.get("event") == "review"]
            if log:
                st.markdown("**Audit trail**")
                st.dataframe(log[-10:], use_container_width=True)

# ---------- 5. benchmark ----------
with tabs[4]:
    st.subheader("Benchmark on planted contradictions")
    st.write("Plants every kind of contradiction (and every look-alike) in the selected extraction and scores "
             "the checker. Rules only is free; Rules + LLM uses the API.")
    if st.button("Run the benchmark", type="primary", key="bench"):
        base_f, _ = run_check(base, mode, nct, provenance)
        keys = {f.key() for f in base_f}
        rows = [evaluate_case(c, keys, mode, {"registry_nct_id": nct, "nct_id": nct}) for c in generate(base, nct)]
        s = metrics.summarise(rows)
        c = st.columns(4)
        c[0].metric("Recall", f"{s['overall']['recall']:.0%}" if s["overall"]["recall"] is not None else "-")
        c[1].metric("Precision", f"{s['overall']['precision']:.0%}" if s["overall"]["precision"] is not None else "-")
        c[2].metric("Critical recall", f"{s['critical_recall']:.0%}" if s["critical_recall"] is not None else "-")
        c[3].metric("Look-alikes wrongly flagged", f"{s['hard_negative_fpr']:.0%}" if s["hard_negative_fpr"] is not None else "-")
        st.dataframe([{"Planted": OP_NAMES.get(r["operator"], r["operator"]),
                       "Kind": "look-alike" if r["hard_negative"] else r["type"].replace("_", " "),
                       "Result": ("no alert ✓" if r["n_false_pos"] == 0 else "false alarm ✗") if r["hard_negative"]
                       else ("caught ✓" if r["detected"] else "missed ✗"),
                       "False alarms": r["n_false_pos"]} for r in rows], use_container_width=True)
        st.caption("The synthetic protocol was written alongside the rules, so this is a demonstration, not a result. "
                   "Real numbers come from the 23 public protocols.")
