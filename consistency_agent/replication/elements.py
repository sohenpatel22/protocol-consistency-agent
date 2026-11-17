"""The data elements we extract, one RAG call each.

Field names and nesting come from the paper's example outputs (Listings 2-7). Those are
examples, not formal schemas, so types are inferred from the examples. Weights are Table 7.
Retrieval queries and guides are ours; the paper doesn't publish its prompts.
Two fields not in the paper, target_enrollment and countries, are added because the
consistency checks use them."""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

SCHEMAS: dict[str, dict] = {
    "general_information": {
        "general_information": {"study_nct_id": "", "protocol_id": "", "protocol_version": "",
                                "protocol_date": "", "study_title": "", "sponsor_name": "", "phase": "",
                                "therapeutic_area": "", "disease_or_condition": "", "allocation": "",
                                "masking": "", "target_enrollment": "", "number_of_sites": 0,
                                "countries": [], "estimated_duration": ""},
        "objectives_and_endpoints": {"primary_objectives": [], "secondary_objectives": [],
                                     "primary_endpoints": [], "secondary_endpoints": [],
                                     "measurement_method": [], "endpoint_timepoint": ""},
    },
    "inclusion_exclusion": {"inclusion": [], "exclusion": []},
    "adverse_events": {
        "ae_definitions_and_classifications": {"ae_definition": "", "sae_definition": "",
                                               "severity_grading": "", "ae_relationship": ""},
        "reporting_requirements": {"reporting_timeframes": "", "ae_data_collection_requirements": "",
                                   "reporting_contacts": ""},
        "safety_monitoring_and_management": {"safety_monitoring_plan": "", "discontinuation_criteria": "",
                                             "emergency_procedures": ""},
        "specific_ae_information": {"expected_aes": "", "potential_risks": "",
                                    "concomitant_medication_restrictions": "",
                                    "special_population_considerations": ""},
    },
    "interventions": {
        "arms": [{"arm_name": "", "interventions": [{"type": "", "name": "", "dosage": "", "schedule": ""}]}],
        "treatment_level_information": [{"product_name": "", "dose": "", "administration": "",
                                         "restrictions": "", "modifications": ""}],
    },
    "site_requirements": {
        "site_requirements": [{"equipment": [], "certifications": [],
                               "sample_handling": {"sample_description": "", "processing_timeline": "",
                                                   "storage_conditions": "", "transport_requirements": "",
                                                   "access_control": "", "chain_of_custody": ""}}],
        "ip_storage": [{"product_name": "", "temperature_requirements": "", "environmental_conditions": "",
                        "storage_location": "", "access_restrictions": "", "accountability_tracking": ""}],
    },
    "schedule_of_events": {
        "schedule_of_events": [{"visit_number": 0, "visit_time": "", "procedures": [{"procedure_name": ""}]}],
    },
}

CATEGORY_CODES = {"gen": "general_information", "ie": "inclusion_exclusion", "ae": "adverse_events",
                  "inter": "interventions", "site": "site_requirements", "soe": "schedule_of_events"}


@dataclass
class Element:
    key: str
    category: str                  # schema key, e.g. "adverse_events"
    label: str                     # Table 7 row
    weight: float
    paths: list[str]
    queries: list[str]
    guide: str = ""
    top_k: int | None = None
    extra: dict = field(default_factory=dict)

    def template(self) -> dict:
        """The JSON fragment the model must fill: {path: empty value from the schema}."""
        return {p: copy.deepcopy(get_path(SCHEMAS[self.category], p)) for p in self.paths}


G, IE, AE, IN, SR = ("general_information", "inclusion_exclusion", "adverse_events",
                     "interventions", "site_requirements")
GI, OE = "general_information.", "objectives_and_endpoints."
ADC, ARR, ASM, ASP = ("ae_definitions_and_classifications.", "reporting_requirements.",
                      "safety_monitoring_and_management.", "specific_ae_information.")
SR0 = "site_requirements[0]."

ELEMENTS: list[Element] = [
    # ---- General information (Table 7 "gen") ------------------------------------------
    Element("nct_id", G, "Study NCT ID", 0.15, [GI + "study_nct_id"],
            ["NCT number ClinicalTrials.gov identifier", "study registration number NCT"],
            "Copy the NCT number exactly: 'NCT' followed by 8 digits."),
    Element("title", G, "Study Title", 0.08, [GI + "study_title"],
            ["protocol title", "full study title title page"], "Full official title, verbatim."),
    Element("protocol_id", G, "Protocol ID, Version, Date", 0.07,
            [GI + "protocol_id", GI + "protocol_version", GI + "protocol_date"],
            ["protocol number version amendment date", "protocol identifier version history"],
            "Use the most recent version/amendment and its date."),
    Element("sponsor", G, "Sponsor Name", 0.05, [GI + "sponsor_name"],
            ["sponsor name", "study sponsor funding organisation"]),
    Element("phase", G, "Phase", 0.10, [GI + "phase"], ["study phase phase 1 2 3", "phase of development"]),
    Element("therapeutic_area", G, "Therapeutic Area", 0.01, [GI + "therapeutic_area"],
            ["therapeutic area indication disease background"]),
    Element("disease", G, "Disease/Condition", 0.03, [GI + "disease_or_condition"],
            ["disease condition under study indication", "patient population diagnosis"]),
    Element("allocation", G, "Allocation", 0.02, [GI + "allocation"],
            ["randomization allocation randomized non-randomized", "study design treatment assignment"]),
    Element("masking", G, "Masking", 0.02, [GI + "masking"],
            ["blinding masking open-label double-blind", "unblinding procedures"]),
    Element("enrollment", G, "Target Enrollment", 0.00, [GI + "target_enrollment"],
            ["sample size number of participants enrolled", "target accrual statistical considerations"]),
    Element("sites", G, "Countries and Sites", 0.00, [GI + "number_of_sites", GI + "countries"],
            ["number of sites centers countries", "participating institutions"]),
    Element("timeline", G, "Timeline (Start/End/Duration)", 0.00, [GI + "estimated_duration"],
            ["study duration treatment duration follow-up period", "duration of participation end of study"],
            "Give the participant's total duration (treatment + follow-up) with units, and the study duration if stated."),
    Element("primary_objectives", G, "Primary Objectives", 0.25, [OE + "primary_objectives"],
            ["primary objective", "study objectives primary aim"], "One list item per objective, verbatim or close."),
    Element("secondary_objectives", G, "Secondary Objectives", 0.02, [OE + "secondary_objectives"],
            ["secondary objectives", "exploratory objectives secondary aims"]),
    Element("endpoints", G, "Primary/Secondary Endpoints", 0.20,
            [OE + "primary_endpoints", OE + "secondary_endpoints", OE + "measurement_method",
             OE + "endpoint_timepoint"],
            ["primary endpoint outcome measure", "secondary endpoints outcome measures time frame",
             "endpoint assessment method"]),
    # ---- Inclusion / exclusion ("ie") -------------------------------------------------
    Element("inclusion", IE, "Inclusion Criteria", 0.50, ["inclusion"],
            ["inclusion criteria", "eligibility criteria patients must meet all of the following",
             "laboratory values required for eligibility"],
            "Every inclusion criterion as its own list item, verbatim, keeping thresholds, units and sub-items.",
            top_k=12),
    Element("exclusion", IE, "Exclusion Criteria", 0.50, ["exclusion"],
            ["exclusion criteria", "patients will be excluded if any of the following apply",
             "prohibited prior therapy exclusion"],
            "Every exclusion criterion as its own list item, verbatim, keeping thresholds, units and sub-items.",
            top_k=12),
    # ---- Adverse events ("ae") --------------------------------------------------------
    Element("ae_def", AE, "AE Definition", 0.15, [ADC + "ae_definition"],
            ["definition of adverse event", "adverse event is any untoward medical occurrence"]),
    Element("sae_def", AE, "SAE Definition", 0.20, [ADC + "sae_definition"],
            ["serious adverse event definition", "SAE results in death life-threatening hospitalization"]),
    Element("severity", AE, "Severity Grading", 0.05, [ADC + "severity_grading"],
            ["severity grading CTCAE version", "intensity mild moderate severe grade"]),
    Element("relationship", AE, "AE Relationship Assessment", 0.02, [ADC + "ae_relationship"],
            ["causality relationship to study drug attribution", "related unrelated possibly probably"]),
    Element("reporting_timeframes", AE, "Reporting Timeframes", 0.12, [ARR + "reporting_timeframes"],
            ["SAE reporting within 24 hours", "expedited reporting timeframe IND safety report",
             "adverse event reporting period"]),
    Element("data_collection", AE, "Data Collection Requirements", 0.08, [ARR + "ae_data_collection_requirements"],
            ["adverse event recording collection case report form", "period of AE collection"]),
    Element("reporting_contacts", AE, "Reporting Contacts", 0.02, [ARR + "reporting_contacts"],
            ["SAE reporting contact fax email pharmacovigilance", "report to sponsor medical monitor IRB"]),
    Element("safety_monitoring", AE, "Safety Monitoring Plan", 0.08, [ASM + "safety_monitoring_plan"],
            ["data safety monitoring board DSMB", "safety monitoring plan stopping rules"]),
    Element("discontinuation", AE, "Discontinuation Criteria", 0.12, [ASM + "discontinuation_criteria"],
            ["discontinuation of study treatment criteria", "withdrawal of participants",
             "treatment must be permanently discontinued if"],
            "Include lab thresholds and dose-modification triggers that stop treatment."),
    Element("emergency", AE, "Emergency Procedures", 0.05, [ASM + "emergency_procedures"],
            ["emergency unblinding procedures", "overdose management emergency"]),
    Element("expected_aes", AE, "Expected AEs", 0.04, [ASP + "expected_aes"],
            ["expected adverse events known toxicities", "risks of study drug side effects"]),
    Element("potential_risks", AE, "Potential Risks", 0.03, [ASP + "potential_risks"],
            ["potential risks benefits", "risk assessment"]),
    Element("conmeds", AE, "Concomitant Medication Restrictions", 0.03,
            [ASP + "concomitant_medication_restrictions"],
            ["prohibited concomitant medications", "permitted concomitant therapy restrictions CYP3A4"]),
    Element("special_pop", AE, "Special Population Considerations", 0.01,
            [ASP + "special_population_considerations"],
            ["pregnancy contraception women of childbearing potential", "elderly renal hepatic impairment"]),
    # ---- Interventions ("inter") ------------------------------------------------------
    Element("arms", IN, "Intervention Arms", 0.50, ["arms"],
            ["treatment arms study design schema", "arm A arm B intervention dose schedule",
             "dosing regimen cycle length"],
            "One entry per arm; each intervention with its type (drug, placebo, procedure...), name, "
            "dose with units, and schedule (days/cycle).", top_k=10),
    Element("treatment_info", IN, "Treatment-Level Information", 0.50, ["treatment_level_information"],
            ["study drug administration route dose", "dose modifications dose reductions delays",
             "investigational product description restrictions"],
            "One entry per product; 'modifications' must keep the dose levels and the triggers.", top_k=10),
    # ---- Site requirements ("site") ---------------------------------------------------
    Element("equipment", SR, "Equipment", 0.20, [SR0 + "equipment"],
            ["equipment required ECG MRI centrifuge freezer", "imaging scanner laboratory equipment"],
            "List each piece of equipment or facility the site needs."),
    Element("certifications", SR, "Certifications", 0.10, [SR0 + "certifications"],
            ["training certification GCP", "site qualification accreditation CLIA"]),
    Element("sample_handling", SR, "Sample Handling", 0.35, [SR0 + "sample_handling"],
            ["blood sample collection processing centrifuge", "specimen storage shipping frozen -80",
             "laboratory manual biospecimen chain of custody"]),
    Element("ip_storage", SR, "IP Storage", 0.35, ["ip_storage"],
            ["study drug storage temperature", "investigational product storage handling accountability",
             "store refrigerated 2-8 C protect from light"]),
]

ELEMENT_BY_KEY = {e.key: e for e in ELEMENTS}
SOE_ELEMENT = Element("soe", "schedule_of_events", "Schedule of Events", 1.00, ["schedule_of_events"],
                      ["schedule of assessments", "schedule of events study calendar visit procedures"])


def element_for_field(field_id: str) -> Element | None:
    """Which data element produced a field, e.g. 'interventions.arms[0].interventions[1].dosage' -> arms."""
    category, _, rest = field_id.partition(".")
    for el in all_elements():
        if el.category == category and any(rest == p or rest.startswith(p + ".") or rest.startswith(p + "[")
                                           for p in el.paths):
            return el
    return None


def elements_for(category: str) -> list[Element]:
    if category == "schedule_of_events":
        return [SOE_ELEMENT]
    return [e for e in ELEMENTS if e.category == category]


def all_elements() -> list[Element]:
    return ELEMENTS + [SOE_ELEMENT]


# ---- dotted-path helpers ("a.b[0].c") ---------------------------------------------------
_PART = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def _parts(path: str) -> list[Any]:
    return [int(i) if i else k for k, i in _PART.findall(path)]


def get_path(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for p in _parts(path):
        try:
            cur = cur[p]
        except (KeyError, IndexError, TypeError):
            return default
    return cur


def set_path(obj: Any, path: str, value: Any) -> None:
    parts = _parts(path)
    cur = obj
    for p, nxt in zip(parts, parts[1:]):
        if isinstance(p, int):
            while len(cur) <= p:
                cur.append({} if isinstance(nxt, str) else [])
            cur = cur[p]
        else:
            if p not in cur or cur[p] is None:
                cur[p] = {} if isinstance(nxt, str) else []
            cur = cur[p]
    last = parts[-1]
    if isinstance(last, int):
        while len(cur) <= last:
            cur.append(None)
    cur[last] = value


def empty_category(category: str) -> dict:
    """Schema with list placeholders emptied, ready to be filled by element chains."""
    doc = copy.deepcopy(SCHEMAS[category])
    for k in ("arms", "treatment_level_information", "ip_storage", "schedule_of_events"):
        if k in doc:
            doc[k] = []
    return doc
