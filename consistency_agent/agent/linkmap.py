"""Which fields in different sections have to agree, and what we call it when they don't."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .normalize import Field, fields_matching


@dataclass(frozen=True)
class Link:
    type: str
    a: str                 # field-id regex, side A
    b: str                 # field-id regex, side B
    severity: str          # critical | major | minor
    question: str


LINKS: list[Link] = [
    Link("dose_mismatch", r"^interventions\.arms\[\d+\]\.interventions\[\d+\]\.(name|dosage)$",
         r"^interventions\.treatment_level_information\[\d+\]\.(product_name|dose|modifications)$", "critical",
         "Does the dose given for a product in the arm description disagree with the dose in the "
         "treatment-level information (after unit conversion)?"),
    Link("dose_modification_vs_ae", r"^interventions\.treatment_level_information\[\d+\]\.modifications$",
         r"^adverse_events\.safety_monitoring_and_management\.discontinuation_criteria$", "critical",
         "Does the dose-modification rule in the interventions section (reduce/hold the dose at a toxicity "
         "grade) contradict the adverse-event section (permanently stop treatment at the same or a lower "
         "grade of the same toxicity)?"),
    Link("lab_threshold_conflict", r"^inclusion_exclusion\.(inclusion|exclusion)\[\d+\]$",
         r"^adverse_events\.safety_monitoring_and_management\.discontinuation_criteria$", "critical",
         "Would a participant who just meets a laboratory eligibility threshold already meet a "
         "treatment-discontinuation threshold for the same lab test?"),
    Link("duration_vs_schedule", r"^general_information\.general_information\.estimated_duration$",
         r"^schedule_of_events\.schedule_of_events\[\d+\]\.visit_time$", "major",
         "Does the stated duration of participation disagree with the timing of the last visits in the "
         "schedule of events?"),
    Link("storage_vs_equipment", r"^site_requirements\.(ip_storage\[\d+\]\.temperature_requirements|site_requirements\[\d+\]\.sample_handling\.storage_conditions)$",
         r"^site_requirements\.site_requirements\[\d+\]\.equipment\[\d+\]$", "major",
         "Does a storage temperature require equipment (refrigerator, -20 C freezer, -70/-80 C freezer) "
         "that is missing from the site equipment list?"),
    Link("procedure_vs_equipment", r"^schedule_of_events\.schedule_of_events\[\d+\]\.procedures\[\d+\]\.procedure_name$",
         r"^site_requirements\.site_requirements\[\d+\]\.equipment\[\d+\]$", "minor",
         "Does a scheduled procedure need equipment (ECG machine, MRI, CT, echocardiography, centrifuge) "
         "that is missing from the site equipment list?"),
    Link("drug_name_mismatch", r"^interventions\..*\.(name|product_name)$",
         r"^site_requirements\.ip_storage\[\d+\]\.product_name$", "minor",
         "Is the same product named inconsistently (misspelling, different drug) between interventions "
         "and IP storage?"),
    Link("nct_id_invalid", r"^general_information\.general_information\.study_nct_id$",
         r"^general_information\.general_information\.study_nct_id$", "critical",
         "Is the NCT ID malformed (not NCT + 8 digits) or different from the registry record?"),
    Link("prohibited_med_conflict", r"^adverse_events\.specific_ae_information\.concomitant_medication_restrictions$",
         r"^interventions\..*\.(name|product_name)$", "critical",
         "Does the concomitant-medication section prohibit a drug that the protocol itself gives as a "
         "study intervention?"),
    Link("safety_lab_not_scheduled", r"^adverse_events\.safety_monitoring_and_management\.(discontinuation_criteria|safety_monitoring_plan)$",
         r"^schedule_of_events\.schedule_of_events\[\d+\]\.procedures\[\d+\]\.procedure_name$", "major",
         "Do the stopping/monitoring rules depend on a lab test or ECG that never appears in the "
         "schedule of events?"),
    Link("boilerplate_duplicate", r"^adverse_events\.[a-z_]+\.[a-z_]+$", r"^adverse_events\.[a-z_]+\.[a-z_]+$",
         "minor", "Are two different adverse-event fields filled with the same copied text?"),
    Link("phase_mismatch", r"^general_information\.general_information\.phase$",
         r"^general_information\.general_information\.study_title$", "major",
         "Does the phase field disagree with the phase stated in the study title?"),
]
LINK_BY_TYPE = {l.type: l for l in LINKS}
SEVERITY_RANK = {"critical": 3, "major": 2, "minor": 1}


@dataclass
class Evidence:
    field: str
    quote: str


@dataclass
class Finding:
    type: str
    fields: list[str]
    evidence: list[Evidence]
    explanation: str
    source: str                         # rules | llm | hybrid
    severity: str = ""
    confidence: float = 0.0             # raw score before calibration
    calibrated: float | None = None
    likely_cause: str = ""              # extraction_error | missing_condition | protocol_inconsistency
    id: str = ""
    extra: dict = field(default_factory=dict)

    def key(self) -> tuple:
        return (self.type, tuple(sorted(self.fields)))

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Finding":
        d = dict(d)
        d["evidence"] = [Evidence(**e) for e in d.get("evidence", [])]
        return Finding(**d)


def link_fields(fields: list[Field], link: Link) -> tuple[list[Field], list[Field]]:
    return fields_matching(fields, link.a), fields_matching(fields, link.b)
