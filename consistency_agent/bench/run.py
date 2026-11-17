"""Run the benchmark: plant, check, score.

Findings on each planted copy are compared with findings on the untouched extraction, so
problems that were already there don't count for or against the planted one."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ..agent.pipeline import check
from ..agent.severity import fit_platt, save_calibration
from ..agent import normalize as N
from ..config import BENCH, RAW, RESULTS
from . import metrics
from .inject import Case, generate


def _matches(f, case: Case) -> bool:
    if f.type != case.type and f.type != "other_conflict":   # the open sweep may catch it too
        return False
    return bool(set(f.fields) & set(case.fields)) or case.operator in ("procedure_mri", "safety_lab_remove")


def evaluate_case(case: Case, base_keys: set, mode: str, context: dict, resolve: bool = False) -> dict:
    findings, stats = check(case.doc, mode, context, log=lambda *_: None)
    new = [f for f in findings if f.key() not in base_keys]
    hit = [f for f in new if case.type and _matches(f, case)]
    fps = [f for f in new if f not in hit]
    row = {"case_id": case.case_id, "doc_id": case.case_id.split(":")[0], "operator": case.operator,
           "type": case.type, "hard_negative": case.hard_negative,
           "severity": (hit[0].severity if hit else _sev(case.type)), "detected": bool(hit),
           "n_false_pos": len(fps), "false_pos": [{"type": f.type, "fields": f.fields} for f in fps],
           "confidences": [(f.confidence, True) for f in hit[:1]] + [(f.confidence, False) for f in fps],
           "note": case.note}
    if resolve and hit and case.original:
        row["resolution"] = score_resolution(case, hit[0], context)
    return row


def score_resolution(case: Case, finding, context: dict) -> dict:
    """Every planted conflict is an extraction-side change, so the right fix restores the original
    value. Success = the loop proposes that fix and the re-check is clean."""
    from ..agent.resolve import decide, propose, recheck
    nct = case.case_id.split(":")[0]
    if not (RAW / f"{nct}.pdf").exists():
        return {"skipped": "no protocol PDF"}
    try:
        prop = propose(nct, case.doc, finding, context=context)
    except Exception as exc:                       # keep the benchmark running; count as a failure
        return {"success": False, "error": str(exc)}
    chk = recheck(case.doc, finding, prop, context)
    action = decide(prop, chk)
    field = prop.get("field")
    want = case.original.get(field) if field else None
    restored = want is not None and N.similar(N.norm_quote(str(prop.get("proposed_value"))),
                                              N.norm_quote(str(want))) >= 0.9
    return {"action": action, "field": field, "restored": restored,
            "new_conflict": bool(chk.get("new_conflicts")), "success": action == "suggest_fix" and restored}


def _sev(t):
    from ..agent.linkmap import LINK_BY_TYPE
    return LINK_BY_TYPE[t].severity if t in LINK_BY_TYPE else ""


def run(bases: dict[str, dict], modes: list[str], seed: int = 0, operators: list[str] | None = None,
        registry: dict[str, str] | None = None, tag: str = "", resolve: bool = False, log=print) -> dict:
    """``bases``: doc_id -> extraction dict. Returns and writes a summary per mode."""
    registry = registry or {}
    cases_by_doc = {d: generate(doc, d, seed, operators) for d, doc in bases.items()}
    n_cases = sum(map(len, cases_by_doc.values()))
    log(f"{len(bases)} base documents, {n_cases} injected cases")
    summary = {"created": datetime.now().isoformat(timespec="seconds"), "seed": seed, "tag": tag, "modes": {}}
    rows_by_mode = {}
    for mode in modes:
        rows = []
        for doc_id, doc in bases.items():
            ctx = {"registry_nct_id": registry.get(doc_id), "nct_id": doc_id}
            base_f, _ = check(doc, mode, ctx, log=lambda *_: None)
            base_keys = {f.key() for f in base_f}
            for case in cases_by_doc[doc_id]:
                rows.append(evaluate_case(case, base_keys, mode, ctx, resolve=resolve and mode == modes[-1]))
        rows_by_mode[mode] = rows
        summary["modes"][mode] = metrics.summarise(rows)
        res = [r["resolution"] for r in rows if "success" in r.get("resolution", {})]
        if res:
            crit = [r["resolution"] for r in rows if "success" in r.get("resolution", {}) and r["severity"] == "critical"]
            summary["modes"][mode]["re_extraction_success_rate"] = round(sum(x["success"] for x in res) / len(res), 3)
            summary["modes"][mode]["agent_kpi_critical_resolved_no_new_conflict"] = (
                round(sum(x["success"] and not x["new_conflict"] for x in crit) / len(crit), 3) if crit else None)
        o = summary["modes"][mode]["overall"]
        log(f"[{mode}] precision {o['precision']} recall {o['recall']} f1 {o['f1']} "
            f"hard-neg FPR {summary['modes'][mode]['hard_negative_fpr']}")
    # Platt calibration from the richest mode that ran
    for mode in ("hybrid", "llm", "rules"):
        if mode in rows_by_mode:
            pairs = [c for r in rows_by_mode[mode] for c in r["confidences"]]
            if len({t for _, t in pairs}) == 2:
                save_calibration({**fit_platt([p for p, _ in pairs], [int(t) for _, t in pairs]), "mode": mode})
            break
    out = RESULTS / f"bench_summary{('_' + tag) if tag else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    detail = BENCH / f"bench_cases{('_' + tag) if tag else ''}.json"
    detail.parent.mkdir(parents=True, exist_ok=True)
    detail.write_text(json.dumps(rows_by_mode, indent=1), encoding="utf-8")
    log(f"Wrote {out}")
    return summary


def load_bases(paths: list[Path]) -> tuple[dict[str, dict], dict[str, str]]:
    bases, registry = {}, {}
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        bases[d["nct_id"]] = d["extraction"]
        registry[d["nct_id"]] = d["nct_id"]
    return bases, registry
