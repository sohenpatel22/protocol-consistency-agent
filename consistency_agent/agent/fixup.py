"""Run B3: check an extraction, run the fix-up agent, and apply its suggested fixes to a copy.

This is for scoring only. In real use a reviewer approves each fix; here every "suggest_fix"
is accepted so we can measure "accuracy if the reviewer accepted every suggestion", and we
report how many conflicts were escalated or turned into sponsor queries instead.
"""
from __future__ import annotations

import copy

from ..replication.elements import get_path, set_path
from .pipeline import check
from .resolve import resolve_all


def fixup(result: dict, system: str, mode: str = "hybrid", severities=("critical", "major"), log=print) -> dict:
    nct = result["nct_id"]
    ctx = {"registry_nct_id": nct, "nct_id": nct, "provenance": result.get("provenance", {})}
    findings, stats = check(result["extraction"], mode, ctx, log=log)
    todo = [f for f in findings if f.severity in severities]
    proposals = resolve_all(nct, result["extraction"], todo, ctx, log=log)
    fixed = copy.deepcopy(result["extraction"])
    applied = []
    for p in proposals:
        if p["action"] == "suggest_fix":
            field = p["proposal"]["field"]
            applied.append({"finding": p["finding_id"], "field": field, "old": get_path(fixed, field),
                            "new": p["proposal"]["proposed_value"], "cause": p["proposal"].get("cause")})
            set_path(fixed, field, p["proposal"]["proposed_value"])
    actions = [p["action"] for p in proposals]
    return {**result, "system": system, "extraction": fixed,
            "fixup": {"from": result.get("system"), "n_findings": len(findings), "n_resolved_attempted": len(todo),
                      "applied": applied, "n_applied": len(applied),
                      "n_escalated": actions.count("escalate"), "n_sponsor_queries": actions.count("sponsor_query"),
                      "n_suspects_dropped": stats.get("n_suspects_dropped", 0)}}
