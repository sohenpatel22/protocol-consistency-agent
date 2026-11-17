"""Run the checker on one extraction and write the findings and a short report."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import FINDINGS_OUT
from . import audit
from .linkmap import Finding
from .rules import run_rules
from .severity import merge, rank

MODES = ("rules", "llm", "hybrid")


def check(extraction: dict, mode: str = "hybrid", context: dict | None = None, log=print) -> tuple[list[Finding], dict]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    stats: dict = {"mode": mode}
    rule_f = run_rules(extraction, context) if mode in ("rules", "hybrid") else []
    llm_f: list[Finding] = []
    if mode in ("llm", "hybrid"):
        from .verifier import run_verifier
        llm_f, vstats = run_verifier(extraction, context, log=log)
        stats.update(vstats)
    merged = merge(rule_f, llm_f)
    # Open-sweep findings that no typed check confirmed are only suspects: the sweep saw the JSON,
    # not the protocol. Keep one only if the protocol text backs it up; otherwise list it separately.
    findings = [f for f in merged if not (f.type == "other_conflict" and f.source == "llm")]
    suspects = [f for f in merged if f.type == "other_conflict" and f.source == "llm"]
    stats["suspects"] = []
    for f in suspects:
        result = _confirm(f, extraction, context or {}, log)
        if result.get("confirmed"):
            f.extra["confirmed_in_protocol"] = result
            if result.get("likely_cause") in ("extraction_error", "missing_condition", "protocol_inconsistency"):
                f.likely_cause = result["likely_cause"]
            findings.append(f)
        else:
            stats["suspects"].append({**f.to_dict(), "confirmation": result})
    findings = rank(findings)
    stats["n_findings"] = len(findings)
    stats["n_suspects_dropped"] = len(stats["suspects"])
    return findings, stats


def _confirm(f: Finding, extraction: dict, context: dict, log) -> dict:
    """Check a sweep suspect against the protocol. Without a protocol to read, it stays a suspect."""
    from ..config import RAW
    from ..llm import LLMError
    from .resolve import confirm_suspect
    nct = context.get("nct_id")
    retriever = context.get("retriever")
    if retriever is None and not (nct and (RAW / f"{nct}.pdf").exists()):
        return {"confirmed": False, "reason": "no protocol available to confirm against"}
    try:
        return confirm_suspect(nct or "", extraction, f, retriever)
    except (LLMError, ValueError) as exc:
        log(f"  confirm suspect: ERROR {exc}")
        return {"confirmed": False, "reason": f"error: {exc}"}


def attach_sources(findings: list[Finding], provenance: dict) -> None:
    """Add the protocol pages/sections each conflicting value was extracted from."""
    from ..replication.elements import element_for_field
    for f in findings:
        src = {}
        for fid in f.fields:
            el = element_for_field(fid)
            prov = provenance.get(el.key, {}) if el else {}
            cited = next((v for k, v in prov.get("fields", {}).items() if fid.split(".", 1)[1].startswith(k)), None)
            if cited:
                src[fid] = {"pages": [cited.get("page")], "section": cited.get("section"), "quote": cited.get("quote")}
            elif prov:
                src[fid] = {"pages": prov.get("pages", [])[:12], "sections": prov.get("sections", [])[:4]}
        f.extra["sources"] = src


def run(nct_id: str, extraction: dict, mode: str = "hybrid", context: dict | None = None,
        provenance: dict | None = None, log=print) -> dict:
    context = {**(context or {}), "nct_id": nct_id, "provenance": provenance or {}}
    findings, stats = check(extraction, mode, context, log)
    attach_sources(findings, provenance or {})
    suspects = stats.pop("suspects", [])
    out_dir = FINDINGS_OUT / nct_id
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"nct_id": nct_id, "mode": mode, "extraction_digest": audit.digest(extraction), "stats": stats,
              "findings": [f.to_dict() for f in findings], "unconfirmed_suspects": suspects}
    (out_dir / f"findings_{mode}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"report_{mode}.md").write_text(to_markdown(report), encoding="utf-8")
    for f in findings:
        audit.record(nct_id, "finding", finding_id=f.id, type=f.type, severity=f.severity, source=f.source,
                     fields=f.fields, confidence=f.confidence, extraction_digest=report["extraction_digest"])
    return report


def load_findings(nct_id: str, mode: str = "hybrid") -> tuple[dict, list[Finding]]:
    rep = json.loads((FINDINGS_OUT / nct_id / f"findings_{mode}.json").read_text(encoding="utf-8"))
    return rep, [Finding.from_dict(d) for d in rep["findings"]]


def to_markdown(report: dict) -> str:
    lines = [f"# Consistency report: {report['nct_id']}", "",
             f"Mode: `{report['mode']}`. Findings: {len(report['findings'])}. "
             f"Extraction digest: `{report['extraction_digest']}`.", ""]
    if not report["findings"]:
        lines.append("No cross-section conflicts found.")
    for f in report["findings"]:
        conf = f["calibrated"] if f.get("calibrated") is not None else f["confidence"]
        lines += [f"## {f['id']} [{f['severity']}] {f['type']}",
                  f"- Source: {f['source']}; confidence: {conf}; likely cause: {f.get('likely_cause') or 'n/a'}",
                  f"- {f['explanation']}"]
        for e in f["evidence"]:
            src = f.get("extra", {}).get("sources", {}).get(e["field"], {})
            where = f" (protocol p. {', '.join(str(p) for p in src['pages'])})" if src.get("pages") else ""
            lines.append(f"  - `{e['field']}`{where}: \"{e['quote'][:300]}\"")
        lines.append("")
    if report.get("unconfirmed_suspects"):
        lines += ["## Open-sweep suspects the protocol did not confirm (not reported as findings)", ""]
        for s in report["unconfirmed_suspects"]:
            lines.append(f"- {s['explanation']} ({s['confirmation'].get('reason', '')})")
    return "\n".join(lines)


def write_proposals(nct_id: str, proposals: list[dict]) -> Path:
    p = FINDINGS_OUT / nct_id / "proposals.json"
    p.write_text(json.dumps(proposals, indent=1, ensure_ascii=False), encoding="utf-8")
    return p
