"""One-command pilot on one (or a few) protocols, with a hard spending cap.

Steps (each is skipped if its output already exists, so a stopped run can be resumed):
  1. reference answers (the paper's method)
  2. Run A: the paper's pipeline, scored by the judge          (optional, --no-baseline)
  3. Run B2: our pipeline (Opus 5.5 + upgrades), scored
  4. consistency check on B2 (findings report)
  5. Run B3: fix-up agent, fixes applied, scored
  6. planted-contradiction benchmark (rules: free; --bench-llm adds the LLM checks)
Writes results/pilot_<NCT>.md and .json with accuracy, findings, fixes, benchmark and cost.
"""
from __future__ import annotations

import json
import os
import time

from . import llm
from .config import JUDGEMENTS, RAW, RESULTS, SETTINGS, apply_variant
from .replication.extract import load, out_path, save

SECTION_NAMES = {"gen": "General info", "ie": "Inclusion/exclusion", "ae": "Adverse events",
                 "inter": "Interventions", "site": "Site requirements", "soe": "Schedule of events",
                 "overall": "Overall"}


def _over_budget(cap: float) -> bool:
    return llm.spent() >= cap


def run_pilot(nct: str, max_usd: float, baseline: bool = True, bench_llm: bool = False, effort: str = "low",
              log=print) -> dict:
    from .agent.fixup import fixup
    from .agent.linkmap import Finding
    from .agent.pipeline import run as check_run
    from .bench import metrics
    from .bench.inject import generate
    from .bench.run import evaluate_case
    from .agent.pipeline import check
    from .replication import annotate, extract, judge
    from .replication.annotate import load_gt
    from .replication.sampling import download, load_sample

    os.environ["AGENT_MAX_USD"] = str(max_usd)
    t0 = time.time()
    study = next(s for s in load_sample() if s["nct_id"] == nct)
    download(study["protocol_url"], RAW / f"{nct}.pdf")
    steps: list[str] = []

    def step(name, fn):
        if _over_budget(max_usd):
            steps.append(f"{name}: skipped (spending cap reached)")
            return None
        log(f"[pilot] {name} (spent so far ~${llm.spent():.2f})")
        out = fn()
        steps.append(f"{name}: done")
        return out

    # 1. reference answers
    try:
        load_gt(nct)
        steps.append("reference answers: already built")
    except FileNotFoundError:
        apply_variant("A")
        step("reference answers", lambda: annotate.annotate_protocol(nct, log=log))

    def extract_and_judge(variant: str):
        system = apply_variant(variant)
        if variant != "A":
            SETTINGS.effort = effort
        step(f"extract {variant}", lambda: extract.run_resumable(nct, system, log=log))
        if out_path(system, nct).exists() and not (JUDGEMENTS / system / f"{nct}.json").exists():
            step(f"judge {variant}", lambda: judge.judge_protocol(system, nct, log=log))
        return system

    # 2-3. Run A and Run B2
    if baseline:
        extract_and_judge("A")
    b2 = extract_and_judge("B2")

    # 4-5. check B2, then fix-up -> B3
    report, b3 = None, "rag_b3"
    if out_path(b2, nct).exists():
        res = load(b2, nct)
        report = step("consistency check on B2",
                      lambda: check_run(nct, res["extraction"], "hybrid", {"registry_nct_id": nct},
                                        res.get("provenance"), log=log))
        if report is not None and not out_path(b3, nct).exists():
            found = [Finding.from_dict(d) for d in report["findings"]]
            step("fix-up agent -> B3", lambda: save(fixup(res, b3, "hybrid", log=log, findings=found,
                                                          stats=report["stats"])))
        if out_path(b3, nct).exists() and not (JUDGEMENTS / b3 / f"{nct}.json").exists():
            step("judge B3", lambda: judge.judge_protocol(b3, nct, log=log))

    # 6. benchmark on B2's extraction
    bench = {}
    if out_path(b2, nct).exists():
        doc = load(b2, nct)["extraction"]
        for mode in ["rules"] + (["hybrid"] if bench_llm else []):
            def _bench(mode=mode):
                ctx = {"registry_nct_id": nct, "nct_id": nct}
                base_keys = {f.key() for f in check(doc, mode, ctx, log=lambda *_: None)[0]}
                rows = [evaluate_case(c, base_keys, mode, ctx) for c in generate(doc, nct)]
                return metrics.summarise(rows, n_boot=200)
            bench[mode] = step(f"benchmark ({mode})", _bench)

    summary = _summarise(nct, study, report, bench, steps, time.time() - t0, max_usd)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"pilot_{nct}.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    (RESULTS / f"pilot_{nct}.md").write_text(to_markdown(summary), encoding="utf-8")
    return summary


def _scores(system: str, nct: str) -> dict | None:
    p = JUDGEMENTS / system / f"{nct}.json"
    return json.loads(p.read_text(encoding="utf-8"))["categories"] if p.exists() else None


def _summarise(nct, study, report, bench, steps, seconds, cap) -> dict:
    fx = None
    if out_path("rag_b3", nct).exists():
        fx = load("rag_b3", nct).get("fixup")
    findings = report["findings"] if report else []
    return {
        "nct_id": nct, "title": study["title"], "pages": study.get("pages"),
        "accuracy": {"A (paper method)": _scores("rag", nct), "B2 (our pipeline)": _scores("rag_b2", nct),
                     "B3 (+ fix-up agent)": _scores("rag_b3", nct)},
        "findings": {"total": len(findings),
                     "by_severity": {s: sum(f["severity"] == s for f in findings) for s in ("critical", "major", "minor")},
                     "by_type": {t: sum(f["type"] == t for f in findings) for t in sorted({f["type"] for f in findings})},
                     "suspects_dropped": (report or {}).get("stats", {}).get("n_suspects_dropped", 0),
                     "examples": [{k: f[k] for k in ("severity", "type", "explanation")} for f in findings[:5]]},
        "fixup": fx, "benchmark": bench, "steps": steps,
        "cost_usd_est": round(llm.spent(), 2), "cap_usd": cap, "minutes": round(seconds / 60, 1),
        "settings": {"embeddings": SETTINGS.embeddings, "rerank": SETTINGS.rerank or "off (no Cohere key)",
                     "b2_extractor": "anthropic:claude-opus-5-5", "effort": SETTINGS.effort},
    }


def to_markdown(s: dict) -> str:
    L = [f"# Pilot: {s['nct_id']}", "", f"{s['title']} ({s['pages']} pages)", "",
         "**One protocol is a demonstration, not a statistical result.**", "", "## Extraction accuracy (judge, %)", ""]
    systems = [k for k, v in s["accuracy"].items() if v]
    if systems:
        L += ["| Section | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
        for key, name in SECTION_NAMES.items():
            L.append(f"| {name} | " + " | ".join(str(s["accuracy"][k].get(key, "-")) for k in systems) + " |")
    else:
        L.append("Not scored yet.")
    f = s["findings"]
    L += ["", "## Consistency check (B2)", "",
          f"- Findings: **{f['total']}** (critical {f['by_severity']['critical']}, major {f['by_severity']['major']}, "
          f"minor {f['by_severity']['minor']})",
          f"- Open-sweep suspects not confirmed in the protocol (dropped): {f['suspects_dropped']}"]
    for t, n in f["by_type"].items():
        L.append(f"  - {t.replace('_', ' ')}: {n}")
    for e in f["examples"]:
        L.append(f"- [{e['severity']}] {e['explanation']}")
    if s.get("fixup"):
        x = s["fixup"]
        L += ["", "## Fix-up agent (B3)", "",
              f"- Conflicts sent to the agent: {x['n_resolved_attempted']}; fixes suggested and applied: {x['n_applied']}; "
              f"escalated to a person: {x['n_escalated']}; sponsor queries: {x['n_sponsor_queries']}"]
    for mode, b in (s.get("benchmark") or {}).items():
        if b:
            o = b["overall"]
            L += ["", f"## Planted contradictions ({mode})", "",
                  f"- Recall {o['recall']}, precision {o['precision']}, critical recall {b['critical_recall']}, "
                  f"look-alikes wrongly flagged {b['hard_negative_fpr']} ({b['n_conflicts']} planted, {b['n_hard_negatives']} look-alikes)"]
    L += ["", "## Run", "", f"- Estimated cost: ~${s['cost_usd_est']} (cap ${s['cap_usd']}); time {s['minutes']} min",
          f"- Retrieval: {s['settings']['embeddings']}; reranker: {s['settings']['rerank']}"]
    L += [f"- {x}" for x in s["steps"]]
    return "\n".join(L)
