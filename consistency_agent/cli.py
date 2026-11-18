"""Command line entry point. Run `consistency-agent -h` to see the commands."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

from .config import FINDINGS_OUT, EXTRACTIONS, RESULTS, ROOT, RUNS, SETTINGS, ensure_dirs

FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_extraction.json"


def _ids(args) -> list[str]:
    from .replication.sampling import load_sample
    ids = [s["nct_id"] for s in load_sample()]
    if getattr(args, "ids", None):
        want = set(args.ids.split(","))
        ids = [i for i in ids if i in want]
    if getattr(args, "limit", None):
        ids = ids[:args.limit]
    return ids


def cmd_sample(args):
    from .replication import sampling
    if args.download_only:
        sampling.ensure_pdfs(sampling.load_sample())
    else:
        sampling.run()


def cmd_extract(args):
    from .config import apply_variant
    from .replication import extract, standalone
    if args.effort:
        SETTINGS.effort = args.effort
    system = args.system
    if system == "rag" and args.variant:
        system = apply_variant(args.variant)        # A -> rag, B1 -> rag_b1, B2 -> rag_b2
    for nct in _ids(args):
        if system == "standalone":
            if extract.out_path(system, nct).exists() and not args.force:
                print(f"{nct}: exists, skipping (use --force)")
                continue
            print(f"  -> {extract.save(standalone.extract_protocol(nct))}")
            continue
        print(f"{nct}: {system}, extractor={SETTINGS.model('extractor')}, retrieval={SETTINGS.retrieval}, "
              f"rerank={SETTINGS.rerank or 'off'}, conditions={SETTINGS.conditional_values}")
        res, how = extract.run_resumable(nct, system, force=args.force)
        print(f"  {how}: {len(res.get('errors', {}))} element(s) with errors")


def cmd_fixup(args):
    """Run B3: check + fix-up agent on one system, apply the suggested fixes, save as another system."""
    from .agent.fixup import fixup
    from .replication.extract import load, save
    if args.effort:
        SETTINGS.effort = args.effort
    for nct in _ids(args):
        res = fixup(load(args.source, nct), args.to, args.mode)
        fx = res["fixup"]
        print(f"{nct}: {fx['n_findings']} findings, {fx['n_applied']} fixes applied, {fx['n_escalated']} escalated, "
              f"{fx['n_sponsor_queries']} sponsor queries -> {save(res)}")


def cmd_preflight(args):
    from . import preflight
    rows = preflight.run(args.nct, live=args.live, max_usd=args.max_usd)
    print(preflight.to_text(rows))
    if any(r[1] == preflight.FAIL for r in rows):
        sys.exit(1)


def cmd_pilot(args):
    from .pilot import run_pilot
    for nct in (args.ids.split(",") if args.ids else ["NCT02080260"]):
        s = run_pilot(nct, args.max_usd, baseline=not args.no_baseline, bench_llm=args.bench_llm, effort=args.effort)
        print((RESULTS / f"pilot_{nct}.md").read_text(encoding="utf-8"))
        print(f"\nSaved results/pilot_{nct}.md")


def cmd_compare(args):
    from .bench.compare import compare
    rep = compare(args.systems.split(","), args.baseline)
    print((RESULTS / "compare.md").read_text(encoding="utf-8"))


def cmd_judge_check(args):
    from .replication import judge_check
    if args.agreement:
        print(json.dumps(judge_check.agreement(args.agreement), indent=1))
    else:
        print(f"Score the 'human_score' column, then run with --agreement: "
              f"{judge_check.export_sample(args.system, _ids(args), args.n)}")


def cmd_annotate(args):
    from .replication import annotate
    if args.apply_reviews is not None:
        print(f"Applied {annotate.apply_reviews(args.apply_reviews)} reviewed rows")
        return
    ids = _ids(args)
    for nct in ids:
        print(f"{nct}: building hybrid ground truth")
        annotate.annotate_protocol(nct)
    print(f"Human review queue: {annotate.write_review_queue(ids)}")


def cmd_judge(args):
    from .replication import judge
    ids = _ids(args)
    for nct in ids:
        print(f"{nct}: judging {args.system}")
        judge.judge_protocol(args.system, nct)
    s = judge.summarise(args.system, ids)
    print(json.dumps(s, indent=1))
    (RESULTS / f"judge_{args.system}.json").write_text(json.dumps(s, indent=1), encoding="utf-8")


def cmd_check(args):
    from .agent import pipeline
    from .replication.extract import load
    for nct in _ids(args):
        res = load(args.system, nct)
        rep = pipeline.run(nct, res["extraction"], args.mode, {"registry_nct_id": nct, "nct_id": nct},
                           res.get("provenance"))
        print(f"{nct}: {len(rep['findings'])} findings -> {FINDINGS_OUT / nct / f'report_{args.mode}.md'}")


def cmd_resolve(args):
    from .agent import pipeline, resolve
    from .replication.extract import load
    for nct in _ids(args):
        res = load(args.system, nct)
        ext = res["extraction"]
        _, findings = pipeline.load_findings(nct, args.mode)
        if args.severity:
            findings = [f for f in findings if f.severity in args.severity.split(",")]
        props = resolve.resolve_all(nct, ext, findings, {"registry_nct_id": nct, "nct_id": nct,
                                                         "provenance": res.get("provenance", {})})
        print(f"{nct}: {len(props)} proposals -> {pipeline.write_proposals(nct, props)}")


def cmd_review(args):
    """Human review gate: approve / reject one proposal; approved fixes are written to a reviewed copy."""
    from .agent import pipeline, resolve
    from .replication.extract import load, out_path
    nct = args.nct_id
    path = FINDINGS_OUT / nct / "proposals.json"
    props = json.loads(path.read_text(encoding="utf-8"))
    if args.list:
        for p in props:
            print(f"{p['finding_id']} [{p['severity']}] {p['type']}: {p['action']} "
                  f"({p['proposal'].get('cause')}) review={p['review']['status']}")
        return
    reviewed = out_path("reviewed", nct)
    ext = json.loads(reviewed.read_text(encoding="utf-8")) if reviewed.exists() else load(args.system, nct)
    rec = resolve.review(nct, props, args.finding, args.decision, args.note or "", ext["extraction"])
    pipeline.write_proposals(nct, props)
    reviewed.parent.mkdir(parents=True, exist_ok=True)
    ext["system"] = "reviewed"
    reviewed.write_text(json.dumps(ext, indent=1), encoding="utf-8")
    print(f"{args.finding}: {rec['review']['status']} (audit: {FINDINGS_OUT / nct / 'audit.jsonl'})")


def cmd_bench(args):
    from .bench import run as bench
    if args.base == "fixture":
        paths = [FIXTURE]
    else:
        paths = sorted((EXTRACTIONS / args.base).glob("*.json"))
        if args.limit:
            paths = paths[:args.limit]
    if not paths:
        sys.exit(f"No base extractions in {EXTRACTIONS / args.base}. Run `consistency-agent extract` first or use --base fixture.")
    bases, reg = bench.load_bases(paths)
    bench.run(bases, args.modes.split(","), seed=args.seed, registry=reg, tag=args.tag or args.base,
              resolve=args.resolve)


def cmd_cost(args):
    totals = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0, "usd": 0.0})
    p = RUNS / "usage.jsonl"
    if not p.exists():
        print("No usage logged yet.")
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        t = totals[(r["role"], r["model"])]
        t["calls"] += 1
        t["input_tokens"] += r.get("input_tokens", 0)
        t["output_tokens"] += r.get("output_tokens", 0)
        t["seconds"] += r.get("seconds", 0)
        t["usd"] += r.get("usd_est") or 0
    for (role, model), t in sorted(totals.items()):
        print(f"{role:12s} {model:32s} calls={t['calls']:5d} in={t['input_tokens']:>10,} "
              f"out={t['output_tokens']:>9,} time={t['seconds'] / 60:.1f} min  ~${t['usd']:.2f}")
    print(f"Estimated total: ~${sum(t['usd'] for t in totals.values()):.2f} (list prices in llm.PRICES; check before relying on it)")


def cmd_models(args):
    for role in ("extractor", "soe_vision", "baseline", "judge", "annotator_a", "annotator_b", "adjudicator", "verifier"):
        print(f"{role:12s} {SETTINGS.model(role)}")
    print(f"embeddings   {SETTINGS.embeddings}")


def main(argv=None):
    ensure_dirs()
    ap = argparse.ArgumentParser(prog="consistency-agent", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_)
        p.set_defaults(fn=fn)
        return p

    def sel(p):
        p.add_argument("--ids", help="comma-separated NCT IDs (default: whole sample)")
        p.add_argument("--limit", type=int)

    p = add("sample", cmd_sample, "select the 23 protocols with the paper's design and download them")
    p.add_argument("--download-only", action="store_true", help="re-download PDFs listed in data/sample.json")
    p = add("extract", cmd_extract, "run the paper's RAG pipeline (Run A / B1 / B2) or the standalone baseline")
    p.add_argument("--system", choices=["rag", "standalone"], default="rag")
    p.add_argument("--variant", choices=["A", "B1", "B2"], default="A",
                   help="A = the paper; B1 = Opus 5.5; B2 = B1 + hybrid search, reranker, conditional values, citations")
    p.add_argument("--effort", choices=["low", "medium", "high"], help="thinking effort for Opus/Sonnet 5.x")
    p.add_argument("--force", action="store_true")
    sel(p)
    p = add("fixup", cmd_fixup, "Run B3: check + fix-up agent, apply suggested fixes to a copy")
    p.add_argument("--from", dest="source", default="rag_b2")
    p.add_argument("--to", default="rag_b3")
    p.add_argument("--mode", choices=["rules", "llm", "hybrid"], default="hybrid")
    p.add_argument("--effort", choices=["low", "medium", "high"])
    sel(p)
    p = add("preflight", cmd_preflight, "check keys, model names and a tiny live call per model before a run")
    p.add_argument("--nct", default="NCT02080260")
    p.add_argument("--live", action="store_true", help="also make one tiny call per model (about 10 cents)")
    p.add_argument("--max-usd", type=float, default=1.0)
    p = add("pilot", cmd_pilot, "one-command pilot on one protocol, with a spending cap")
    p.add_argument("--ids", help="NCT ID(s); default NCT02080260")
    p.add_argument("--max-usd", type=float, default=15.0, help="stop calling models once this is spent")
    p.add_argument("--no-baseline", action="store_true", help="skip Run A (the paper's pipeline)")
    p.add_argument("--bench-llm", action="store_true", help="also run the planted-contradiction benchmark with the LLM")
    p.add_argument("--effort", choices=["low", "medium", "high"], default="low")
    p = add("compare", cmd_compare, "paired comparison of systems scored by the judge")
    p.add_argument("--systems", default="rag,standalone,rag_b1,rag_b2,rag_b3")
    p.add_argument("--baseline", default="rag")
    p = add("judge-check", cmd_judge_check, "export a blind sample for a person to score, or compute agreement")
    p.add_argument("--system", default="rag")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--agreement", metavar="CSV")
    sel(p)
    p = add("annotate", cmd_annotate, "build the hybrid ground truth and the human review queue")
    p.add_argument("--apply-reviews", metavar="CSV", nargs="?", const="", default=None)
    sel(p)
    p = add("judge", cmd_judge, "score an extraction system with the LLM judge and Table 7 weights")
    p.add_argument("--system", default="rag")
    sel(p)
    p = add("check", cmd_check, "run the consistency checks on extractions")
    p.add_argument("--system", default="rag")
    p.add_argument("--mode", choices=["rules", "llm", "hybrid"], default="hybrid")
    sel(p)
    p = add("resolve", cmd_resolve, "propose fixes / sponsor queries for findings")
    p.add_argument("--system", default="rag")
    p.add_argument("--mode", default="hybrid")
    p.add_argument("--severity", help="e.g. critical,major")
    sel(p)
    p = add("review", cmd_review, "human review gate for one proposal")
    p.add_argument("nct_id")
    p.add_argument("--finding")
    p.add_argument("--decision", choices=["approve", "reject"])
    p.add_argument("--note")
    p.add_argument("--list", action="store_true")
    p.add_argument("--system", default="rag")
    p = add("bench", cmd_bench, "synthetic contradiction benchmark")
    p.add_argument("--base", default="rag", help="extraction system folder, or 'fixture' for the offline demo")
    p.add_argument("--modes", default="rules,llm,hybrid")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--tag")
    p.add_argument("--resolve", action="store_true", help="also score the fix-up loop (needs keys + PDFs)")
    add("cost", cmd_cost, "summarise logged token usage")
    add("models", cmd_models, "show the model assigned to each role")
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
