"""Pre-flight checks: find setup problems before any real money is spent.

Free checks (no model calls): key and credit balance, that every model name exists, the protocol PDF,
and an offline run of the whole checker. Tiny live checks (about 10 cents in total): one small call
per model the run will use, a picture call for the schedule-of-events model, the thinking-effort
setting, embeddings and reranking.

    consistency-agent preflight            # free checks only
    consistency-agent preflight --live     # plus the tiny live calls
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from . import llm
from .config import RAW, SETTINGS, apply_variant, default_rerank, ensure_dirs

OK, WARN, FAIL = "ok", "warn", "FAIL"


def _roles_used(variants: list[str]) -> dict[str, list[str]]:
    """model id -> roles that use it, for the given run variants."""
    used: dict[str, list[str]] = {}
    for v in variants:
        apply_variant(v)
        roles = ["extractor", "soe_vision", "verifier"]
        if v == variants[0]:
            roles += ["baseline", "judge", "annotator_a", "annotator_b", "adjudicator"]
        for r in roles:
            used.setdefault(SETTINGS.model(r), [])
            if f"{v}:{r}" not in used[SETTINGS.model(r)]:
                used[SETTINGS.model(r)].append(f"{v}:{r}")
    apply_variant("A")
    return used


def _png_with_text(text: str) -> Path:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (360, 120), "white")
    ImageDraw.Draw(img).text((20, 40), text, fill="black")
    img = img.resize((720, 240))
    path = Path(tempfile.gettempdir()) / "preflight_probe.png"
    img.save(path)
    return path


def run(nct: str | None = None, live: bool = False, variants: tuple[str, ...] = ("A", "B2"),
        max_usd: float = 1.0) -> list[tuple[str, str, str]]:
    """Returns (check, status, detail) rows. Any FAIL means: do not start the run."""
    ensure_dirs()
    rows: list[tuple[str, str, str]] = []

    def add(name, status, detail=""):
        rows.append((name, status, detail))

    via = SETTINGS.via
    add("route", OK, f"model calls go {'through OpenRouter' if via == 'openrouter' else 'directly to Anthropic/OpenAI'}")
    used = _roles_used(list(variants))

    # ---- key, balance, model names (free) ----
    if via == "openrouter":
        import requests
        key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not key:
            add("OpenRouter key", FAIL, "OPENROUTER_API_KEY is empty (add it to .env)")
            return rows
        try:
            r = requests.get(llm.OPENROUTER_URL + "/key", headers={"Authorization": f"Bearer {key}"}, timeout=30)
            if r.status_code != 200:
                add("OpenRouter key", FAIL, f"the key was rejected (HTTP {r.status_code})")
                return rows
            d = r.json()["data"]
            left = d.get("limit_remaining")
            add("OpenRouter key", OK if left is None or left >= max_usd else WARN,
                f"valid; spending limit ${d.get('limit')}, remaining ${left}, used ${d.get('usage')}")
            ids = {m["id"] for m in requests.get(llm.OPENROUTER_URL + "/models", timeout=30).json()["data"]}
            missing = [m.split(":", 1)[1] for m in used if m.split(":", 1)[1] not in ids]
            add("model names", FAIL if missing else OK,
                f"not found on OpenRouter: {missing}" if missing else f"all {len(used)} models exist")
        except requests.RequestException as exc:
            add("OpenRouter reachable", FAIL, f"{type(exc).__name__}: {exc}")
            return rows
    else:
        for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            add(env, OK if os.getenv(env) else WARN, "set" if os.getenv(env) else "empty")

    # ---- protocol and parsing (free) ----
    if nct:
        pdf = RAW / f"{nct}.pdf"
        if not pdf.exists():
            add("protocol PDF", FAIL, f"{pdf} is missing (run `consistency-agent sample --download-only`)")
        else:
            from .replication.pdfparse import parse_pdf
            from .replication.soe import candidate_pages
            chunks = parse_pdf(pdf)
            pages = candidate_pages(pdf)
            add("protocol PDF", OK if chunks else FAIL, f"{len(chunks)} chunks; schedule-of-events pages found: {[p + 1 for p in pages]}")
            if not pages:
                add("schedule of events", WARN, "no schedule-of-events page was found; that section will be empty")

    # ---- offline run of the whole checker (free) ----
    try:
        fx = json.loads((Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "synthetic_extraction.json")
                        .read_text(encoding="utf-8"))["extraction"]
        from .agent.pipeline import check
        from .bench.inject import generate
        n = len(generate(fx, "pf"))
        findings, _ = check(fx, "rules", {"registry_nct_id": "NCT09999999"}, log=lambda *_: None)
        add("checker (offline)", OK if not findings else WARN, f"{n} planted cases generated; base findings: {len(findings)}")
    except Exception as exc:                                   # noqa: BLE001
        add("checker (offline)", FAIL, f"{type(exc).__name__}: {exc}")

    if not live:
        add("live calls", WARN, "skipped (add --live to test each model with a tiny call)")
        return rows

    # ---- tiny live calls ----
    os.environ["AGENT_MAX_USD"] = str(max_usd)
    SETTINGS.effort = "low"
    sys_prompt = "Reply with a JSON object only."
    for model, roles in used.items():
        spec_role = roles[0].split(":", 1)[1]
        before = llm.spent()
        try:
            text = llm._dispatch(*model.split(":", 1), llm.Call(spec_role, sys_prompt, 'Reply with {"ok": true, "n": 7}',
                                                                  None, 1500))
            data = llm.parse_json(text)
            good = isinstance(data, dict) and data.get("n") == 7
            add(f"call {model.split(':', 1)[1]}", OK if good else FAIL,
                f"{'JSON round trip ok' if good else 'unexpected reply: ' + text[:80]}; cost ${llm.spent() - before:.4f}; used by {', '.join(roles)}")
        except Exception as exc:                               # noqa: BLE001
            add(f"call {model.split(':', 1)[1]}", FAIL, f"{type(exc).__name__}: {str(exc)[:200]}")

    # picture call for the schedule-of-events models
    probe = _png_with_text("VISIT 42")
    for model in {SETTINGS.model("soe_vision"), "openrouter:anthropic/claude-opus-5.5" if via == "openrouter" else "anthropic:claude-opus-5-5"}:
        before = llm.spent()
        try:
            text = llm._dispatch(*model.split(":", 1), llm.Call("soe_vision", sys_prompt,
                                                                  'What number is in the image? Reply {"n": <number>}',
                                                                  [probe], 1500))
            ok = isinstance(llm.parse_json(text), dict) and str(llm.parse_json(text).get("n")) == "42"
            add(f"picture call {model.split(':', 1)[1]}", OK if ok else WARN,
                f"{'read the image correctly' if ok else 'reply: ' + text[:80]}; cost ${llm.spent() - before:.4f}")
        except Exception as exc:                               # noqa: BLE001
            add(f"picture call {model.split(':', 1)[1]}", FAIL, f"{type(exc).__name__}: {str(exc)[:200]}")

    # embeddings and reranking
    try:
        from .replication.retrieval import _embed, _rerank_call
        emb = SETTINGS.embeddings
        if emb != "bm25":
            prov, _, m = emb.partition(":")
            v = _embed(prov, ["store at 2-8 C"], m, "search_query")
            add("embeddings", OK, f"{emb}: {len(v[0])} dimensions")
        else:
            add("embeddings", WARN, "no embedding model available; retrieval falls back to keyword search (BM25)")
        rr = SETTINGS.rerank or default_rerank()
        if rr:
            prov, _, m = rr.partition(":")
            order = _rerank_call(prov, "storage temperature", ["age 18 or older", "store at 2-8 C"], m)
            add("reranker", OK if order[0] == 1 else WARN, f"{rr}: order {order}")
        else:
            add("reranker", WARN, "no reranker available; Run B2 will run without it")
    except Exception as exc:                                   # noqa: BLE001
        add("embeddings/reranker", FAIL, f"{type(exc).__name__}: {str(exc)[:200]}")

    add("live cost", OK, f"~${llm.spent():.4f} spent on checks")
    return rows


def to_text(rows) -> str:
    w = max(len(r[0]) for r in rows)
    lines = [f"{r[1].upper():5s} {r[0]:{w}s}  {r[2]}" for r in rows]
    bad = sum(r[1] == FAIL for r in rows)
    lines.append("")
    lines.append("READY" if not bad else f"NOT READY: {bad} check(s) failed")
    return "\n".join(lines)
