"""Append-only audit log (JSONL) of findings, proposals and review decisions."""
from __future__ import annotations

import getpass
import hashlib
import json
import time

from ..config import FINDINGS_OUT


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def record(nct_id: str, event: str, **data) -> dict:
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "nct_id": nct_id, "event": event,
           "actor": data.pop("actor", "agent"), **data}
    path = FINDINGS_OUT / nct_id / "audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def reviewer() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "reviewer"


def read(nct_id: str) -> list[dict]:
    path = FINDINGS_OUT / nct_id / "audit.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
