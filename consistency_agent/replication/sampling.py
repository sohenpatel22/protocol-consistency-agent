"""Pick and download the 23 protocols the same way the paper did.

The paper (Section 3.1 and Listing 1) keeps ClinicalTrials.gov studies that have a protocol
document, whose first intervention is a drug, whose primary purpose is treatment and that
run in Canada or the United States. It labels each study oncology, cardiology or other with
two keyword lists on the primary condition, samples 9 / 7 / 7, and drops scanned PDFs.

The NCT IDs themselves were not published, so we repeat the method with a fixed seed.
The keyword lists below are copied from Listing 1, quirks included ("atrial" sits in the
oncology list, "tumor " has a trailing space, and the cardiology list is applied second
so it wins a tie).
"""
from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import pymupdf as fitz
import requests

from ..config import RAW, SAMPLE_FILE, SAMPLE_SEED, SNAPSHOT_DATE, STRATA

API = "https://clinicaltrials.gov/api/v2/studies"
CDN = "https://cdn.clinicaltrials.gov/large-docs/{suffix}/{nct}/{filename}"
UA = {"User-Agent": "protocol-consistency-agent/0.1 (research replication)"}

FILTER = (
    "AREA[StudyType]INTERVENTIONAL AND AREA[DesignPrimaryPurpose]TREATMENT "
    "AND AREA[InterventionType]DRUG AND AREA[LargeDocHasProtocol]true "
    'AND (AREA[LocationCountry]"United States" OR AREA[LocationCountry]Canada)'
)
FIELDS = "NCTId,BriefTitle,Condition,Phase,LargeDoc,InterventionType"

# Listing 1 of the paper, verbatim
ONCOLOGY = re.compile("cancer|carcinoma|neoplasm|myeloma|lymphoma|leukemia|tumor |melanoma|"
                      "neuroblastoma|glioblastoma|sarcoma|atrial", re.I)
CARDIO = re.compile("heart|cardiac|cardial|stroke|artery|coronary|cardio|hypertension", re.I)


def classify(primary_condition: str) -> str:
    text = (primary_condition or "").lower()
    area = "other"
    if ONCOLOGY.search(text):
        area = "oncology"
    if CARDIO.search(text):
        area = "cardiovascular"
    return area


def fetch_candidates(page_size: int = 1000, max_pages: int = 40) -> list[dict]:
    """All studies matching the paper's filters, with their protocol document metadata."""
    out: list[dict] = []
    token = None
    for _ in range(max_pages):
        params = {"format": "json", "pageSize": page_size, "filter.advanced": FILTER, "fields": FIELDS}
        if token:
            params["pageToken"] = token
        r = requests.get(API, params=params, headers=UA, timeout=60)
        r.raise_for_status()
        payload = r.json()
        for s in payload.get("studies", []):
            rec = _candidate(s)
            if rec:
                out.append(rec)
        token = payload.get("nextPageToken")
        if not token:
            break
        time.sleep(0.3)
    return out


def _candidate(study: dict) -> dict | None:
    ps = study.get("protocolSection", {})
    nct = ps.get("identificationModule", {}).get("nctId")
    docs = study.get("documentSection", {}).get("largeDocumentModule", {}).get("largeDocs", [])
    prot = [d for d in docs if d.get("hasProtocol") and d.get("filename")
            and (d.get("uploadDate") or "9999")[:10] <= SNAPSHOT_DATE]
    if not nct or not prot:
        return None
    prot.sort(key=lambda d: d.get("uploadDate", ""))
    doc = prot[-1]                          # latest protocol version available at the snapshot
    conditions = ps.get("conditionsModule", {}).get("conditions", [])
    title = ps.get("identificationModule", {}).get("briefTitle", "")
    interventions = ps.get("armsInterventionsModule", {}).get("interventions", [])
    if not interventions or interventions[0].get("type") != "DRUG":    # Listing 1: first intervention is a drug
        return None
    icf = [d for d in docs if d.get("hasIcf") and d.get("filename")]
    return {
        "nct_id": nct,
        "title": title,
        "conditions": conditions,
        "phases": ps.get("designModule", {}).get("phases", []),
        "area": classify(conditions[0] if conditions else ""),
        "protocol_file": doc["filename"],
        "protocol_upload_date": doc.get("uploadDate"),
        "protocol_url": CDN.format(suffix=nct[-2:], nct=nct, filename=doc["filename"]),
        "icf_url": CDN.format(suffix=nct[-2:], nct=nct, filename=icf[0]["filename"]) if icf else None,
    }


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with requests.get(url, headers=UA, timeout=120, stream=True) as r:
        r.raise_for_status()
        if "pdf" not in r.headers.get("content-type", ""):
            raise ValueError(f"Not a PDF: {url}")
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
    return dest


STOPWORDS = {"the", "and", "of", "to", "in", "a", "for", "be", "is", "or", "with", "will", "study",
             "patients", "on", "by", "at", "are", "as"}


def text_layer_stats(pdf: Path, sample_pages: int = 30) -> dict:
    """Characters per page and share of common English words over the first pages.

    Scanned PDFs have (almost) no text layer; some PDFs have a text layer that does not decode
    (font-shifted glyphs such as '5HIHUHQFHV' for 'References'). Both are excluded, following
    the paper's exclusion of documents that appear scanned.
    """
    with fitz.open(pdf) as doc:
        n = doc.page_count
        texts = [doc[i].get_text("text") for i in range(min(n, sample_pages))]
    chars = [len(t.strip()) for t in texts]
    toks = re.findall(r"[A-Za-z]+", " ".join(texts).lower())
    return {"pages": n, "mean_chars_per_page": round(sum(chars) / max(len(chars), 1), 1),
            "stopword_ratio": round(sum(t in STOPWORDS for t in toks) / max(len(toks), 1), 3)}


def is_scanned(stats: dict, min_chars: int = 400, min_stopword_ratio: float = 0.05) -> bool:
    return stats["mean_chars_per_page"] < min_chars or stats.get("stopword_ratio", 1) < min_stopword_ratio


def select_sample(candidates: list[dict], seed: int = SAMPLE_SEED, max_pages: int = 400,
                  log=print) -> list[dict]:
    """Random stratified sample; candidates are downloaded and checked until each quota is met."""
    rng = random.Random(seed)
    chosen: list[dict] = []
    for area, quota in STRATA.items():
        pool = sorted((c for c in candidates if c["area"] == area), key=lambda c: c["nct_id"])
        rng.shuffle(pool)
        got = 0
        for cand in pool:
            if got == quota:
                break
            pdf = RAW / f"{cand['nct_id']}.pdf"
            try:
                download(cand["protocol_url"], pdf)
                stats = text_layer_stats(pdf)
            except Exception as exc:               # network / malformed PDF: skip, like the paper's filtering
                log(f"  skip {cand['nct_id']}: {exc}")
                continue
            if is_scanned(stats) or stats["pages"] > max_pages:
                log(f"  skip {cand['nct_id']}: scanned, unreadable or too long ({stats})")
                continue
            chosen.append({**cand, **stats})
            got += 1
            log(f"  {area}: {cand['nct_id']} ({stats['pages']} pages)")
        if got < quota:
            log(f"WARNING: only {got}/{quota} usable {area} protocols")
    return chosen


def run(log=print) -> list[dict]:
    log("Querying ClinicalTrials.gov with the paper's filters...")
    cands = fetch_candidates()
    by_area = {a: sum(c["area"] == a for c in cands) for a in STRATA}
    log(f"{len(cands)} candidates with a protocol uploaded on or before {SNAPSHOT_DATE}: {by_area}")
    sample = select_sample(cands, log=log)
    SAMPLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    meta = {"snapshot_date": SNAPSHOT_DATE, "seed": SAMPLE_SEED, "strata": STRATA,
            "n_candidates": len(cands), "candidates_by_area": by_area, "studies": sample}
    SAMPLE_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"Wrote {SAMPLE_FILE} ({len(sample)} protocols)")
    return sample


def load_sample() -> list[dict]:
    if not SAMPLE_FILE.exists():
        raise FileNotFoundError("No sample yet. Run: consistency-agent sample")
    return json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))["studies"]


def ensure_pdfs(studies: list[dict]) -> None:
    for s in studies:
        download(s["protocol_url"], RAW / f"{s['nct_id']}.pdf")
