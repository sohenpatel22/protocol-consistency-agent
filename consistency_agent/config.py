"""Paths, model choices and run settings.

Two model profiles: "paper" uses the models named in the paper, "current" uses newer
Claude models for extraction. Any role can be overridden with AGENT_MODEL_<ROLE>."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
RAW = DATA / "raw"                    # downloaded protocol PDFs (git-ignored)
CACHE = DATA / "cache"                # parsed chunks, embeddings (git-ignored)
EXTRACTIONS = DATA / "extractions"    # extraction outputs per system
GROUND_TRUTH = DATA / "ground_truth"  # hybrid AI/human annotations
JUDGEMENTS = DATA / "judgements"      # LLM-as-judge scores
FINDINGS_OUT = DATA / "findings"        # consistency findings, proposals, audit log
BENCH = DATA / "bench"                # injected benchmark documents
RESULTS = ROOT / "results"            # committed result summaries
RUNS = ROOT / "runs"                  # usage/cost logs (git-ignored)
SAMPLE_FILE = DATA / "sample.json"    # the 23 selected protocols (committed)

# Paper sampling design (Section 3.1 of the paper).
SNAPSHOT_DATE = "2025-03-31"          # "studies listed ... at the end of March 2025"
STRATA = {"oncology": 9, "cardiovascular": 7, "other": 7}
SAMPLE_SEED = 20250331

PROFILES: dict[str, dict[str, str]] = {
    "paper": {
        # RAG 1 in the paper: Claude Sonnet 4.5 (text), Claude Opus 4.5 (schedule of events)
        "extractor": "anthropic:claude-sonnet-4-5",
        "soe_vision": "anthropic:claude-opus-4-5",
        # Standalone baseline and LLM-as-judge: gpt-4o
        "baseline": "openai:gpt-4o",
        "judge": "openai:gpt-4o",
        # Ground truth: candidates from gpt-4o and "Claude 3" (retired; nearest
        # small Claude model used instead), adjudicated by gpt-5.
        "annotator_a": "openai:gpt-4o",
        "annotator_b": "anthropic:claude-haiku-4-5",
        "adjudicator": "openai:gpt-5",
        # The consistency checker is new work (not in the paper): current Claude model.
        "verifier": "anthropic:claude-opus-5-5",
    },
    "current": {
        "extractor": "anthropic:claude-opus-5-5",
        "soe_vision": "anthropic:claude-opus-5-5",
        "baseline": "openai:gpt-4o",
        "judge": "openai:gpt-4o",
        "annotator_a": "openai:gpt-4o",
        "annotator_b": "anthropic:claude-haiku-5-5",
        "adjudicator": "openai:gpt-5",
        "verifier": "anthropic:claude-opus-5-5",
    },
}


@dataclass
class Settings:
    profile: str = field(default_factory=lambda: os.getenv("AGENT_PROFILE", "paper"))
    embeddings: str = field(
        default_factory=lambda: os.getenv(
            "AGENT_EMBEDDINGS", "cohere:embed-v4.0" if os.getenv("COHERE_API_KEY") else "bm25"
        )
    )
    top_k: int = 8                    # chunks retrieved per data element
    context_char_budget: int = 30_000  # max characters of retrieved context per call
    max_chunk_chars: int = 6_000      # sections longer than this are split (and reunited at retrieval)
    standalone_char_limit: int = 440_000  # ~110k tokens for gpt-4o's 128k window
    # ask the extractor for a page + quote per field (not in the paper; off for a faithful replication)
    field_citations: bool = field(default_factory=lambda: os.getenv("AGENT_FIELD_CITATIONS", "0") == "1")

    def model(self, role: str) -> str:
        env = os.getenv(f"AGENT_MODEL_{role.upper()}")
        if env:
            return env
        try:
            return PROFILES[self.profile][role]
        except KeyError as exc:
            raise KeyError(f"Unknown profile/role: {self.profile}/{role}") from exc


SETTINGS = Settings()


def ensure_dirs() -> None:
    for p in (RAW, CACHE, EXTRACTIONS, GROUND_TRUTH, JUDGEMENTS, FINDINGS_OUT, BENCH, RESULTS, RUNS):
        p.mkdir(parents=True, exist_ok=True)
