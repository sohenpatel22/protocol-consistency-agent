"""Paths, model choices and run settings.

Two model profiles: "paper" uses the models named in the paper, "current" uses newer
Claude models for extraction. Any role can be overridden with AGENT_MODEL_<ROLE>."""
from __future__ import annotations

import os
import re
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


def _via() -> str:
    """Where model calls go: "direct" (Anthropic/OpenAI SDKs) or "openrouter" (one key for everything).

    Auto: OpenRouter when only OPENROUTER_API_KEY is set. Override with AGENT_VIA."""
    env = os.getenv("AGENT_VIA")
    if env:
        return env
    direct = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    return "openrouter" if os.getenv("OPENROUTER_API_KEY") and not direct else "direct"


def to_openrouter(spec: str) -> str:
    """'anthropic:claude-opus-5-5' -> 'openrouter:anthropic/claude-opus-5.5' (OpenRouter's naming)."""
    provider, _, model = spec.partition(":")
    if provider not in ("anthropic", "openai"):
        return spec
    model = re.sub(r"-(\d)-(\d)$", lambda m: f"-{m.group(1)}.{m.group(2)}", model)   # sonnet-4-5 -> sonnet-4.5
    return f"openrouter:{provider}/{model}"


def _default_embeddings() -> str:
    if os.getenv("AGENT_EMBEDDINGS"):
        return os.environ["AGENT_EMBEDDINGS"]
    if os.getenv("COHERE_API_KEY"):
        return "cohere:embed-v4.0"
    if _via() == "openrouter":
        return "openrouter:cohere/embed-5-pro"      # the closest Cohere embedding OpenRouter offers
    return "bm25"


def default_rerank() -> str:
    if os.getenv("COHERE_API_KEY"):
        return "cohere:rerank-v3.5"
    return "openrouter:cohere/rerank-v3.5" if _via() == "openrouter" else ""


@dataclass
class Settings:
    profile: str = field(default_factory=lambda: os.getenv("AGENT_PROFILE", "paper"))
    via: str = field(default_factory=_via)
    embeddings: str = field(default_factory=_default_embeddings)
    top_k: int = 8                    # chunks retrieved per data element
    context_char_budget: int = 30_000  # max characters of retrieved context per call
    max_chunk_chars: int = 6_000      # sections longer than this are split (and reunited at retrieval)
    standalone_char_limit: int = 440_000  # ~110k tokens for gpt-4o's 128k window
    # ask the extractor for a page + quote per field (not in the paper; off for a faithful replication)
    field_citations: bool = field(default_factory=lambda: os.getenv("AGENT_FIELD_CITATIONS", "0") == "1")
    # ---- Run B upgrades (all off by default = the paper's setup) ----
    # "dense" = embeddings only (paper); "hybrid" = embeddings + BM25 keyword search, fused
    retrieval: str = field(default_factory=lambda: os.getenv("AGENT_RETRIEVAL", "dense"))
    # "<provider>:<model>" turns on a reranker: retrieve wide_k candidates, keep the best
    rerank: str = field(default_factory=lambda: os.getenv("AGENT_RERANK", ""))
    wide_k: int = 50
    # let a field carry several values with their conditions ("200 mg (Part A); 175 mg (Part B)")
    conditional_values: bool = field(default_factory=lambda: os.getenv("AGENT_CONDITIONAL_VALUES", "0") == "1")
    # thinking effort for models that support it (Opus/Sonnet 5.x): low | medium | high
    effort: str = field(default_factory=lambda: os.getenv("AGENT_EFFORT", ""))
    fix_rounds: int = 2               # fix-up agent: rounds before a person decides
    model_overrides: dict = field(default_factory=dict)

    def model(self, role: str) -> str:
        env = os.getenv(f"AGENT_MODEL_{role.upper()}")
        if env:
            return env
        if role in self.model_overrides:
            spec = self.model_overrides[role]
        else:
            try:
                spec = PROFILES[self.profile][role]
            except KeyError as exc:
                raise KeyError(f"Unknown profile/role: {self.profile}/{role}") from exc
        return to_openrouter(spec) if self.via == "openrouter" else spec


SETTINGS = Settings()

# Run A = the paper; Run B adds one thing at a time so every gain can be explained.
# Each variant writes its extractions to its own folder (data/extractions/<system>).
VARIANTS: dict[str, dict] = {
    "A": {"system": "rag", "settings": {}, "models": {}},
    "B1": {"system": "rag_b1", "settings": {},
           "models": {"extractor": "anthropic:claude-opus-5-5", "soe_vision": "anthropic:claude-opus-5-5"}},
    "B2": {"system": "rag_b2",
           "settings": {"retrieval": "hybrid", "rerank": "auto", "conditional_values": True,
                        "field_citations": True},
           "models": {"extractor": "anthropic:claude-opus-5-5", "soe_vision": "anthropic:claude-opus-5-5"}},
}


BASE_SETTINGS = {"retrieval": "dense", "rerank": "", "conditional_values": False, "field_citations": False}


def apply_variant(name: str) -> str:
    """Switch SETTINGS to a run variant; returns the extraction system name it writes to.

    Resets the upgrade settings first, so going from B2 back to A really is the paper's setup."""
    v = VARIANTS[name]
    for k, val in {**BASE_SETTINGS, **v["settings"]}.items():
        if k == "rerank" and val == "auto":
            val = default_rerank()                # "" when no reranker is available
        setattr(SETTINGS, k, val)
    SETTINGS.model_overrides = dict(v["models"])
    return v["system"]


def ensure_dirs() -> None:
    for p in (RAW, CACHE, EXTRACTIONS, GROUND_TRUTH, JUDGEMENTS, FINDINGS_OUT, BENCH, RESULTS, RUNS):
        p.mkdir(parents=True, exist_ok=True)
