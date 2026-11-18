"""Find the chunks for one data element: score, take the top k, bring back the other
parts of split sections, and put everything back in page order.

Uses Cohere embeddings like the paper when COHERE_API_KEY is set, BM25 otherwise."""
from __future__ import annotations

import math
import os
import re
import time
from collections import Counter

import numpy as np

from ..config import CACHE, SETTINGS
from .pdfparse import Chunk

TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def _tok(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.toks = [_tok(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg = sum(map(len, self.toks)) / max(len(self.toks), 1)
        df = Counter(t for doc in self.toks for t in set(doc))
        n = len(self.toks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(doc) for doc in self.toks]

    def scores(self, query: str) -> np.ndarray:
        q = _tok(query)
        out = np.zeros(len(self.toks))
        for i, (tf, doc) in enumerate(zip(self.tf, self.toks)):
            norm = self.k1 * (1 - self.b + self.b * len(doc) / max(self.avg, 1))
            out[i] = sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in q if t in tf)
        return out


def _chunk_text(c: Chunk) -> str:
    return f"{c.header()}\n{c.text}"


class Retriever:
    def __init__(self, nct_id: str, chunks: list[Chunk], backend: str | None = None):
        self.nct_id, self.chunks = nct_id, chunks
        self.backend = backend or SETTINGS.embeddings
        self._bm25: BM25 | None = None
        self._emb: np.ndarray | None = None
        self._qcache: dict[str, np.ndarray] = {}

    # --- scoring -----------------------------------------------------------------------
    def _scores(self, query: str) -> np.ndarray:
        if self.backend == "bm25":
            if self._bm25 is None:
                self._bm25 = BM25([_chunk_text(c) for c in self.chunks])
            return self._bm25.scores(query)
        provider, _, model = self.backend.partition(":")
        if provider not in ("cohere", "openrouter"):
            raise ValueError(f"Unknown embeddings backend {self.backend}")
        if self._emb is None:
            self._emb = self._doc_embeddings(provider, model)
        if query not in self._qcache:                       # the same query is asked again in later rounds
            self._qcache[query] = np.array(_embed(provider, [query], model, "search_query")[0])
        q = self._qcache[query]
        return self._emb @ (q / (np.linalg.norm(q) + 1e-9))

    def _doc_embeddings(self, provider: str, model: str) -> np.ndarray:
        cache = CACHE / self.nct_id / f"emb_{model.replace('/', '_')}.npy"
        if cache.exists():
            arr = np.load(cache)
            if len(arr) == len(self.chunks):
                return arr
        texts = [_chunk_text(c)[:20000] for c in self.chunks]
        vecs = []
        for i in range(0, len(texts), 32):
            vecs.extend(_embed(provider, texts[i:i + 32], model, "search_document"))
        arr = np.array(vecs, dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, arr)
        return arr

    def _keyword_scores(self, query: str) -> np.ndarray:
        if self._bm25 is None:
            self._bm25 = BM25([_chunk_text(c) for c in self.chunks])
        return self._bm25.scores(query)

    def rank(self, queries: list[str], hybrid: bool | None = None) -> list[int]:
        """Chunk indices, best first. Hybrid fuses embedding and keyword rankings (reciprocal rank fusion)."""
        hybrid = SETTINGS.retrieval == "hybrid" if hybrid is None else hybrid
        dense = np.max(np.vstack([self._scores(q) for q in queries]), axis=0)
        if not hybrid or self.backend == "bm25":
            return list(np.argsort(-dense))
        keyword = np.max(np.vstack([self._keyword_scores(q) for q in queries]), axis=0)
        fused = np.zeros(len(self.chunks))
        for s in (dense, keyword):
            for rank_pos, idx in enumerate(np.argsort(-s)):
                fused[idx] += 1.0 / (60 + rank_pos)          # standard RRF constant
        return list(np.argsort(-fused))

    # --- retrieval ---------------------------------------------------------------------
    def retrieve(self, queries: list[str], top_k: int | None = None,
                 char_budget: int | None = None, wide: bool = False) -> list[Chunk]:
        """Rank -> (optional) rerank a wide candidate set -> top-k -> reunite groups -> reading order.

        ``wide=True`` forces the widen + rerank path (used by the fix-up agent's second round)."""
        top_k = top_k or SETTINGS.top_k
        char_budget = char_budget or SETTINGS.context_char_budget
        if not self.chunks:
            return []
        ranked = self.rank(queries, hybrid=True if wide else None)
        if SETTINGS.rerank:
            ranked = self._rerank(queries, ranked[:SETTINGS.wide_k]) + ranked[SETTINGS.wide_k:]
        picked: list[int] = []
        used = 0
        groups = {}
        for i, c in enumerate(self.chunks):
            groups.setdefault(c.group, []).append(i)
        for idx in ranked:
            if len(picked) >= top_k and picked:
                break
            if idx in picked:
                continue
            # chunk reunion: bring back every part of the section this chunk came from
            family = [j for j in groups[self.chunks[idx].group] if j not in picked]
            size = sum(len(self.chunks[j].text) for j in family)
            if used + size > char_budget:
                family = [idx]
                size = len(self.chunks[idx].text)
                if used + size > char_budget:
                    continue
            picked.extend(family)
            used += size
        return [self.chunks[i] for i in sorted(picked, key=lambda j: self.chunks[j].order)]

    def _rerank(self, queries: list[str], candidates: list[int]) -> list[int]:
        provider, _, model = SETTINGS.rerank.partition(":")
        if provider not in ("cohere", "openrouter") or not candidates:
            return candidates
        docs = [_chunk_text(self.chunks[i])[:4000] for i in candidates]
        order = _rerank_call(provider, " | ".join(queries)[:2000], docs, model)
        return [candidates[i] for i in order]

    def pages(self, pages: list[int], neighbours: int = 1) -> list[Chunk]:
        """Open pages directly: every chunk that touches the given pages (plus neighbours)."""
        want = {p + d for p in pages for d in range(-neighbours, neighbours + 1)}
        hit = [c for c in self.chunks if any(p in want for p in range(c.page_start, c.page_end + 1))]
        return sorted(hit, key=lambda c: c.order)


def _openrouter_post(path: str, body: dict) -> dict:
    """POST to OpenRouter (embeddings, rerank). Failures become LLMError so one bad call is handled per field."""
    import requests

    from ..llm import OPENROUTER_URL, LLMError
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise LLMError("OPENROUTER_API_KEY is not set (add it to .env)")
    last = ""
    for attempt in range(4):
        try:
            r = requests.post(OPENROUTER_URL + path, json=body, timeout=120,
                              headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        except requests.RequestException as exc:
            last = f"{type(exc).__name__}: {exc}"
        else:
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code not in (408, 429, 500, 502, 503, 504):
                break
        time.sleep(2 ** attempt)
    raise LLMError(f"OpenRouter {path} failed: {last}")


def _rerank_call(provider: str, query: str, docs: list[str], model: str) -> list[int]:
    """Document indices, most relevant first."""
    if provider == "openrouter":
        data = _openrouter_post("/rerank", {"model": model, "query": query, "documents": docs, "top_n": len(docs)})
        return [r["index"] for r in data["results"]]
    import cohere  # optional dependency: pip install .[cohere]

    resp = cohere.ClientV2().rerank(model=model, query=query, documents=docs, top_n=len(docs))
    return [r.index for r in resp.results]


def _embed(provider: str, texts: list[str], model: str, input_type: str) -> list[list[float]]:
    if provider == "openrouter":
        data = _openrouter_post("/embeddings", {"model": model, "input": texts, "input_type": input_type})
        return [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]
    return _cohere_embed(texts, model, input_type)


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.header()} | pages {c.page_start}-{c.page_end}]\n{c.text}" for c in chunks)


def _cohere_embed(texts: list[str], model: str, input_type: str) -> list[list[float]]:
    import cohere  # optional dependency: pip install .[cohere]

    client = cohere.ClientV2()
    resp = client.embed(texts=texts, model=model, input_type=input_type, embedding_types=["float"])
    return [list(v) for v in resp.embeddings.float_]
