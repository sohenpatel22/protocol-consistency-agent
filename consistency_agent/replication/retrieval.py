"""Find the chunks for one data element: score, take the top k, bring back the other
parts of split sections, and put everything back in page order.

Uses Cohere embeddings like the paper when COHERE_API_KEY is set, BM25 otherwise."""
from __future__ import annotations

import math
import re
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

    # --- scoring -----------------------------------------------------------------------
    def _scores(self, query: str) -> np.ndarray:
        if self.backend == "bm25":
            if self._bm25 is None:
                self._bm25 = BM25([_chunk_text(c) for c in self.chunks])
            return self._bm25.scores(query)
        provider, _, model = self.backend.partition(":")
        if provider != "cohere":
            raise ValueError(f"Unknown embeddings backend {self.backend}")
        if self._emb is None:
            self._emb = self._doc_embeddings(model)
        q = np.array(_cohere_embed([query], model, "search_query")[0])
        return self._emb @ (q / (np.linalg.norm(q) + 1e-9))

    def _doc_embeddings(self, model: str) -> np.ndarray:
        cache = CACHE / self.nct_id / f"emb_{model.replace('/', '_')}.npy"
        if cache.exists():
            arr = np.load(cache)
            if len(arr) == len(self.chunks):
                return arr
        texts = [_chunk_text(c)[:20000] for c in self.chunks]
        vecs = []
        for i in range(0, len(texts), 64):
            vecs.extend(_cohere_embed(texts[i:i + 64], model, "search_document"))
        arr = np.array(vecs, dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, arr)
        return arr

    # --- retrieval ---------------------------------------------------------------------
    def retrieve(self, queries: list[str], top_k: int | None = None,
                 char_budget: int | None = None) -> list[Chunk]:
        """Max score over the element's queries -> top-k -> reunite groups -> reading order."""
        top_k = top_k or SETTINGS.top_k
        char_budget = char_budget or SETTINGS.context_char_budget
        if not self.chunks:
            return []
        scores = np.max(np.vstack([self._scores(q) for q in queries]), axis=0)
        ranked = list(np.argsort(-scores))
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


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.header()} | pages {c.page_start}-{c.page_end}]\n{c.text}" for c in chunks)


def _cohere_embed(texts: list[str], model: str, input_type: str) -> list[list[float]]:
    import cohere  # optional dependency: pip install .[cohere]

    client = cohere.ClientV2()
    resp = client.embed(texts=texts, model=model, input_type=input_type, embedding_types=["float"])
    return [list(v) for v in resp.embeddings.float_]
