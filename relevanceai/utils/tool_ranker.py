"""Hybrid tool-retrieval ranker.

Ranks a library of Relevance AI tools by relevance to a task query, so an agent
can load a small top-k slice on demand instead of putting the whole library in
context.

This is an *adapted port* (Mode 2) of the hybrid ranker from:

    "Comparative Approaches to Agent Retrieval over Large Skill Libraries"
    (arXiv:2608.06196)

The paper's ranker fuses **lexical** retrieval with **dense-embedding**
retrieval. This module keeps that fusion structure -- two complementary
retrieval signals, min-max normalised across the candidate set and combined with
a tunable weight -- at full fidelity, but substitutes the *auxiliary*
components with parameter-free, dependency-free proxies the SDK can host
without pulling in an embedding model or a vector store:

* **lexical** signal: BM25 (exact-term matching with IDF + document-length
  normalisation) -- the canonical parameter-free lexical retriever the paper's
  "lexical" component stands in for.
* **dense** signal: a parameter-free proxy of the paper's "local embedding
  pass" -- cosine similarity over smoothed TF-IDF vectors, which captures
  topical / term-co-occurrence adjacency without a learned encoder.
* the paper's bespoke optimiser, 690-skill corpus, and 117-query benchmark
  suite are intentionally out of scope (evaluation belongs downstream) -- but
  the paper's headline metric, ``hit@k``, is exposed so retrieval quality can
  be measured the same way.

Pure standard library -- no third-party dependencies.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "tokenize",
    "bm25_scores",
    "tfidf_cosine_scores",
    "hybrid_rank",
    "rank_tools",
    "hit_at_k",
    "mean_hit_at_k",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# BM25 defaults (Robertson & Zaragoza).
_BM25_K1 = 1.5
_BM25_B = 0.75


def tokenize(text: str) -> List[str]:
    """Lowercase ``text`` and split it into alphanumeric tokens."""
    return _TOKEN_RE.findall((text or "").lower())


def _term_counts(text: str) -> Counter:
    return Counter(tokenize(text))


def _document_frequency(docs_tf: Sequence[Counter]) -> Dict[str, int]:
    df: Dict[str, int] = {}
    for tf in docs_tf:
        for term in tf:
            df[term] = df.get(term, 0) + 1
    return df


def bm25_scores(query: str, documents: Sequence[str]) -> List[float]:
    """BM25 lexical relevance of each document against ``query``.

    Returns one score per document (higher is more relevant).
    """
    n_docs = len(documents)
    if n_docs == 0:
        return []

    query_terms = tokenize(query)
    docs_tf = [_term_counts(d) for d in documents]
    df = _document_frequency(docs_tf)

    avgdl = sum(sum(tf.values()) for tf in docs_tf) / n_docs or 1.0

    scores: List[float] = []
    for tf in docs_tf:
        doc_len = sum(tf.values())
        denom_len = _BM25_K1 * (1 - _BM25_B + _BM25_B * (doc_len / avgdl))
        score = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            idf = math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
            score += idf * (f * (_BM25_K1 + 1)) / (f + denom_len)
        scores.append(score)
    return scores


def tfidf_cosine_scores(query: str, documents: Sequence[str]) -> List[float]:
    """Parameter-free proxy of dense-embedding similarity.

    Cosine similarity between the query and each document over smoothed TF-IDF
    vectors. Stands in for the paper's "local embedding pass" without a learned
    encoder.
    """
    n_docs = len(documents)
    if n_docs == 0:
        return []

    docs_tf = [_term_counts(d) for d in documents]
    df = _document_frequency(docs_tf)
    # Smoothed IDF (sklearn-style), always strictly positive. Terms that appear
    # only in the query (df == 0) are weighted with the df=0 case of the same
    # formula, so a query can carry vocabulary absent from every document.
    unseen_idf = math.log(1.0 + n_docs) + 1.0
    idf = {
        term: math.log((1.0 + n_docs) / (1.0 + d)) + 1.0 for term, d in df.items()
    }

    def vectorize(tf: Counter) -> Dict[str, float]:
        return {term: count * idf.get(term, unseen_idf) for term, count in tf.items()}

    def norm(vec: Dict[str, float]) -> float:
        return math.sqrt(sum(v * v for v in vec.values()))

    q_vec = vectorize(_term_counts(query))
    q_norm = norm(q_vec) or 1.0

    scores: List[float] = []
    for tf in docs_tf:
        d_vec = vectorize(tf)
        d_norm = norm(d_vec)
        if d_norm == 0.0:
            scores.append(0.0)
            continue
        small, large = (q_vec, d_vec) if len(q_vec) <= len(d_vec) else (d_vec, q_vec)
        dot = sum(val * large.get(term, 0.0) for term, val in small.items())
        scores.append(dot / (q_norm * d_norm))
    return scores


def _minmax(values: Sequence[float]) -> List[float]:
    """Scale ``values`` into [0, 1]. A constant column maps to all zeros."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.0 for _ in values]
    span = hi - lo
    return [(v - lo) / span for v in values]


def hybrid_rank(
    query: str,
    documents: Sequence[str],
    top_k: Optional[int] = None,
    lexical_weight: float = 0.5,
    dense_weight: float = 0.5,
) -> List[Tuple[int, float]]:
    """Fuse the lexical and dense signals into a single ranking.

    Each signal is min-max normalised across the candidate set, then combined
    with ``lexical_weight`` / ``dense_weight``. Returns ``(original_index,
    score)`` pairs sorted by score descending, truncated to ``top_k``.
    """
    if not documents:
        return []
    total = lexical_weight + dense_weight
    if total <= 0:
        raise ValueError("lexical_weight + dense_weight must be positive")

    w_lex = lexical_weight / total
    w_dense = dense_weight / total
    lex = _minmax(bm25_scores(query, documents))
    dense = _minmax(tfidf_cosine_scores(query, documents))
    combined = [w_lex * l + w_dense * d for l, d in zip(lex, dense)]

    ranked = sorted(enumerate(combined), key=lambda kv: kv[1], reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
    return ranked


def _tool_text(tool) -> str:
    """Concatenate the searchable text fields of a Relevance AI ``Tool``."""
    metadata = getattr(tool, "metadata", None)
    parts = [
        getattr(metadata, "title", None),
        getattr(metadata, "description", None),
        getattr(metadata, "prompt_description", None),
    ]
    return " ".join(p for p in parts if p)


def rank_tools(
    query: str,
    tools: Sequence,
    top_k: int = 5,
    lexical_weight: float = 0.5,
    dense_weight: float = 0.5,
) -> List:
    """Rank Relevance AI ``Tool`` objects by relevance to ``query``.

    Thin adapter over :func:`hybrid_rank`: scores each tool against its
    title + description (+ prompt_description) text and returns the top-k
    ``Tool`` objects in ranked order. This is the on-demand, sparse-loading
    retrieval surface the paper recommends over loading the whole library.
    """
    if not tools:
        return []
    documents = [_tool_text(t) for t in tools]
    ranked = hybrid_rank(
        query,
        documents,
        top_k=top_k,
        lexical_weight=lexical_weight,
        dense_weight=dense_weight,
    )
    return [tools[idx] for idx, _ in ranked]


def hit_at_k(ranked_ids: Sequence[str], gold_id: str, k: int = 5) -> int:
    """1 if ``gold_id`` appears in the top-k of ``ranked_ids``, else 0.

    The paper's headline retrieval metric (hit@5); exposed so retrieval quality
    is measurable the same way.
    """
    return 1 if gold_id in list(ranked_ids)[:k] else 0


def mean_hit_at_k(
    ranked_ids_per_query: Iterable[Sequence[str]],
    gold_ids: Sequence[str],
    k: int = 5,
) -> float:
    """Mean hit@k across a set of (ranked_ids, gold_id) pairs."""
    gold_ids = list(gold_ids)
    rows = list(ranked_ids_per_query)
    if not rows:
        return 0.0
    return sum(hit_at_k(r, g, k) for r, g in zip(rows, gold_ids)) / len(rows)
