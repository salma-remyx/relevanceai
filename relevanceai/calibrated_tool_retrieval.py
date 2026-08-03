"""Calibrated retrieval over a tool/skill library.

Tool descriptions in a large library share a regular "background" of generic
descriptive phrasing ("this tool takes ... and returns ..."). That shared
background inflates ordinary similarity scores and buries the few tokens that
actually distinguish one capability from the next, so a query for the right
tool gets out-scored by whichever description happens to echo the background.

This module retrieves the right tool by calibrating that background, adapting
the two training-free mechanisms from:

    SkillSight: Seeing Through Shared Descriptions for Accurate Skill
    Retrieval (Xiao et al., arxiv:2607.18785)

  * Semantic Background Calibration (SBC) — estimate the shared-background
    subspace from generic (low-IDF) tokens and project each description onto
    its orthogonal complement, so similarity induced by shared phrasing
    disappears.
  * Lexical Evidence Calibration (LEC) — score token overlap with IDF
    weights, so shared-background tokens are down-weighted and discriminative
    token-level evidence is recovered.

Adaptation note (Mode 2). The paper pairs these calibrations with a learned
dense encoder plus a reranker. This SDK ships no embedding model, so the
learned encoder is replaced by a parameter-free TF-IDF vector space. The
calibration logic itself — the paper's actual contribution — is reproduced
verbatim: SBC is a genuine background-subspace projection (in the bag-of-words
basis, projecting out the generic-token axes), and LEC is IDF-weighted
evidence. Only the auxiliary encoder is substituted. Nothing here requires a
network call or a model download; it ranks the ``Tool.metadata.description``
strings that ``ToolsManager.list_tools`` already returns.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "SkillDoc",
    "SkillHit",
    "SkillLibrary",
    "calibrated_retrieve",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Genuine English function words only — a floor below the data-driven IDF
# background detection. Domain words such as "tool"/"data"/"input" are
# intentionally NOT listed here so the per-library IDF pass can surface them
# as shared background when they recur across a library.
_STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have in is it its of on or that the
    to was were will with this these those which you your our their his her them
    they we i me my us using used uses use can could would should may might must
    into over when where while then than so such only also more most some any all
    each every both either neither not no nor but if else via per out up down off
    new get got do does did done been being am about above after again against
    because before below between during further here how itself other own same she
    there what who whom why within without across along among behind beyond except
    inside near outside past since toward under until upon takes take taking make
    makes making provide provides providing return based following like just very
    s t d ll re ve
    """.split()
)


def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens, dropping function words and 1-char noise."""
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS and len(tok) > 1]


@dataclass(frozen=True)
class SkillDoc:
    """A rankable tool/skill description.

    ``key`` is whatever stable handle the caller wants back (e.g. a
    ``studio_id``); ``text`` is the description ranked against the query.
    """

    key: str
    text: str


@dataclass
class SkillHit:
    """A ranked retrieval result."""

    key: str
    text: str
    rank: int
    score: float           # combined calibrated score (alpha-blend of the two)
    semantic_score: float  # SBC: cosine after projecting out background
    lexical_score: float   # LEC: IDF-weighted token overlap
    raw_score: float       # baseline dense cosine with background kept


def _unit_cosine(a: Counter, b: Counter, skip: Optional[set]) -> float:
    """Cosine similarity between two term->weight counters, skipping ``skip``.

    ``skip`` is the background axis set for SBC; pass ``None`` for the raw
    baseline (background kept). Zero vectors score 0.
    """
    if not a or not b:
        return 0.0
    # Iterate the smaller counter to keep this cheap.
    if len(a) > len(b):
        a, b = b, a
    dot = 0.0
    for term, w in a.items():
        if skip and term in skip:
            continue
        other = b.get(term)
        if other is not None and not (skip and term in skip):
            dot += w * other
    if dot == 0.0:
        return 0.0
    na = math.sqrt(sum(w * w for t, w in a.items() if not (skip and t in skip)))
    nb = math.sqrt(sum(w * w for t, w in b.items() if not (skip and t in skip)))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class SkillLibrary:
    """A tool/skill library with calibrated (background-projected) retrieval.

    Parameters
    ----------
    docs:
        The library to rank against.
    background_ratio:
        Share of docs a token must appear in to count as shared background.
        Generic phrases that recur across many tool descriptions (the paper's
        "shared descriptive background") clear this bar; discriminative tokens
        do not. ``0.5`` keeps single-doc tokens out of the background on small
        libraries.
    extra_background:
        Extra tokens to always treat as background (e.g. a project-specific
        boilerplate prefix), in addition to the IDF-derived set.
    """

    def __init__(
        self,
        docs: Sequence[SkillDoc],
        background_ratio: float = 0.5,
        extra_background: Optional[Iterable[str]] = None,
    ) -> None:
        if not docs:
            raise ValueError("SkillLibrary requires at least one document")
        self.docs: List[SkillDoc] = list(docs)
        self.n_docs = len(self.docs)

        tokenized = [_tokenize(d.text) for d in self.docs]

        # Document frequency -> smoothed IDF, the paper's "generic token" signal.
        df: Counter = Counter()
        for toks in tokenized:
            for term in set(toks):
                df[term] += 1
        self._idf = {
            term: math.log((1 + self.n_docs) / (1 + doc_freq)) + 1.0
            for term, doc_freq in df.items()
        }

        # Shared-background subspace: tokens recurring across enough of the
        # library, plus any caller-supplied boilerplate.
        threshold = background_ratio * self.n_docs
        self.background_tokens: set = {
            term for term, doc_freq in df.items() if doc_freq >= threshold
        }
        if extra_background:
            self.background_tokens.update(extra_background)

        # Precompute raw (background-kept) and calibrated (background-removed)
        # TF-IDF document vectors. SBC projects out the background axes here.
        self._raw_vectors: List[Counter] = []
        self._calibrated_vectors: List[Counter] = []
        for toks in tokenized:
            tf = Counter(toks)
            self._raw_vectors.append(Counter({t: c * self._idf.get(t, 0.0) for t, c in tf.items()}))
            self._calibrated_vectors.append(
                Counter(
                    {t: c * self._idf.get(t, 0.0) for t, c in tf.items()
                     if t not in self.background_tokens}
                )
            )

    def idf(self, term: str) -> float:
        """Smoothed inverse document frequency for ``term`` (unseen -> max)."""
        if term in self._idf:
            return self._idf[term]
        return math.log((1 + self.n_docs) / 1.0) + 1.0

    def _query_vectors(self, query: str) -> Tuple[Counter, Counter, List[str]]:
        toks = _tokenize(query)
        tf = Counter(toks)
        raw_q = Counter({t: c * self.idf(t) for t, c in tf.items()})
        calibrated_q = Counter(
            {t: c * self.idf(t) for t, c in tf.items() if t not in self.background_tokens}
        )
        return raw_q, calibrated_q, toks

    def _lexical_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """LEC: IDF-weighted token overlap, normalized by query evidence mass.

        Background tokens are kept but contribute their (low) IDF weight, so a
        description that only overlaps the query on shared phrasing scores low.
        """
        if not query_tokens:
            return 0.0
        doc_terms = set(self._raw_vectors[doc_idx])
        evidence = 0.0
        total = 0.0
        for term in query_tokens:
            w = self.idf(term)
            total += w
            if term in doc_terms:
                evidence += w
        if total == 0.0:
            return 0.0
        return evidence / total

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        alpha: float = 0.5,
    ) -> List[SkillHit]:
        """Rank the library against ``query``.

        ``alpha`` blends the two calibrated spaces: SBC semantic similarity
        and LEC lexical evidence (``alpha=1.0`` is semantic-only,
        ``alpha=0.0`` is lexical-only). ``raw_score`` on every hit is the
        baseline dense cosine with the background kept, for comparison.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        raw_q, calibrated_q, query_tokens = self._query_vectors(query)
        hits: List[SkillHit] = []
        for idx, doc in enumerate(self.docs):
            semantic = _unit_cosine(calibrated_q, self._calibrated_vectors[idx], skip=None)
            lexical = self._lexical_score(query_tokens, idx)
            raw = _unit_cosine(raw_q, self._raw_vectors[idx], skip=None)
            hits.append(
                SkillHit(
                    key=doc.key,
                    text=doc.text,
                    rank=0,
                    score=alpha * semantic + (1.0 - alpha) * lexical,
                    semantic_score=semantic,
                    lexical_score=lexical,
                    raw_score=raw,
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        limit = self.n_docs if top_k is None else max(0, top_k)
        for rank, hit in enumerate(hits[:limit]):
            hit.rank = rank
        return hits[:limit]

    @classmethod
    def from_tools(cls, tools: Iterable, background_ratio: float = 0.5) -> "SkillLibrary":
        """Build a library from the SDK's ``Tool`` objects.

        Reads ``tool.metadata.description`` (falling back to ``title``) and
        keys results by ``tool.metadata.studio_id``. Duck-typed on ``metadata``
        so no SDK import is needed inside the calibration core.
        """
        docs: List[SkillDoc] = []
        for tool in tools:
            meta = getattr(tool, "metadata", tool)
            key = getattr(meta, "studio_id", None) or getattr(meta, "tool_id", None) or str(id(tool))
            text = getattr(meta, "description", None) or getattr(meta, "title", None) or ""
            docs.append(SkillDoc(key=key, text=text))
        return cls(docs, background_ratio=background_ratio)


def calibrated_retrieve(
    query: str,
    tools: Iterable,
    top_k: Optional[int] = None,
    alpha: float = 0.5,
    background_ratio: float = 0.5,
) -> List[SkillHit]:
    """Convenience wrapper: rank an iterable of ``Tool`` objects for ``query``.

    Combines ``SkillLibrary.from_tools`` and ``SkillLibrary.retrieve`` in one
    call — the intended client-side surface over ``ToolsManager.list_tools``.
    """
    library = SkillLibrary.from_tools(tools, background_ratio=background_ratio)
    return library.retrieve(query, top_k=top_k, alpha=alpha)
