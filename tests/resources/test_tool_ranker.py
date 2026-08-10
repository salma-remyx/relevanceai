from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# Import from a NON-NEW module -- the SDK's Tool type -- to prove the ranker
# integrates with the existing tool-management path rather than being a
# self-contained stub.
from relevanceai.resources.tool import Tool
from relevanceai.utils.tool_ranker import (
    bm25_scores,
    hit_at_k,
    hybrid_rank,
    mean_hit_at_k,
    rank_tools,
    tfidf_cosine_scores,
)


def _make_tool(client, tool_id, title, description):
    return Tool(
        client=client,
        studio_id=tool_id,
        title=title,
        description=description,
        _id=f"{tool_id}-id",
        project="test-project",
    )


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def tool_library(mock_client):
    return [
        _make_tool(
            mock_client,
            "search-website",
            "Search Website",
            "Search a company website for contact emails and phone numbers.",
        ),
        _make_tool(
            mock_client,
            "summarize-text",
            "Summarize Text",
            "Summarize a long document into a short list of key points.",
        ),
        _make_tool(
            mock_client,
            "translate-text",
            "Translate Text",
            "Translate text between different natural languages.",
        ),
        _make_tool(
            mock_client,
            "sentiment-analysis",
            "Sentiment Analysis",
            "Classify the sentiment of customer product reviews.",
        ),
    ]


class TestToolRanker:
    def test_rank_tools_returns_gold_first(self, tool_library):
        """The hybrid ranker surfaces the matching tool at rank 0."""
        query = "find contact emails on a company website"
        ranked = rank_tools(query, tool_library, top_k=5)

        assert len(ranked) == len(tool_library)
        assert {t.tool_id for t in ranked} == {t.tool_id for t in tool_library}
        assert ranked[0].tool_id == "search-website"

    def test_rank_tools_respects_top_k(self, tool_library):
        ranked = rank_tools("translate text between languages", tool_library, top_k=2)
        assert len(ranked) == 2
        assert ranked[0].tool_id == "translate-text"

    def test_rank_tools_empty_library(self):
        assert rank_tools("anything", []) == []

    def test_hit_at_k_metric(self, tool_library):
        """hit@k behaves as the paper's metric (1 if gold in top-k, else 0)."""
        ranked = rank_tools("sentiment of these customer reviews", tool_library, top_k=5)
        ranked_ids = [t.tool_id for t in ranked]

        assert hit_at_k(ranked_ids, "sentiment-analysis", k=5) == 1
        # gold id not present in the ranking at all
        assert hit_at_k(ranked_ids, "does-not-exist", k=5) == 0
        # gold present but the k window is empty
        assert hit_at_k(ranked_ids, "sentiment-analysis", k=0) == 0

    def test_mean_hit_at_k(self, tool_library):
        queries_and_gold = [
            ("find contact emails on a company website", "search-website"),
            ("translate this text into french", "translate-text"),
            ("summarize a long document into key points", "summarize-text"),
        ]
        per_query = [
            [t.tool_id for t in rank_tools(q, tool_library, top_k=5)]
            for q, _ in queries_and_gold
        ]
        golds = [g for _, g in queries_and_gold]

        assert mean_hit_at_k(per_query, golds, k=5) == 1.0

    def test_hybrid_fuses_both_signals(self):
        """Lexical and dense proxies both contribute to the fused score."""
        query = "contact emails"
        docs = [
            "Search a company website for contact emails and phone numbers.",
            "Translate text between different natural languages.",
        ]
        lex = bm25_scores(query, docs)
        dense = tfidf_cosine_scores(query, docs)
        assert lex[0] > lex[1]
        assert dense[0] > dense[1]

        ranked = hybrid_rank(query, docs, top_k=2)
        assert ranked[0][0] == 0
