"""
Tests for relevanceai.calibrated_tool_retrieval.

These exercise the integration surface the module is built for: the SDK's
``Tool`` objects (returned by ``ToolsManager.list_tools``) flow into
``SkillLibrary.from_tools`` and come back ranked with shared descriptive
background calibrated out. They import the existing, non-new SDK modules
(``relevanceai.resources.tool`` / ``relevanceai.resources.tools``) the
retriever wraps, so the wiring is under test, not just the math.
"""

import pytest
from unittest.mock import MagicMock

from relevanceai.resources.tools import ToolsManager
from relevanceai.resources.tool import Tool
from relevanceai.calibrated_tool_retrieval import (
    SkillDoc,
    SkillLibrary,
    calibrated_retrieve,
)

# A library whose descriptions share a generic "background" prefix and differ
# only in a discriminative clause. This is the failure mode SkillSight targets:
# the shared phrasing inflates ordinary similarity and obscures the token that
# names the required capability.
BACKGROUND = "This tool takes input data and returns output data."

SHARED_BG_TOOLS = [
    {
        "studio_id": "translate",
        "title": "Translate French English",
        "description": f"{BACKGROUND} It translates French text into English.",
    },
    {
        "studio_id": "invoice",
        "title": "Extract Invoice Lines",
        "description": f"{BACKGROUND} It extracts line items from an invoice PDF.",
    },
    {
        "studio_id": "background-only",
        "title": "Generic Pass-Through",
        "description": BACKGROUND,
    },
]
QUERY = "tool that takes input data and returns output data and translates French to English"


def _library_from_shared_bg_docs() -> SkillLibrary:
    return SkillLibrary([SkillDoc(key=t["studio_id"], text=t["description"]) for t in SHARED_BG_TOOLS])


class TestBackgroundDetection:
    def test_shared_phrases_detected_as_background(self):
        """Tokens recurring across the library are flagged as background."""
        library = _library_from_shared_bg_docs()
        # Generic boilerplate that appears in every description -> background.
        assert {"tool", "data", "input", "output", "returns"} <= library.background_tokens
        # Discriminative tokens appear in one description only -> not background.
        assert {"translates", "french", "english", "invoice", "extracts"}.isdisjoint(
            library.background_tokens
        )


class TestCalibrationMechanism:
    def test_background_only_decoy_scored_zero_semantically(self):
        """A description whose only query overlap is shared background gets no
        SBC credit once the background subspace is projected out, while the
        raw baseline (background kept) still inflates it."""
        library = _library_from_shared_bg_docs()
        hits = {h.key: h for h in library.retrieve(QUERY)}

        # Raw baseline is fooled: the background-only description overlaps the
        # query entirely on shared phrasing, so its raw cosine is high.
        assert hits["background-only"].raw_score > 0.0
        # Calibration removes that shared-background subspace entirely.
        assert hits["background-only"].semantic_score == 0.0

    def test_target_ranks_first(self):
        """The capability the query actually names comes out on top."""
        library = _library_from_shared_bg_docs()
        ranked = library.retrieve(QUERY)
        assert ranked[0].key == "translate"
        assert ranked[0].semantic_score > 0.0
        # And strictly above the background-only decoy overall.
        hits = {h.key: h for h in ranked}
        assert hits["translate"].score > hits["background-only"].score


class TestFromToolsIntegration:
    """End-to-end through the real Tool/ToolsManager wiring."""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.project = "test-project"
        client.region = "test-region"
        return client

    @pytest.fixture
    def tools_manager(self, mock_client):
        return ToolsManager(client=mock_client)

    def _list_tools(self, tools_manager):
        """Use the real ToolsManager.list_tools path (mocked transport) to build
        genuine Tool objects, exactly as the example script does."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": SHARED_BG_TOOLS}
        tools_manager._client.get = MagicMock(return_value=mock_response)
        tools = tools_manager.list_tools()
        assert all(isinstance(t, Tool) for t in tools)
        return tools

    def test_retriever_wraps_live_tool_objects(self, tools_manager):
        """ToolsManager.list_tools() -> SkillLibrary.from_tools -> ranked hits."""
        tools = self._list_tools(tools_manager)
        hits = calibrated_retrieve(QUERY, tools, top_k=3)
        # The capability named in the query is retrieved first, beating decoys
        # that only share the generic descriptive background.
        assert hits[0].key == "translate"
        assert len(hits) == 3

    def test_from_tools_reads_metadata_description(self, tools_manager):
        """``from_tools`` pulls the description off Tool.metadata, not the title."""
        tools = self._list_tools(tools_manager)
        library = SkillLibrary.from_tools(tools)
        # Discriminative token from the description (absent from titles) is in vocab.
        assert library.idf("translates") > 0.0
        ranked = library.retrieve(QUERY)
        assert ranked[0].key == "translate"
