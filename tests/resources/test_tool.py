import pytest
from unittest.mock import MagicMock
from relevanceai.resources.tool import Tool


class TestTool:
    @pytest.fixture
    def mock_client(self):
        """Fixture to create a mock RelevanceAI client."""
        client = MagicMock()
        client.project = "test-project"
        client.region = "us-east-1"
        return client

    @pytest.fixture
    def tool(self, mock_client):
        """Fixture to create a Tool instance."""
        return Tool(client=mock_client, tool_id="test-tool")

    def test_tool_init(self, tool):
        """Test initialization of the Tool class."""
        assert tool.tool_id == "test-tool"

    def test_update(self, tool):
        """Test updating tool properties."""
        mock_response = {"status": "success"}
        tool._post = MagicMock(return_value=mock_response)

        updates = {"title": "Updated Tool"}
        result = tool.update(updates)

        tool._post.assert_called_once_with(
            "studios/bulk_update",
            body={"partial_update": True, "updates": [{"title": "Updated Tool", "studio_id": "test-tool"}]},
            cast_to=dict,
        )
        assert result == {"status": "success"}

    def test_trigger(self, tool):
        """Test triggering a tool."""
        mock_response = {"output": {"result": "test output"}, "status": "complete", "errors": [], "executionTime": 1.23}
        tool._post = MagicMock(return_value=mock_response)

        params = {"param1": "value1"}
        result = tool.trigger(params)

        tool._post.assert_called_once_with(
            path=f"studios/{tool.tool_id}/trigger_limited",
            body={"params": params, "project": tool._client.project},
            cast_to=dict,
        )
        assert result["output"] == {"result": "test output"}
        assert result["status"] == "complete"
        assert result["errors"] == []
        assert result["executionTime"] == 1.23

    def test_get_params_schema(self, tool):
        """Test getting params schema."""
        mock_response = {"studio": {"params_schema": {"properties": {"test": "schema"}}}}
        tool._get = MagicMock(return_value=mock_response)

        result = tool.get_params_schema()
        assert result == {"test": "schema"}

    def test_update_metadata(self, tool):
        """Test updating tool metadata."""
        get_response = {"studio": {"title": "Old Title", "description": "Old Description", "public": False}}
        tool._get = MagicMock(return_value=get_response)

        post_response = {"status": "success"}
        tool._post = MagicMock(return_value=post_response)

        result = tool.update_metadata(title="New Title", description="New Description", public=True)

        tool._post.assert_called_once_with(
            "studios/bulk_update",
            body={
                "updates": [
                    {"studio_id": "test-tool", "title": "New Title", "description": "New Description", "public": True}
                ],
                "partial_update": True,
            },
            cast_to=dict,
        )
        assert result["status"] == "success"

    def test_get_link(self, tool):
        """Test getting the web link for a tool."""
        expected_link = "https://app.relevanceai.com/agents/us-east-1/test-project/test-tool"
        result = tool.get_link()
        assert result == expected_link

    def test_repr(self, tool):
        """Test string representation of the tool."""
        expected_repr = 'Tool(tool_id="test-tool")'
        assert repr(tool) == expected_repr
