import pytest
from unittest.mock import MagicMock, patch
from relevanceai.resources.agent import Agent


class TestAgent:
    @pytest.fixture
    def mock_client(self):
        """Fixture to create a mock RelevanceAI client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def agent(self, mock_client):
        """Fixture to create an Agent instance."""
        metadata = {
            "agent_id": "test-agent",
            "name": "Test Agent",
            "_id": "agent-id-123",
            "project": "default-project",
            "actions": [],
        }
        return Agent(client=mock_client, **metadata)

    def test_agent_init(self, agent):
        """Test initialization of the Agent class."""
        assert agent.agent_id == "test-agent"
        assert agent.metadata.get("name") == "Test Agent"

    def test_list_tools(self, agent):
        """Test listing tools."""
        mock_tool = {"tool_id": "tool-1", "title": "Test Tool", "metadata": {"type": "regular", "title": "Test Tool"}}
        agent.list_tools = MagicMock(return_value=[mock_tool])
        result = agent.list_tools()
        assert len(result) == 1
        assert result[0]["tool_id"] == "tool-1"
        assert result[0]["title"] == "Test Tool"

    @patch("relevanceai.resources.agent.Tool")
    def test_list_subagents(self, mock_tool_class, agent):
        """Test listing subagents."""
        mock_agent_tool = {
            "tool_id": "agent-1",
            "title": "Subagent",
            "metadata": {"type": "agent", "title": "Subagent"},
        }
        mock_regular_tool = {
            "tool_id": "tool-1",
            "title": "Regular Tool",
            "metadata": {"type": "regular", "title": "Regular Tool"},
        }

        agent._post = MagicMock(
            return_value={
                "results": [
                    {"tool_id": "agent-1", "type": "agent", "title": "Subagent"},
                    {"tool_id": "tool-1", "type": "regular", "title": "Regular Tool"},
                ]
            }
        )

        mock_tool_class.side_effect = [mock_agent_tool, mock_regular_tool]

        result = agent.list_subagents()

        assert len(result) == 1
        assert result[0]["tool_id"] == "agent-1"
        assert result[0]["title"] == "Subagent"

        agent._post.assert_called_once_with("agents/tools/list", body={"agent_ids": ["test-agent"]}, cast_to=dict)

    def test_retrieve_task(self, agent):
        """Test retrieving a task by conversation ID."""
        task_mock = MagicMock(knowledge_set="conversation-1")
        agent.list_tasks = MagicMock(return_value=[task_mock])
        task = agent.retrieve_task("conversation-1")
        assert task.knowledge_set == "conversation-1"

    def test_approve_task(self, agent):
        """Test approving a task."""
        task_view_mock = {
            "results": [
                {
                    "content": {
                        "requires_confirmation": True,
                        "tool_config": {"id": "tool-1"},
                        "action_details": {"action": "action-1", "action_request_id": "req-1"},
                    }
                }
            ]
        }
        agent.view_task_steps = MagicMock(return_value=task_view_mock)
        agent.trigger_task_from_action = MagicMock(return_value="Triggered")
        result = agent.approve_task("conversation-1", tool_id="tool-1")
        assert result == "Triggered"

    def test_rerun_task(self, agent):
        """Test rerun task with valid trigger message."""
        agent._get_trigger_message = MagicMock(return_value=("Test message", "message-123"))

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "task_id": "task-123",
            "status": "pending",
            "job_info": {"job_id": "job-123", "studio_id": "studio-123"},
            "conversation_id": "conversation-123",
            "agent_id": "test-agent",
            "state": "pending-approval",
        }
        agent._post = MagicMock(return_value=mock_response)

        result = agent.rerun_task("conversation-123")

        agent._post.assert_called_once_with(
            "agents/trigger",
            body={
                "agent_id": "test-agent",
                "conversation_id": "conversation-123",
                "message": {"role": "user", "content": "Test message"},
                "edit_message_id": "message-123",
            },
            cast_to=dict,
        )

        assert isinstance(result, MagicMock)

    def test_get_trigger_message_success(self, agent):
        """Test getting trigger message with valid response."""
        mock_response = {
            "trigger_message": {
                "content": {"is_trigger_message": True, "text": "Hello world"},
                "item_id": "message-123",
            }
        }
        agent._get = MagicMock(return_value=mock_response)

        result = agent._get_trigger_message("conversation-123")

        agent._get.assert_called_once_with("agents/test-agent/tasks/conversation-123/trigger_message", cast_to=dict)

        assert result == ("Hello world", "message-123")

    def test_get_trigger_message_no_trigger(self, agent):
        """Test getting trigger message when no trigger message exists."""
        mock_response = {
            "trigger_message": {
                "content": {"is_trigger_message": False, "text": "Hello world"},
                "item_id": "message-123",
            }
        }
        agent._get = MagicMock(return_value=mock_response)

        result = agent._get_trigger_message("conversation-123")

        assert result is None

    def test_get_trigger_message_no_data(self, agent):
        """Test getting trigger message when no data exists."""
        mock_response = {"trigger_message": None}
        agent._get = MagicMock(return_value=mock_response)

        result = agent._get_trigger_message("conversation-123")

        assert result is None

    def test_schedule_action_in_task(self, agent):
        """Test successful scheduling of an action in a task."""
        mock_response = {
            "trigger_id": "trigger-123",
            "conversation_id": "conv-123",
            "message": "Test message",
            "scheduled_time": "2024-03-20T10:00:00Z",
            "status": "pending",
        }
        agent._post = MagicMock(return_value=mock_response)

        result = agent.schedule_action_in_task(
            conversation_id="conv-123", message="Test message", minutes_until_schedule=30
        )

        agent._post.assert_called_once_with(
            "agents/test-agent/scheduled_triggers_item/create",
            body={"conversation_id": "conv-123", "message": "Test message", "minutes_until_schedule": 30},
            params=None,
            cast_to=dict,
        )

        assert isinstance(result, dict)
        assert result["trigger_id"] == "trigger-123"
        assert result["conversation_id"] == "conv-123"
        assert result["message"] == "Test message"

    def test_get_task_output_preview_success(self, agent):
        """Test getting task output preview when task is complete."""
        mock_response = {"results": [{"status": "complete", "output_preview": "Task output summary"}]}
        agent._get = MagicMock(return_value=mock_response)

        result = agent.get_task_output_preview("conversation-123")

        agent._get.assert_called_once_with(
            "agents/conversations/studios/list",
            params={
                "conversation_id": "conversation-123",
                "agent_id": "test-agent",
                "page_size": 100,
            },
            cast_to=dict,
        )

        assert result == "Task output summary"

    def test_remove_all_tools(self, agent):
        """Test removing all tools from an agent."""
        agent._post = MagicMock(return_value={"status": "success"})

        result = agent.remove_all_tools()

        agent._post.assert_called_once_with(
            "agents/upsert",
            body={
                "agent_id": "test-agent",
                "actions": [],
                "partial_update": True,
            },
            cast_to=dict,
        )

        assert agent.metadata["actions"] == []
        assert result == {"status": "success"}

    def test_add_tool(self, agent):
        """Test adding a tool to an agent."""
        agent._post = MagicMock(return_value={"status": "success"})
        agent.metadata["actions"] = []

        result = agent.add_tool("tool-123")

        agent._post.assert_called_once_with(
            "agents/upsert",
            body={
                "agent_id": "test-agent",
                "actions": [{"chain_id": "tool-123"}],
                "partial_update": True,
            },
            cast_to=dict,
        )

        assert len(agent.metadata["actions"]) == 1
        assert agent.metadata["actions"][0]["chain_id"] == "tool-123"
        assert result == {"status": "success"}

    def test_remove_tool(self, agent):
        """Test removing a specific tool from an agent."""
        agent._post = MagicMock(return_value={"status": "success"})

        agent.metadata["actions"] = [{"chain_id": "tool-123"}, {"chain_id": "tool-456"}]

        result = agent.remove_tool("tool-123")

        agent._post.assert_called_once_with(
            "agents/upsert",
            body={
                "agent_id": "test-agent",
                "actions": [{"chain_id": "tool-456"}],
                "partial_update": True,
            },
            cast_to=dict,
        )

        assert len(agent.metadata["actions"]) == 1
        assert agent.metadata["actions"][0]["chain_id"] == "tool-456"
        assert result == {"status": "success"}

    def test_add_subagent(self, agent):
        """Test adding a subagent to an agent."""
        agent._post = MagicMock(return_value={"status": "success"})

        mock_subagent = MagicMock()
        mock_subagent.metadata = {"agent_id": "subagent-123", "name": "Test Subagent"}
        agent._client.agents.retrieve_agent = MagicMock(return_value=mock_subagent)

        agent.metadata["actions"] = []

        result = agent.add_subagent("subagent-123", action_behaviour="always-ask")

        agent._post.assert_called_once_with(
            "agents/upsert",
            body={
                "agent_id": "test-agent",
                "actions": [
                    {
                        "action_behaviour": "always-ask",
                        "agent_id": "subagent-123",
                        "default_values": {},
                        "title": "Test Subagent",
                    }
                ],
                "partial_update": True,
            },
            cast_to=dict,
        )

        assert len(agent.metadata["actions"]) == 1
        assert agent.metadata["actions"][0]["agent_id"] == "subagent-123"
        assert result == {"status": "success"}

    def test_remove_subagent(self, agent):
        """Test removing a subagent from an agent."""
        agent._post = MagicMock(return_value={"status": "success"})

        agent.metadata["actions"] = [
            {"chain_id": "tool-123"},
            {"agent_id": "subagent-123", "action_behaviour": "always-ask", "title": "Test Subagent"},
            {"chain_id": "tool-456"},
        ]

        result = agent.remove_subagent("subagent-123")

        agent._post.assert_called_once_with(
            "agents/upsert",
            body={
                "agent_id": "test-agent",
                "actions": [{"chain_id": "tool-123"}, {"chain_id": "tool-456"}],
                "partial_update": True,
            },
            cast_to=dict,
        )

        assert len(agent.metadata["actions"]) == 2
        assert all("agent_id" not in action for action in agent.metadata["actions"])
        assert result == {"status": "success"}

    def test_update_core_instructions(self, agent):
        """Test updating core instructions of an agent."""
        agent._post = MagicMock(return_value={"status": "success"})

        core_instructions = "New core instructions for the agent"
        result = agent.update_core_instructions(core_instructions)

        agent._post.assert_called_once_with(
            "agents/upsert",
            body={
                "agent_id": "test-agent",
                "system_prompt": core_instructions,
                "partial_update": True,
            },
            cast_to=dict,
        )

        assert result == {"status": "success"}

    def test_update_template_settings(self, agent):
        """Test updating template settings with various parameter types."""
        mock_string_param = MagicMock(
            value="test_name",
            description="Test name parameter",
            required=True,
            model_dump=lambda exclude_none=False: {
                "value": "test_name",
                "description": "Test name parameter",
                "type": "string",
            },
        )

        mock_integer_param = MagicMock(
            value=25,
            description="Test age parameter",
            required=False,
            model_dump=lambda exclude_none=False: {"value": 25, "description": "Test age parameter", "type": "integer"},
        )

        mock_null_param = MagicMock(
            value=None,
            description="Test title parameter",
            required=False,
            model_dump=lambda exclude_none=False: {"description": "Test title parameter", "type": "string"},
        )

        agent._post = MagicMock(return_value={"status": "success"})

        params = {"name": mock_string_param, "age": mock_integer_param, "title": mock_null_param}

        result = agent.update_template_settings(params)

        agent._post.assert_called_once_with(
            "agents/upsert",
            body={
                "agent_id": "test-agent",
                "params": {"name": "test_name", "age": 25},
                "params_schema": {
                    "properties": {
                        "name": {"value": "test_name", "description": "Test name parameter", "type": "string"},
                        "age": {"value": 25, "description": "Test age parameter", "type": "integer"},
                        "title": {"description": "Test title parameter", "type": "string"},
                    },
                    "required": ["name"],
                },
                "partial_update": True,
            },
            cast_to=dict,
        )

        assert result == {"status": "success"}

    def test_update_template_settings_empty_params(self, agent):
        """Test updating template settings with empty parameters."""
        agent._post = MagicMock(return_value={"status": "success"})

        result = agent.update_template_settings({})

        agent._post.assert_called_once_with(
            "agents/upsert",
            body={
                "agent_id": "test-agent",
                "params": {},
                "params_schema": {"properties": {}, "required": []},
                "partial_update": True,
            },
            cast_to=dict,
        )

        assert result == {"status": "success"}

    def test_update_template_settings_partial_update_false(self, agent):
        """Test updating template settings with partial_update=False."""
        mock_string_param = MagicMock(
            value="test_name",
            description="Test name parameter",
            required=True,
            model_dump=lambda exclude_none=False: {
                "value": "test_name",
                "description": "Test name parameter",
                "type": "string",
            },
        )

        agent._post = MagicMock(return_value={"status": "success"})

        params = {"name": mock_string_param}

        result = agent.update_template_settings(params, partial_update=False)

        agent._post.assert_called_once_with(
            "agents/upsert",
            body={
                "agent_id": "test-agent",
                "params": {"name": "test_name"},
                "params_schema": {
                    "properties": {
                        "name": {"value": "test_name", "description": "Test name parameter", "type": "string"}
                    },
                    "required": ["name"],
                },
                "partial_update": False,
            },
            cast_to=dict,
        )

        assert result == {"status": "success"}

    def test_get_link(self, agent):
        """Test getting the web link for an agent."""
        agent._client.region = "us-east-1"
        agent._client.project = "test-project"

        expected_link = "https://app.relevanceai.com/agents/us-east-1/test-project/test-agent"
        result = agent.get_link()

        assert result == expected_link
