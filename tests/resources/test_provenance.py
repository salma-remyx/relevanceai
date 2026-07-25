"""Integration tests for the agent provenance graph builder.

These build a real ``TaskView`` (from ``relevanceai.types.task``) and exercise
the wiring through ``relevanceai.resources.agent.Agent`` — i.e. they prove the
provenance module integrates with the SDK's actual call site
(``Agent.view_task_steps``), not just self-test the new module.
"""

from unittest.mock import MagicMock

import pytest

from relevanceai.provenance import (
    ProvenanceGraph,
    build_provenance,
    quotient_graph,
    recurring_patterns,
)
from relevanceai.resources.agent import Agent
from relevanceai.types.task import (
    ActionDetails,
    Content as UserMessage,
    Content1 as AgentMessage,
    Content2 as ToolRun,
    OriginalMessageIds,
    OriginalMessageIds1,
    OriginalMessageIds2,
    Params,
    TaskStep,
    TaskView,
    ToolConfig,
    ToolRunState,
)


def _user_step(item_id: str, text: str) -> TaskStep:
    return TaskStep(
        item_id=item_id,
        insert_date_="2026-01-01T00:00:00Z",
        is_expanded_by_default=False,
        is_in_hidden_group=False,
        content=UserMessage(
            type="user-message",
            text=text,
            is_trigger_message=True,
            original_message_ids=OriginalMessageIds(user="user-1"),
        ),
    )


def _agent_step(item_id: str, text: str) -> TaskStep:
    return TaskStep(
        item_id=item_id,
        insert_date_="2026-01-01T00:00:00Z",
        is_expanded_by_default=False,
        is_in_hidden_group=False,
        content=AgentMessage(
            type="agent-message",
            text=text,
            original_message_ids=OriginalMessageIds1(agent="agent-1"),
        ),
    )


def _tool_step(
    item_id: str,
    request_id: str,
    tool_id: str,
    title: str,
    action: str,
    output=None,
    declared=None,
) -> TaskStep:
    declared = declared or {}
    return TaskStep(
        item_id=item_id,
        insert_date_="2026-01-01T00:00:00Z",
        is_expanded_by_default=False,
        is_in_hidden_group=False,
        content=ToolRun(
            type="tool-run",
            tool_run_state=ToolRunState.finished,
            tool_config=ToolConfig(
                type="tool",
                title=title,
                description="a tool",
                id=tool_id,
                version="1",
                params_schema={},
            ),
            action_details=ActionDetails(action_request_id=request_id, action=action),
            requires_confirmation=False,
            params=Params(
                valid=True,
                **{"json": declared, "resolved": declared},
            ),
            output=output,
            original_message_ids=OriginalMessageIds2(**{"action-request": request_id}),
        ),
    )


def _sample_task_view() -> TaskView:
    """A trajectory where Summarize consumes Search's output via a template ref."""
    return TaskView(
        results=[
            _user_step("u1", "Research RelevanceAI"),
            _tool_step(
                "s2",
                "req-search",
                "tid-search",
                "Search Website",
                "search",
                output={"value": "Relevance AI builds AI agents"},
            ),
            _tool_step(
                "s3",
                "req-sum",
                "tid-summarize",
                "Summarize",
                "summarize",
                declared={"text": "${{ output.value }}"},
            ),
            _agent_step("s4", "Summary: Relevance AI builds AI agents"),
        ]
    )


@pytest.fixture
def graph():
    return build_provenance(_sample_task_view(), conversation_id="conv-1")


class TestProvenanceGraph:
    def test_actions_and_artifacts(self, graph):
        assert {a.attributes["tool_id"] for a in graph.actions()} == {
            "tid-search",
            "tid-summarize",
        }
        outputs = [n for n in graph.artifacts() if n.node_type == "tool-output"]
        assert len(outputs) == 1
        assert outputs[0].attributes["key"] == "value"

    def test_data_dependency_between_tool_runs(self, graph):
        """The Summarize action must depend on the Search action's output."""
        summarise_incoming = [
            e for e in graph.edges if e.target == "action:req-sum"
        ]
        assert summarise_incoming, "Summarize action has no incoming dependency"
        edge = summarise_incoming[0]
        assert edge.relation == "consumed-by"
        # dependency source is Search's produced artifact, not Search itself
        assert edge.source == "artifact:req-search:value"

    def test_message_feeds_and_produces(self, graph):
        relations = {(e.source, e.target, e.relation) for e in graph.edges}
        # user trigger message feeds the first tool action
        assert ("message:u1", "action:req-search", "feeds") in relations
        # the search action produces its output artifact
        assert ("action:req-search", "artifact:req-search:value", "produces") in relations
        # the summarise action produces the final agent message
        assert ("action:req-sum", "message:s4", "produces") in relations

    def test_to_dot_is_valid_graph(self, graph):
        dot = graph.to_dot()
        assert dot.startswith("digraph provenance {")
        assert '"action:req-search"' in dot
        assert "->" in dot


class TestQuotientGraph:
    def test_merges_recurring_tools_across_trajectories(self, graph):
        # second trajectory: same tools, different request ids + conversation
        second = build_provenance(_sample_task_view(), conversation_id="conv-2")
        joined = quotient_graph([graph, second])

        search_nodes = [
            n for n in joined.nodes.values() if n.kind == "action"
            and n.attributes.get("tool_id") == "tid-search"
        ]
        assert len(search_nodes) == 1, "recurring tool should merge to one node"
        assert search_nodes[0].attributes["occurrences"] == 2
        # two trajectories x two actions each -> two merged action nodes
        assert len([n for n in joined.nodes.values() if n.kind == "action"]) == 2

    def test_distinguishes_different_tools(self):
        other = build_provenance(
            TaskView(
                results=[
                    _user_step("u1", "Do something else"),
                    _tool_step(
                        "s2", "req-x", "tid-other", "Other Tool", "run", output={}
                    ),
                ]
            ),
            conversation_id="conv-other",
        )
        joined = quotient_graph([build_provenance(_sample_task_view()), other])
        tool_ids = {
            n.attributes["tool_id"]
            for n in joined.nodes.values()
            if n.kind == "action"
        }
        assert tool_ids == {"tid-search", "tid-summarize", "tid-other"}


class TestRecurringPatterns:
    def test_surfaces_tool_adjacency(self, graph):
        second = build_provenance(_sample_task_view(), conversation_id="conv-2")
        patterns = {
            tuple(p["pattern"]): p["count"]
            for p in recurring_patterns([graph, second], n=2)
        }
        assert patterns[("tid-search", "tid-summarize")] == 2


class TestAgentWiring:
    def test_builds_from_agent_view_task_steps(self):
        """Exercises the real call site: Agent.view_task_steps -> build_provenance."""
        client = MagicMock()
        agent = Agent(
            client=client,
            agent_id="test-agent",
            name="Test Agent",
            _id="agent-123",
            project="default-project",
        )
        agent.view_task_steps = MagicMock(return_value=_sample_task_view())

        graph = build_provenance(agent.view_task_steps(conversation_id="conv-1"))
        assert isinstance(graph, ProvenanceGraph)
        assert len(graph.actions()) == 2
        agent.view_task_steps.assert_called_once_with(conversation_id="conv-1")
