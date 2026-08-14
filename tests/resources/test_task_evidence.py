"""Integration tests for the task evidence-grounding check.

These exercise the wiring added to ``examples.approve_tasks_in_bulk`` (the
non-new call-site module) against real ``TaskView`` / ``TaskStep`` objects, so
they prove the capability integrates with the SDK's task-trace surface rather
than only self-testing the new module.
"""

from unittest.mock import MagicMock

import examples.approve_tasks_in_bulk as bulk_demo
from relevanceai.types.task import (
    ActionDetails,
    Content,
    Content1,
    Content2,
    OriginalMessageIds,
    OriginalMessageIds1,
    OriginalMessageIds2,
    Params,
    TaskStep,
    TaskView,
    ToolConfig,
)
from relevanceai.utils.task_evidence import NodeType, build_evidence_graph

GROUNDING_DATA = {
    "summary": (
        "RelevanceAI is an AI company founded in 2020 based in Sydney "
        "building vector search tooling."
    )
}


def _step(content, item_id):
    return TaskStep(
        item_id=item_id,
        insert_date_="",
        is_expanded_by_default=False,
        is_in_hidden_group=False,
        content=content,
    )


def _user_step(text, item_id):
    return _step(
        Content(
            type="user-message",
            text=text,
            original_message_ids=OriginalMessageIds(user="u"),
        ),
        item_id,
    )


def _tool_step(output, item_id):
    return _step(
        Content2(
            type="tool-run",
            tool_run_state="finished",
            tool_config=ToolConfig(
                type="tool",
                title="research",
                description="d",
                id="tid",
                version="1",
                params_schema={},
            ),
            action_details=ActionDetails(action_request_id="r1", action="run"),
            requires_confirmation=False,
            params=Params(valid=True, **{"json": {}}),
            output=output,
            original_message_ids=OriginalMessageIds2(**{"action-request": "ar1"}),
        ),
        item_id,
    )


def _agent_step(text, item_id):
    return _step(
        Content1(
            type="agent-message",
            text=text,
            original_message_ids=OriginalMessageIds1(agent="a1"),
        ),
        item_id,
    )


def _task_view(claim_text):
    """A trace whose final agent claim asserts ``claim_text`` over the tool data."""
    return TaskView(
        results=[
            _user_step("Research RelevanceAI", "u"),
            _tool_step(GROUNDING_DATA, "t"),
            _agent_step(claim_text, "c"),
        ]
    )


def _fake_agent(views):
    """An Agent-like mock whose ``view_task_steps`` returns the given TaskViews."""
    agent = MagicMock()

    def _view(conversation_id):
        return views[conversation_id]

    agent.view_task_steps.side_effect = _view
    return agent


class TestTaskEvidenceIntegration:
    def test_run_evidence_check_flags_ungrounded_claim(self):
        """A claim inventing facts the tool never produced is flagged weak."""
        ungrounded = _task_view(
            "RelevanceAI was acquired by Google in 2019 for 5 billion "
            "and relocated to London."
        )
        agent = _fake_agent({"bad": ungrounded})

        reports = bulk_demo.run_evidence_check(agent, ["bad"])

        report = reports["bad"]
        assert report.task_consistent is False
        assert report.total_claims == 1
        assert report.grounded_claims == 0
        assert report.earliest_weak_node is not None
        assert report.earliest_weak_node.reason == "semantic-misalignment"

    def test_run_evidence_check_accepts_grounded_claim(self):
        """A claim whose content is drawn from the tool output passes."""
        grounded = _task_view(
            "RelevanceAI is an AI company founded in 2020 based in Sydney."
        )
        agent = _fake_agent({"good": grounded})

        reports = bulk_demo.run_evidence_check(agent, ["good"])

        report = reports["good"]
        assert report.task_consistent is True
        assert report.grounded_claims == 1
        assert report.weak_nodes == []
        assert report.earliest_weak_node is None

    def test_build_evidence_graph_projects_real_task_view(self):
        """The graph builder classifies real task-step content types correctly."""
        graph = build_evidence_graph(_task_view("RelevanceAI is based in Sydney."))
        types = [node.node_type for node in graph.nodes]
        assert types == [NodeType.PROBLEM, NodeType.EVIDENCE, NodeType.CLAIM]

    def test_claim_without_evidence_is_missing_dependency(self):
        """A claim with no preceding tool data is a missing-dependency weak node."""
        view = TaskView(results=[_agent_step("Here is my conclusion.", "c")])
        agent = _fake_agent({"empty": view})

        reports = bulk_demo.run_evidence_check(agent, ["empty"])

        report = reports["empty"]
        assert report.task_consistent is False
        assert report.earliest_weak_node.reason == "missing-dependency"
