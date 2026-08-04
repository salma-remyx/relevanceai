import json
from unittest.mock import MagicMock

import pytest

# NON-NEW call-site module: the Agent resource the harness wraps. These tests
# construct a real Agent and drive its existing task-execution methods through
# the harness, proving the integration rather than self-testing the harness.
from relevanceai.resources.agent import Agent

# New capability module under test.
from examples.task_harness import (
    AuditTrail,
    TaskHarness,
    default_judge,
    run_reliability_comparison,
)


def _triggered(conversation_id="conv-1"):
    """Minimal TriggeredTask stand-in exposing the attribute the harness reads."""
    return MagicMock(conversation_id=conversation_id)


class TestTaskHarness:
    @pytest.fixture
    def mock_client(self):
        """Fixture to create a mock RelevanceAI client."""
        return MagicMock()

    @pytest.fixture
    def agent(self, mock_client):
        """Fixture to create a real Agent instance (mocked transport)."""
        metadata = {
            "agent_id": "test-agent",
            "name": "Test Agent",
            "_id": "agent-id-123",
            "project": "default-project",
        }
        return Agent(client=mock_client, **metadata)

    def test_judge_loop_retries_until_accepted(self, agent):
        """Harness reruns the task when the judge rejects, then passes."""
        outputs = iter(["", "Final summary"])  # 1st bad, after rerun good
        agent.trigger_task = MagicMock(return_value=_triggered("conv-1"))
        agent.get_task_output_preview = MagicMock(
            side_effect=lambda *a, **k: next(outputs)
        )
        agent.rerun_task = MagicMock(return_value=_triggered("conv-1"))
        agent.view_task_steps = MagicMock(return_value=MagicMock(results=[]))

        harness = TaskHarness(agent, judge=default_judge)
        output, trail = harness.run("summarise", max_retries=2)

        assert output == "Final summary"
        agent.rerun_task.assert_called_once()
        statuses = trail.statuses()
        assert "rejected" in statuses
        assert statuses[-1] == "ok"  # final judge accepts

    def test_judge_loop_respects_retry_bound(self, agent):
        """When retries are exhausted the harness returns the last output."""
        agent.trigger_task = MagicMock(return_value=_triggered("conv-1"))
        agent.get_task_output_preview = MagicMock(return_value="")
        agent.rerun_task = MagicMock(return_value=_triggered("conv-1"))
        agent.view_task_steps = MagicMock(return_value=MagicMock(results=[]))

        harness = TaskHarness(agent, judge=default_judge)
        output, trail = harness.run("summarise", max_retries=1)

        assert output == ""  # never accepted
        assert agent.rerun_task.call_count == 1  # bounded, not unbounded
        assert trail.to_list()[-1]["status"] == "error"

    def test_hitl_gate_invokes_approve_task(self, agent):
        """When a step requires confirmation, the harness approves it once."""
        step = MagicMock(content=MagicMock(requires_confirmation=True))
        agent.trigger_task = MagicMock(return_value=_triggered("conv-1"))
        agent.get_task_output_preview = MagicMock(return_value="ok")
        agent.rerun_task = MagicMock(return_value=_triggered("conv-1"))
        agent.view_task_steps = MagicMock(return_value=MagicMock(results=[step]))
        agent.approve_task = MagicMock(return_value=_triggered("conv-1"))

        harness = TaskHarness(agent)
        output, trail = harness.run(
            "summarise", max_retries=0, approve_confirmation=True
        )

        agent.approve_task.assert_called_once_with("conv-1")
        assert any(n["node"] == "hitl" for n in trail.to_list())
        assert output == "ok"

    def test_reliability_comparison_harness_beats_baseline(self, agent):
        """Harness retries a flaky agent to a pass where the baseline fails."""
        state = {"reran": False}

        def trigger(message=""):
            return _triggered("conv-1")

        def preview(*a, **k):
            return "good" if state["reran"] else ""

        def rerun(conversation_id):
            state["reran"] = True
            return _triggered(conversation_id)

        agent.trigger_task = MagicMock(side_effect=trigger)
        agent.get_task_output_preview = MagicMock(side_effect=preview)
        agent.rerun_task = MagicMock(side_effect=rerun)
        agent.view_task_steps = MagicMock(return_value=MagicMock(results=[]))

        report = run_reliability_comparison(
            agent, scenarios=["do the thing"], judge=default_judge, max_retries=2
        )

        assert report["n"] == 1
        assert report["baseline_pass_rate"] == 0.0
        assert report["harness_pass_rate"] == 1.0

    def test_audit_trail_is_json_serialisable(self):
        trail = AuditTrail()
        trail.record("trigger", "ok", "conv-1")
        trail.record("judge", "rejected", "empty")

        data = trail.to_list()
        assert len(data) == 2
        assert {d["node"] for d in data} == {"trigger", "judge"}
        # Round-trips through JSON without error.
        json.loads(trail.to_json())
