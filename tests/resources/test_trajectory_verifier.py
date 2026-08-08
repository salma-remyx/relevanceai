import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

# NON-NEW modules: the real call-site surface + telemetry types.
from relevanceai.resources.agent import AsyncAgent
from relevanceai.types.task import TaskView

# The capability under test.
from relevanceai.utils.trajectory_verifier import (
    recompute_total,
    verify_and_repair,
    verify_task,
    verify_trajectory,
)


def _tool_run_step(
    tool_id,
    state="finished",
    resolved=None,
    output=None,
    errors=None,
    item_id="step",
):
    """Build a real TaskStep whose content is a tool-run, from a dict."""
    content = {
        "type": "tool-run",
        "tool_run_state": state,
        "tool_config": {
            "type": "tool",
            "title": tool_id,
            "description": "d",
            "id": tool_id,
            "version": "1",
            "params_schema": {},
        },
        "action_details": {"action_request_id": "r", "action": "run"},
        "requires_confirmation": False,
        "params": {"valid": True, "json": resolved or {}, "resolved": resolved or {}},
        "original_message_ids": {"action-request": "ar"},
    }
    if output is not None:
        content["output"] = output
    if errors is not None:
        content["errors"] = errors
    return {
        "item_id": item_id,
        "insert_date_": "2025-01-01T00:00:00Z",
        "is_expanded_by_default": False,
        "is_in_hidden_group": False,
        "content": content,
    }


def _agent_error_step(item_id="err"):
    return {
        "item_id": item_id,
        "insert_date_": "2025-01-01T00:00:00Z",
        "is_expanded_by_default": False,
        "is_in_hidden_group": False,
        "content": {
            "type": "agent-error",
            "errors": [{"body": "boom"}],
            "original_message_ids": {"agent-error": "ae"},
        },
    }


def _view(steps):
    """Build a real TaskView from raw step dicts."""
    return TaskView.model_validate({"results": steps})


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def async_agent(mock_client):
    return AsyncAgent(
        client=mock_client,
        agent_id="test-agent",
        _id="agent-id",
        project="default-project",
    )


class TestTrajectoryVerifier:
    def test_clean_trajectory_passes(self):
        # Two distinct, finished tool calls with no errors -> healthy run.
        view = _view(
            [
                _tool_run_step("search", resolved={"q": "a"}, output={"value": 3}),
                _tool_run_step("search", resolved={"q": "b"}, output={"value": 4}),
            ]
        )
        report = verify_trajectory(view)
        assert report.passed
        assert report.codes == []
        assert bool(report) is True

    def test_detects_loop(self):
        # Same tool + identical resolved params repeated three times -> loop.
        view = _view(
            [
                _tool_run_step("search", resolved={"q": "same"}),
                _tool_run_step("search", resolved={"q": "same"}),
                _tool_run_step("search", resolved={"q": "same"}),
            ]
        )
        report = verify_trajectory(view)
        assert not report.passed
        assert "loop" in report.codes
        assert report.findings[0].evidence["count"] == 3

    def test_distinct_calls_are_not_a_loop(self):
        # Same tool but different params each time -> not a loop.
        view = _view(
            [
                _tool_run_step("search", resolved={"q": "a"}),
                _tool_run_step("search", resolved={"q": "b"}),
                _tool_run_step("search", resolved={"q": "c"}),
            ]
        )
        assert verify_trajectory(view).passed

    def test_detects_cascading_tool_errors(self):
        view = _view(
            [
                _tool_run_step("fetch", state="error", errors=[{"body": "x"}]),
                _tool_run_step("fetch", state="error", errors=[{"body": "x"}]),
                _tool_run_step("fetch", state="error", errors=[{"body": "x"}]),
            ]
        )
        report = verify_trajectory(view)
        assert "cascading_tool_errors" in report.codes
        assert report.findings[0].evidence["count"] == 3

    def test_detects_agent_error(self):
        view = _view([_tool_run_step("search"), _agent_error_step()])
        report = verify_trajectory(view)
        assert "agent_error" in report.codes

    def test_detects_missing_required_tool(self):
        view = _view([_tool_run_step("search")])
        report = verify_trajectory(view, required_tools=["search", "summarize"])
        assert "missing_required_tools" in report.codes
        missing = [f for f in report.findings if f.code == "missing_required_tools"]
        assert missing[0].evidence["tool_id"] == "summarize"

    def test_recompute_total_and_result_mismatch(self):
        view = _view(
            [
                _tool_run_step("sum", output={"value": 3}),
                _tool_run_step("sum", output={"value": 4, "extra": {"n": 5}}),
            ]
        )
        # 3 + 4 + 5 = 12 recomputed from the tool outputs.
        assert recompute_total(view.results) == 12.0

        # A matching stated total does not trip; a wrong one does.
        assert verify_trajectory(view, stated_total=12.0).passed
        bad = verify_trajectory(view, stated_total=99.0)
        assert "result_mismatch" in bad.codes
        assert bad.findings[0].evidence["recomputed_total"] == 12.0

    def test_verify_task_composes_async_agent_path(self, async_agent):
        # The verifier must pull telemetry through the real AsyncAgent method.
        view = _view(
            [
                _tool_run_step("search", resolved={"q": "same"}),
                _tool_run_step("search", resolved={"q": "same"}),
                _tool_run_step("search", resolved={"q": "same"}),
            ]
        )
        async_agent.view_task_steps = AsyncMock(return_value=view)

        report = asyncio.run(
            verify_task(async_agent, "conv-1", required_tools=["search"])
        )

        async_agent.view_task_steps.assert_awaited_once_with("conv-1")
        assert "loop" in report.codes

    def test_verify_and_repair_reruns_on_flag(self, async_agent):
        view = _view(
            [_tool_run_step("fetch", state="error") for _ in range(4)]
        )
        async_agent.view_task_steps = AsyncMock(return_value=view)
        async_agent.rerun_task = AsyncMock(
            return_value=MagicMock(conversation_id="conv-1")
        )

        report, rerun = asyncio.run(verify_and_repair(async_agent, "conv-1"))

        assert "cascading_tool_errors" in report.codes
        assert rerun is not None
        async_agent.rerun_task.assert_awaited_once_with("conv-1")

    def test_verify_and_repair_leaves_healthy_run_alone(self, async_agent):
        view = _view([_tool_run_step("search", output={"value": 1})])
        async_agent.view_task_steps = AsyncMock(return_value=view)
        async_agent.rerun_task = AsyncMock()

        report, rerun = asyncio.run(verify_and_repair(async_agent, "conv-1"))

        assert report.passed
        assert rerun is None
        async_agent.rerun_task.assert_not_awaited()
