import pytest
from unittest.mock import MagicMock

# Imports from NON-NEW modules exercise the integration: the monitor consumes
# the real TaskView/TaskStep telemetry produced by the existing Agent call site.
from relevanceai.resources.agent import Agent
from relevanceai.types.task import TaskStep, TaskView

from relevanceai.monitoring import FailureMonitor, extract_step_features


def _step(index, content):
    return TaskStep.model_validate(
        {
            "item_id": f"step-{index}",
            "insert_date_": "2026-01-01T00:00:00",
            "is_expanded_by_default": False,
            "is_in_hidden_group": False,
            "content": content,
        }
    )


def _tool_run(tool_id, title="Search", state="finished", errors=None):
    content = {
        "type": "tool-run",
        "tool_run_state": state,
        "tool_config": {
            "type": "tool",
            "title": title,
            "description": "a tool",
            "id": tool_id,
            "version": "1",
            "params_schema": {},
        },
        "action_details": {"action_request_id": "ar", "action": "search"},
        "requires_confirmation": False,
        "params": {"valid": False, "json": "{}"},
        "original_message_ids": {"action-request": "ar"},
    }
    if errors is not None:
        content["errors"] = errors
    return content


def _agent_message(text="done"):
    return {
        "type": "agent-message",
        "text": text,
        "original_message_ids": {"agent": "a"},
    }


def _agent_error():
    return {
        "type": "agent-error",
        "errors": [{"body": "fatal"}],
        "original_message_ids": {"agent-error": "a"},
    }


def _task_view(contents):
    return TaskView(results=[_step(i, c) for i, c in enumerate(contents)])


class TestExtractStepFeatures:
    def test_tool_run_features(self):
        step = _step(0, _tool_run("t1", state="error", errors=[{"body": "boom"}]))
        feat = extract_step_features(step, 0)
        assert feat.kind == "tool-run"
        assert feat.tool_id == "t1"
        assert feat.tool_title == "Search"
        assert feat.is_error is True
        assert feat.error_count == 1

    def test_agent_error_features(self):
        step = _step(0, _agent_error())
        feat = extract_step_features(step, 0)
        assert feat.kind == "agent-error"
        assert feat.is_error is True
        assert feat.error_count == 1


class TestFailureMonitorDetectors:
    def test_loop_detected_after_threshold(self):
        # Same tool called 3 times in a row -> looping.
        contents = [_tool_run("t1"), _tool_run("t1"), _tool_run("t1")]
        report = FailureMonitor(loop_threshold=3).analyze(_task_view(contents))
        kinds = [a.kind for a in report.alarms]
        assert "loop" in kinds
        assert report.healthy is False

    def test_no_loop_below_threshold(self):
        contents = [_tool_run("t1"), _tool_run("t1")]  # only 2 repeats
        report = FailureMonitor(loop_threshold=3).analyze(_task_view(contents))
        assert "loop" not in [a.kind for a in report.alarms]

    def test_error_cascade_detected(self):
        contents = [
            _tool_run("t1", state="error", errors=[{"body": "e"}]),
            _tool_run("t1", state="error", errors=[{"body": "e"}]),
            _tool_run("t1", state="error", errors=[{"body": "e"}]),
        ]
        report = FailureMonitor(error_cascade_threshold=3).analyze(_task_view(contents))
        assert "error-cascade" in [a.kind for a in report.alarms]

    def test_missing_required_tool_flagged(self):
        contents = [_tool_run("t1")]
        monitor = FailureMonitor(required_tools=["required-tool"])
        report = monitor.analyze(_task_view(contents))
        messages = [a.message for a in report.alarms]
        assert any("required-tool" in m for m in messages)

    def test_terminal_agent_error_flagged(self):
        report = FailureMonitor().analyze(_task_view([_agent_error()]))
        assert any(
            "terminal agent-error" in a.message for a in report.alarms
        )

    def test_healthy_run_has_no_alarms(self):
        contents = [_tool_run("t1"), _tool_run("t2"), _agent_message()]
        report = FailureMonitor().analyze(_task_view(contents))
        assert report.alarms == []
        assert report.healthy is True
        assert bool(report) is False


class TestMonitorIntegration:
    """Exercises the wiring through the existing Agent.view_task_steps call site."""

    @pytest.fixture
    def mock_client(self):
        return MagicMock()

    @pytest.fixture
    def agent(self, mock_client):
        return Agent(
            client=mock_client,
            agent_id="test-agent",
            name="Test Agent",
            _id="agent-id-123",
            project="default-project",
        )

    def test_monitor_calls_view_task_steps_and_flags_loop(self, agent):
        failing_view = _task_view(
            [_tool_run("t1"), _tool_run("t1"), _tool_run("t1")]
        )
        agent.view_task_steps = MagicMock(return_value=failing_view)

        report = FailureMonitor(loop_threshold=3).monitor(agent, "conv-1")

        agent.view_task_steps.assert_called_once_with("conv-1")
        assert "loop" in [a.kind for a in report.alarms]

    def test_monitor_reports_healthy_for_clean_run(self, agent):
        agent.view_task_steps = MagicMock(
            return_value=_task_view([_tool_run("t1"), _agent_message()])
        )
        report = FailureMonitor().monitor(agent, "conv-1")
        assert report.healthy is True

    def test_repair_calls_rerun_task(self, agent):
        agent.rerun_task = MagicMock(return_value={"conversation_id": "conv-1"})
        monitor = FailureMonitor()
        monitor.repair(agent, "conv-1")
        agent.rerun_task.assert_called_once_with("conv-1")
