"""Integration tests for the A2E-inspired task audit.

These build real ``TaskView`` / ``TaskStep`` objects from the (non-new)
``relevanceai.types.task`` module and feed them to ``audit_task_view`` /
``audit_steps`` from the new ``relevanceai.utils.task_audit`` module,
proving the audit is wired to the SDK's actual task-lifecycle types.
"""

import pytest

# Imported from a NON-NEW module: the existing task type definitions.
from relevanceai.types.task import (
    ActionDetails,
    By,
    Confirmation,
    Content,
    Content1,
    Content2,
    Content3,
    Error,
    ToolConfig,
    ToolRunState,
    TaskStep,
    TaskView,
)

# The new capability under test.
from relevanceai.utils.task_audit import AuditReport, audit_steps, audit_task_view


def _user_message():
    return Content.model_construct(
        type="user-message",
        text="research the company",
    )


def _agent_message():
    return Content1.model_construct(type="agent-message", text="thinking...")


def _tool_run(tool_id, state, *, errors=None, requires_confirmation=False,
              confirmed=None):
    config = ToolConfig.model_construct(
        type="tool",
        title=tool_id,
        description="",
        id=tool_id,
        version="1",
        params_schema={},
    )
    confirmation = None
    if confirmed is not None:
        confirmation = Confirmation.model_construct(confirmed=confirmed, by=By.user)
    return Content2.model_construct(
        type="tool-run",
        tool_run_state=ToolRunState(state),
        tool_config=config,
        action_details=ActionDetails.model_construct(
            action_request_id="r", action="a"
        ),
        requires_confirmation=requires_confirmation,
        confirmation=confirmation,
        errors=errors,
    )


def _agent_error(n=1):
    return Content3.model_construct(type="agent-error", errors=[Error()] * n)


def _step(content, ts):
    return TaskStep.model_construct(
        item_id=ts,
        insert_date_=ts,
        is_expanded_by_default=False,
        is_in_hidden_group=False,
        content=content,
    )


def _build_trace():
    # A run where: user asks -> agent reasons -> tool A succeeds ->
    # tool B errors -> tool A retried and succeeds (recovery) ->
    # agent explains -> agent-level error.
    return [
        _step(_user_message(), "2024-01-01T00:00:00"),
        _step(_agent_message(), "2024-01-01T00:00:10"),
        _step(_tool_run("A", "finished"), "2024-01-01T00:00:20"),
        _step(_tool_run("B", "error", errors=[Error()]), "2024-01-01T00:00:30"),
        _step(_tool_run("A", "finished"), "2024-01-01T00:00:40"),
        _step(_agent_message(), "2024-01-01T00:00:50"),
        _step(_agent_error(n=2), "2024-01-01T00:01:00"),
    ]


class TestTaskAudit:
    def test_audit_task_view_returns_report(self):
        report = audit_task_view(TaskView.model_construct(results=_build_trace()))
        assert isinstance(report, AuditReport)

    def test_tool_use_metrics(self):
        report = audit_steps(_build_trace())
        tu = report.tool_use
        assert tu.tool_calls == 3
        assert tu.distinct_tools == 2
        assert tu.tool_distribution == {"A": 2, "B": 1}
        assert tu.confirmations_required == 0
        assert tu.confirmations_approved == 0

    def test_error_recovery_metrics(self):
        report = audit_steps(_build_trace())
        er = report.error_recovery
        # tool B errored once (state=error + errors list); 2 agent errors.
        assert er.tool_errors == 1
        assert er.agent_errors == 2
        assert er.total_errors == 3
        # B's error was followed by a finished tool run -> recovered.
        assert er.recovered == 1
        assert er.recovery_rate == pytest.approx(1.0)
        # 3 errors across 7 steps.
        assert er.error_rate == pytest.approx(3 / 7)

    def test_efficiency_and_planning_metrics(self):
        report = audit_steps(_build_trace())
        eff = report.efficiency
        pl = report.planning
        assert eff.steps_total == 7
        # 2 finished, 1 error -> 2/3.
        assert eff.tool_run_success_rate == pytest.approx(2 / 3)
        # (7 steps - 3 tool calls) / 3 tool calls.
        assert eff.planning_overhead == pytest.approx(4 / 3)
        # Span across the 7 timestamps == 60s.
        assert eff.duration_seconds == pytest.approx(60.0)
        assert pl.user_messages == 1
        assert pl.agent_messages == 2
        assert pl.reasoning_steps == 2
        assert pl.tool_call_ratio == pytest.approx(3 / 7)

    def test_confirmation_flow_is_counted(self):
        steps = [
            _step(_tool_run("C", "finished", requires_confirmation=True,
                            confirmed=True), "2024-01-01T00:00:00"),
            _step(_tool_run("C", "pending", requires_confirmation=True,
                            confirmed=False), "2024-01-01T00:00:01"),
        ]
        report = audit_steps(steps)
        assert report.tool_use.confirmations_required == 2
        assert report.tool_use.confirmations_approved == 1

    def test_unrecovered_error_is_not_recovered(self):
        # Erroring tool run is the LAST step -> nothing recovers it.
        steps = [
            _step(_tool_run("A", "finished"), "2024-01-01T00:00:00"),
            _step(_tool_run("B", "error", errors=[Error()]), "2024-01-01T00:00:01"),
        ]
        report = audit_steps(steps)
        assert report.error_recovery.recovered == 0
        assert report.error_recovery.recovery_rate == 0.0

    def test_empty_trace_is_safe(self):
        report = audit_steps([])
        assert report.efficiency.steps_total == 0
        assert report.tool_use.tool_calls == 0
        assert report.to_dict()["error_recovery"]["total_errors"] == 0
        assert "steps=0" in report.summary()

    def test_dict_shaped_content_is_tolerated(self):
        # Defensive path: a step whose content is a plain dict must not raise.
        step = TaskStep.model_construct(
            item_id="x",
            insert_date_="not-a-date",
            is_expanded_by_default=False,
            is_in_hidden_group=False,
            content={"type": "tool-run", "tool_config": {"id": "D"},
                     "tool_run_state": "finished", "requires_confirmation": False},
        )
        report = audit_steps([step])
        assert report.tool_use.tool_calls == 1
        assert report.tool_use.tool_distribution == {"D": 1}
        # Unparseable timestamp -> duration stays None, no crash.
        assert report.efficiency.duration_seconds is None
