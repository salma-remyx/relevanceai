"""Integration tests for the agent failure monitor.

These build real ``TaskView`` telemetry (the same ``**response.json()``
parse path used by ``Agent.view_task_steps``) from the non-new
``relevanceai.types.task`` module and run it through the monitor, proving
the detectors integrate with the SDK's actual step contract.
"""

from __future__ import annotations

from relevanceai.types.task import TaskView
from relevanceai.utils.agent_failure_monitor import (
    MonitorConfig,
    monitor_task_view,
)


def _tool_run_step(
    item_id: str,
    tool_id: str = "search-1",
    action: str = "/search",
    state: str = "finished",
    resolved: dict | None = None,
    errors: list | None = None,
    output: dict | None = None,
    title: str = "Search",
) -> dict:
    return {
        "item_id": item_id,
        "insert_date_": "2026-01-01T00:00:00Z",
        "is_in_hidden_group": False,
        "is_expanded_by_default": False,
        "content": {
            "type": "tool-run",
            "tool_run_state": state,
            "tool_config": {
                "type": "tool",
                "title": title,
                "description": "desc",
                "id": tool_id,
                "version": "latest",
                "params_schema": {},
            },
            "action_details": {"action_request_id": f"ar-{item_id}", "action": action},
            "requires_confirmation": False,
            "params": {
                "valid": True,
                "json": resolved or {},
                "resolved": resolved or {},
            },
            "errors": errors or [],
            "output": output or {},
            "original_message_ids": {"action-request": f"ar-{item_id}"},
        },
    }


def _agent_error_step(item_id: str) -> dict:
    return {
        "item_id": item_id,
        "insert_date_": "2026-01-01T00:00:00Z",
        "is_in_hidden_group": False,
        "is_expanded_by_default": False,
        "content": {
            "type": "agent-error",
            "errors": [{"body": "unrecoverable failure"}],
            "original_message_ids": {"agent-error": f"ae-{item_id}"},
        },
    }


def _task_view(steps: list) -> TaskView:
    return TaskView(results=steps)


def test_healthy_run_is_clean():
    steps = [
        _tool_run_step("s1", tool_id="search-1", resolved={"q": "a"}, output={"count": 4}),
        _tool_run_step(
            "s2", tool_id="send-1", action="/send", resolved={"to": "x"}, output={"count": 6}
        ),
    ]
    report = monitor_task_view(
        _task_view(steps),
        config=MonitorConfig(
            enable_coverage=True,
            required_tools={"search-1"},
            enable_total_verification=True,
            stated_total=10,
            total_key="count",
        ),
    )
    assert report.failed is False
    assert report.findings == []
    assert report.recommended_repair is None
    assert report.step_count == 2
    assert len(report.tool_calls) == 2


def test_loop_detected():
    # same tool + identical resolved args, 4 calls (default threshold 3)
    steps = [
        _tool_run_step(f"s{i}", tool_id="search-1", resolved={"q": "same"}) for i in range(4)
    ]
    report = monitor_task_view(_task_view(steps))
    assert report.failed is True
    loop = next(f for f in report.findings if f.detector == "loop")
    assert loop.evidence["count"] == 4
    assert report.recommended_repair is not None


def test_loop_ignores_different_args():
    # same tool but varying arguments -> not a loop
    steps = [
        _tool_run_step(f"s{i}", tool_id="search-1", resolved={"q": f"q{i}"}) for i in range(4)
    ]
    report = monitor_task_view(_task_view(steps))
    assert report.failed is False


def test_error_cascade_detected():
    steps = [
        _tool_run_step("s1", state="error", errors=[{"body": "timeout"}]),
        _tool_run_step("s2", state="error", errors=[{"body": "timeout"}]),
        _tool_run_step("s3", state="error", errors=[{"body": "timeout"}]),
    ]
    report = monitor_task_view(_task_view(steps))
    assert report.failed is True
    assert any(f.detector == "error_cascade" for f in report.findings)


def test_agent_error_step_counts_in_cascade():
    steps = [
        _tool_run_step("s1", state="error", errors=[{"body": "x"}]),
        _agent_error_step("s2"),
        _tool_run_step("s3", state="error", errors=[{"body": "x"}]),
    ]
    report = monitor_task_view(_task_view(steps))
    assert any(f.detector == "error_cascade" for f in report.findings)


def test_coverage_missing_required_tool():
    steps = [_tool_run_step("s1", tool_id="search-1")]
    report = monitor_task_view(
        _task_view(steps),
        config=MonitorConfig(
            enable_coverage=True, required_tools={"search-1", "summarize-1"}
        ),
    )
    assert report.failed is True
    cov = next(f for f in report.findings if f.detector == "coverage")
    assert "summarize-1" in cov.evidence["missing_tools"]
    assert cov.evidence["seen_tools"] == ["search-1"]


def test_total_mismatch_detected():
    steps = [
        _tool_run_step("s1", output={"count": 4}),
        _tool_run_step("s2", tool_id="send-1", action="/send", output={"count": 6}),
    ]
    report = monitor_task_view(
        _task_view(steps),
        config=MonitorConfig(
            enable_total_verification=True, stated_total=99, total_key="count"
        ),
    )
    assert report.failed is True
    mis = next(f for f in report.findings if f.detector == "total_mismatch")
    assert mis.evidence["recomputed"] == 10
    assert mis.evidence["stated_total"] == 99


def test_total_reconciles_when_correct():
    steps = [
        _tool_run_step("s1", output={"count": 4}),
        _tool_run_step("s2", tool_id="send-1", action="/send", output={"count": 6}),
    ]
    report = monitor_task_view(
        _task_view(steps),
        config=MonitorConfig(
            enable_total_verification=True, stated_total=10, total_key="count"
        ),
    )
    assert report.failed is False
