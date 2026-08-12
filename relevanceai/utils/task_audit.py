"""Read-only multidimensional auditing for agent task traces.

Adapted from the A2E (Agent Auditing Engine) paper,
https://arxiv.org/abs/2608.07346v1, which argues that correctness alone is
too coarse a signal for agent harnesses and proposes a suite of
multidimensional metrics instead: execution efficiency, tool use, task
planning, and error recovery.

This SDK is a thin REST client with no local agent runtime, so A2E's
harness-runner core and its automatically instrumented Monitor cannot be
hosted here (they live server-side). What ports cleanly onto the SDK is
A2E's *metrics scheme* plus its task I/O contract: every step an agent
runs for a task is already exposed through ``Agent.view_task_steps`` as a
``TaskView`` of ``TaskStep`` records. This module walks that trace
read-only and computes A2E-style audit metrics over the very tools the
agent invoked -- the same tools created or cloned via
``create_tool``/``clone_tool``.

This is an inspired-experiment port (Mode 3): the paper's full evaluation
engine is out of reach, but its central idea -- that auditing tool use,
error recovery, efficiency, and planning characterizes a run more
richly than a single pass/fail -- is applied to the SDK's own task
lifecycle as a read-only utility.

Public surface:
    - :func:`audit_task_view` -- audit a ``TaskView`` from ``view_task_steps``.
    - :func:`audit_steps`     -- audit a bare iterable of steps (testing/ replays).
    - :class:`AuditReport`    -- the structured result, with ``summary``/``to_dict``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from relevanceai.types.task import TaskView

# Content discriminators for the ``TaskStep.content`` union.
_USER_MESSAGE = "user-message"
_AGENT_MESSAGE = "agent-message"
_TOOL_RUN = "tool-run"
_AGENT_ERROR = "agent-error"

# Tool-run states (members of ``ToolRunState``) we care about.
_FINISHED = "finished"
_ERROR = "error"


@dataclass
class ToolUseMetrics:
    """How the agent used tools during the task (A2E "tool use" axis)."""

    tool_calls: int = 0
    distinct_tools: int = 0
    tool_distribution: Dict[str, int] = field(default_factory=dict)
    confirmations_required: int = 0
    confirmations_approved: int = 0


@dataclass
class ErrorRecoveryMetrics:
    """Failures observed and how often the run continued past them
    (A2E "error recovery" axis)."""

    tool_errors: int = 0
    agent_errors: int = 0
    total_errors: int = 0
    recovered: int = 0
    recovery_rate: float = 0.0
    error_rate: float = 0.0


@dataclass
class EfficiencyMetrics:
    """Throughput and overhead signals (A2E "execution efficiency" axis)."""

    steps_total: int = 0
    tool_run_success_rate: float = 0.0
    planning_overhead: float = 0.0
    duration_seconds: Optional[float] = None


@dataclass
class PlanningMetrics:
    """Structure of the agent's reasoning versus acting (A2E "task
    planning" axis)."""

    user_messages: int = 0
    agent_messages: int = 0
    reasoning_steps: int = 0
    tool_call_ratio: float = 0.0


@dataclass
class AuditReport:
    """Multidimensional audit result for a single task trace."""

    tool_use: ToolUseMetrics = field(default_factory=ToolUseMetrics)
    error_recovery: ErrorRecoveryMetrics = field(default_factory=ErrorRecoveryMetrics)
    efficiency: EfficiencyMetrics = field(default_factory=EfficiencyMetrics)
    planning: PlanningMetrics = field(default_factory=PlanningMetrics)

    def summary(self) -> str:
        """Compact multi-line human-readable summary of the audit."""
        tu, er, eff, pl = (
            self.tool_use,
            self.error_recovery,
            self.efficiency,
            self.planning,
        )
        lines = [
            "steps={steps} | tools={calls} (distinct={distinct}) | "
            "success_rate={sr:.2f}".format(
                steps=eff.steps_total,
                calls=tu.tool_calls,
                distinct=tu.distinct_tools,
                sr=eff.tool_run_success_rate,
            ),
            "errors={total} (tool={te}, agent={ae}) | recovered={rec} | "
            "recovery_rate={rr:.2f}".format(
                total=er.total_errors,
                te=er.tool_errors,
                ae=er.agent_errors,
                rec=er.recovered,
                rr=er.recovery_rate,
            ),
            "planning: user_msg={u} agent_msg={a} tool_call_ratio={tcr:.2f}".format(
                u=pl.user_messages,
                a=pl.agent_messages,
                tcr=pl.tool_call_ratio,
            ),
        ]
        if eff.duration_seconds is not None:
            lines.append("duration={:.1f}s".format(eff.duration_seconds))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Nested-dict view of the report (for JSON logging / export)."""
        return asdict(self)


def audit_task_view(task_view: TaskView) -> AuditReport:
    """Audit a ``TaskView`` returned by ``Agent.view_task_steps``.

    Walks ``task_view.results`` read-only and returns an
    :class:`AuditReport`. Defensive by design: a trace with unrecognized
    or partially populated steps still yields a report rather than
    raising, since the audit must not break the surrounding pipeline.
    """
    return _audit_steps(list(task_view.results))


def audit_steps(steps: Iterable[Any]) -> AuditReport:
    """Audit a bare iterable of task steps.

    Lower-level entry point for testing or for auditing pre-collected /
    replayed traces that are not wrapped in a ``TaskView``.
    """
    return _audit_steps(list(steps))


def _audit_steps(steps: List[Any]) -> AuditReport:
    tool_use = ToolUseMetrics()
    error_recovery = ErrorRecoveryMetrics()
    planning = PlanningMetrics()

    n_finished = 0
    n_error_state = 0
    finished_step_indices: List[int] = []
    error_step_indices: List[int] = []
    timestamps: List[datetime] = []

    for idx, step in enumerate(steps):
        content = _safe_getattr(step, "content")
        if content is None:
            continue
        kind = _safe_getattr(content, "type")

        if kind == _TOOL_RUN:
            tool_use.tool_calls += 1
            config = _safe_getattr(content, "tool_config")
            tool_id = (
                _safe_getattr(config, "id")
                or _safe_getattr(config, "title")
                or "unknown"
            )
            tool_use.tool_distribution[tool_id] = (
                tool_use.tool_distribution.get(tool_id, 0) + 1
            )

            if _safe_getattr(content, "requires_confirmation"):
                tool_use.confirmations_required += 1
                confirmation = _safe_getattr(content, "confirmation")
                if _safe_getattr(confirmation, "confirmed"):
                    tool_use.confirmations_approved += 1

            state = _safe_getattr(content, "tool_run_state")
            state_val = _coerce_state(state)

            if state_val == _FINISHED:
                n_finished += 1
                finished_step_indices.append(idx)
            elif state_val == _ERROR:
                n_error_state += 1

            # A tool run "had an error" if it errored outright OR carried a
            # non-empty errors list (catches soft failures). These feed the
            # recovery dimension; success_rate above is state-based only.
            if state_val == _ERROR or _has_errors(content):
                error_recovery.tool_errors += 1
                error_step_indices.append(idx)

        elif kind == _AGENT_ERROR:
            error_recovery.agent_errors += _count_errors(content)
        elif kind == _USER_MESSAGE:
            planning.user_messages += 1
        elif kind == _AGENT_MESSAGE:
            planning.agent_messages += 1

        ts = _parse_timestamp(_safe_getattr(step, "insert_date_"))
        if ts is not None:
            timestamps.append(ts)

    tool_use.distinct_tools = len(tool_use.tool_distribution)

    # Recovery proxy: an erroring tool run is "recovered" if any later tool
    # run in the same trace reached a finished state (the agent retried and
    # a subsequent tool call succeeded).
    for err_idx in error_step_indices:
        if any(j > err_idx for j in finished_step_indices):
            error_recovery.recovered += 1

    error_recovery.total_errors = (
        error_recovery.tool_errors + error_recovery.agent_errors
    )
    if error_recovery.tool_errors:
        error_recovery.recovery_rate = (
            error_recovery.recovered / error_recovery.tool_errors
        )

    steps_total = len(steps)
    error_recovery.error_rate = (
        error_recovery.total_errors / steps_total if steps_total else 0.0
    )

    decided = n_finished + n_error_state
    efficiency = EfficiencyMetrics(
        steps_total=steps_total,
        tool_run_success_rate=(n_finished / decided) if decided else 0.0,
        planning_overhead=(
            (steps_total - tool_use.tool_calls) / tool_use.tool_calls
            if tool_use.tool_calls
            else 0.0
        ),
        duration_seconds=_span_seconds(timestamps),
    )

    planning.reasoning_steps = planning.agent_messages
    planning.tool_call_ratio = (
        tool_use.tool_calls / steps_total if steps_total else 0.0
    )

    return AuditReport(
        tool_use=tool_use,
        error_recovery=error_recovery,
        efficiency=efficiency,
        planning=planning,
    )


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    """getattr that also tolerates dict-shaped objects (and None)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _coerce_state(state: Any) -> str:
    """Normalize a ToolRunState enum or raw string into its string value."""
    if state is None:
        return ""
    value = _safe_getattr(state, "value", state)
    return value if isinstance(value, str) else str(value)


def _has_errors(content: Any) -> bool:
    return bool(_safe_getattr(content, "errors"))


def _count_errors(content: Any) -> int:
    errors = _safe_getattr(content, "errors")
    return len(errors) if errors else 0


def _parse_timestamp(raw: Any) -> Optional[datetime]:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _span_seconds(timestamps: List[datetime]) -> Optional[float]:
    if len(timestamps) < 2:
        return None
    try:
        return (max(timestamps) - min(timestamps)).total_seconds()
    except TypeError:
        # Mixed naive/aware timestamps cannot be compared; skip duration.
        return None
