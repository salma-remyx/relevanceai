"""
Real-time agent failure detection and repair from step telemetry.

Adapted from "Real-Time Detection and Repair of LLM Agent Failures"
(arXiv:2608.02464). Maps the paper's three layers onto the SDK's
``Agent.view_task_steps`` -> ``TaskView`` contract:

1. Behavioural monitor: the paper's trained one-class echo-state-network
   ensemble with CUSUM alarms is a per-deployment artefact that does not
   transfer cold, so it is substituted by a *parameter-free* CUSUM over the
   same step-level signals -- tool-call repetition (looping) and error runs
   (tool-error cascades). Same signal, no training.
2. Deterministic verification: the paper's zero-false-positive layer at full
   fidelity -- terminal ``agent-error`` steps, tools stuck in ``error``
   (the SDK's ``UnrecoverableErrorType.max_tool_retries`` mode), missing
   required tools, and a claimed result no finished tool produced.
3. Repair: flagged runs are rolled back and re-run via ``rerun_task``.

Observable telemetry only -- no second LLM, no judge call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .resources.agent import Agent
from .types.task import TaskView, UnrecoverableErrorType


@dataclass
class StepFeatures:
    """A single task step reduced to the signals the detectors consume."""

    index: int
    kind: str  # 'user-message' | 'agent-message' | 'tool-run' | 'agent-error'
    tool_id: Optional[str] = None
    tool_title: Optional[str] = None
    is_error: bool = False
    error_count: int = 0
    requires_confirmation: bool = False


def extract_step_features(step: Any, index: int) -> StepFeatures:
    """Defensively reduce a ``TaskStep`` into a :class:`StepFeatures`.

    Works across the ``content`` union (user-message / agent-message /
    tool-run / agent-error) by reading attributes that every variant exposes,
    so it does not break when the telemetry payload grows new fields.
    """
    content = getattr(step, "content", None)
    kind = str(getattr(content, "type", "") or "")
    features = StepFeatures(index=index, kind=kind)

    if kind == "tool-run":
        tool_config = getattr(content, "tool_config", None)
        features.tool_id = getattr(tool_config, "id", None)
        features.tool_title = getattr(tool_config, "title", None)
        features.requires_confirmation = bool(
            getattr(content, "requires_confirmation", False)
        )
        run_state = getattr(content, "tool_run_state", None)
        # tool_run_state is a ToolRunState enum; .value is 'error' on failure.
        state_value = getattr(run_state, "value", run_state)
        errors = getattr(content, "errors", None) or []
        features.error_count = len(errors)
        features.is_error = state_value == "error" or features.error_count > 0
    elif kind == "agent-error":
        errors = getattr(content, "errors", None) or []
        features.error_count = len(errors)
        features.is_error = True

    return features


@dataclass
class Alarm:
    """A single flagged failure with enough context to drive repair."""

    kind: str  # 'loop' | 'error-cascade' | 'verification' | 'stall'
    severity: str  # 'low' | 'medium' | 'high'
    message: str
    step_indices: List[int] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)


@dataclass
class MonitoringReport:
    """Aggregate result of monitoring one task's telemetry."""

    alarms: List[Alarm] = field(default_factory=list)
    step_count: int = 0

    @property
    def healthy(self) -> bool:
        return not self.alarms

    @property
    def severities(self) -> List[str]:
        return [a.severity for a in self.alarms]

    def summary(self) -> str:
        if self.healthy:
            return f"healthy: no alarms across {self.step_count} steps"
        lines = [f"flagged: {len(self.alarms)} alarm(s) across {self.step_count} steps"]
        for alarm in self.alarms:
            lines.append(f"  [{alarm.severity}] {alarm.kind}: {alarm.message}")
        return "\n".join(lines)

    def __bool__(self) -> bool:  # truthy == flagged
        return not self.healthy


class FailureMonitor:
    """Detect mid-episode agent failures from step telemetry.

    The behavioural detectors (loop, error-cascade) are CUSUM accumulators
    over per-step signals -- the parameter-free proxy for the paper's trained
    echo-state-network ensemble. The verifier is deterministic and intended to
    carry no false alarms.
    """

    def __init__(
        self,
        loop_threshold: int = 3,
        error_cascade_threshold: int = 3,
        required_tools: Optional[List[str]] = None,
        cusum_drift: float = 0.0,
    ) -> None:
        # With the default drift of 0.0 the CUSUM statistic is simply the
        # run-length of consecutive identical tool calls / errors, so the
        # thresholds read as "N repeats in a row".
        self.loop_threshold = loop_threshold
        self.error_cascade_threshold = error_cascade_threshold
        self.required_tools = list(required_tools or [])
        self.cusum_drift = cusum_drift

    def analyze(self, task_view: TaskView) -> MonitoringReport:
        """Run every detector over a ``TaskView`` and return the aggregate."""
        steps = getattr(task_view, "results", []) or []
        features = [extract_step_features(step, i) for i, step in enumerate(steps)]
        report = MonitoringReport(step_count=len(features))
        report.alarms.extend(self._detect_loop(features))
        report.alarms.extend(self._detect_error_cascade(features))
        report.alarms.extend(self._verify_deterministic(features))
        return report

    def monitor(self, agent: Agent, conversation_id: str) -> MonitoringReport:
        """Fetch live telemetry via ``view_task_steps`` and analyse it.

        Integration point with the existing SDK: calls the agent's
        ``view_task_steps`` endpoint and feeds the ``TaskView`` to analyze.
        """
        task_view = agent.view_task_steps(conversation_id)
        return self.analyze(task_view)

    def repair(self, agent: Agent, conversation_id: str) -> Any:
        """Roll back and re-run a flagged task via the SDK's ``rerun_task``.

        Mirrors the paper's repair loop: re-run live for ~one extra model
        call rather than a judge pass.
        """
        return agent.rerun_task(conversation_id)

    def _detect_loop(self, features: List[StepFeatures]) -> List[Alarm]:
        alarms: List[Alarm] = []
        stat = 0.0
        prev_tool: Optional[str] = None
        run_indices: List[int] = []
        armed = False  # one alarm per contiguous looping run

        for feat in features:
            if feat.kind != "tool-run" or feat.tool_id is None:
                # A non-tool step breaks the repetition run.
                stat = 0.0
                prev_tool = None
                run_indices = []
                armed = False
                continue

            if feat.tool_id == prev_tool:
                stat = max(0.0, stat + 1.0 - self.cusum_drift)
                run_indices.append(feat.index)
            else:
                stat = max(0.0, 1.0 - self.cusum_drift)
                run_indices = [feat.index]
                armed = False
            prev_tool = feat.tool_id

            if (
                stat >= self.loop_threshold
                and len(run_indices) >= self.loop_threshold
                and not armed
            ):
                alarms.append(
                    Alarm(
                        kind="loop",
                        severity="high",
                        message=(
                            f"Tool '{feat.tool_title or feat.tool_id}' called "
                            f"{len(run_indices)} times in a row -- agent is "
                            "looping."
                        ),
                        step_indices=list(run_indices),
                        evidence={
                            "tool_id": feat.tool_id,
                            "repeats": len(run_indices),
                        },
                    )
                )
                armed = True
        return alarms

    def _detect_error_cascade(self, features: List[StepFeatures]) -> List[Alarm]:
        alarms: List[Alarm] = []
        stat = 0.0
        run_indices: List[int] = []
        armed = False

        for feat in features:
            if feat.is_error:
                stat = max(0.0, stat + 1.0 - self.cusum_drift)
                run_indices.append(feat.index)
                if stat >= self.error_cascade_threshold and not armed:
                    alarms.append(self._cascade_alarm(run_indices))
                    armed = True
            else:
                stat = 0.0
                run_indices = []
                armed = False
        return alarms

    def _cascade_alarm(self, run_indices: List[int]) -> Alarm:
        terminal = run_indices[-1]
        return Alarm(
            kind="error-cascade",
            severity="high",
            message=(
                f"{len(run_indices)} consecutive tool/agent errors ending at "
                f"step {terminal} -- tool-error cascade "
                f"({UnrecoverableErrorType.max_tool_retries.value})."
            ),
            step_indices=list(run_indices),
            evidence={"consecutive_errors": len(run_indices)},
        )

    # -- deterministic verifier (full fidelity, intended zero false alarms) --

    def _verify_deterministic(self, features: List[StepFeatures]) -> List[Alarm]:
        alarms: List[Alarm] = []

        # Terminal hard failure: an agent-error step means the run gave up.
        agent_errors = [f.index for f in features if f.kind == "agent-error"]
        if agent_errors:
            alarms.append(
                Alarm(
                    kind="verification",
                    severity="high",
                    message=(
                        "Run reached a terminal agent-error state -- unrecoverable."
                    ),
                    step_indices=agent_errors,
                    evidence={"agent_error_steps": agent_errors},
                )
            )

        # Tools stuck in the error state after retries.
        errored_tools = sorted(
            {f.tool_id for f in features if f.kind == "tool-run" and f.is_error}
        )
        if errored_tools:
            indices = [
                f.index
                for f in features
                if f.kind == "tool-run" and f.is_error and f.tool_id
            ]
            alarms.append(
                Alarm(
                    kind="verification",
                    severity="medium",
                    message=(
                        "Tool(s) ended in an error state: "
                        f"{', '.join(errored_tools)} "
                        f"({UnrecoverableErrorType.max_tool_retries.value})."
                    ),
                    step_indices=indices,
                    evidence={"errored_tools": errored_tools},
                )
            )

        # Coverage: every required tool must have been invoked.
        invoked = {
            f.tool_id for f in features if f.kind == "tool-run" and f.tool_id
        }
        missing = [t for t in self.required_tools if t not in invoked]
        if missing:
            alarms.append(
                Alarm(
                    kind="verification",
                    severity="medium",
                    message=f"Required tool(s) never invoked: {', '.join(missing)}.",
                    step_indices=[],
                    evidence={"missing_tools": missing},
                )
            )

        # Recompute the run's stated outcome from the tool results it received:
        # if the agent emitted a final agent-message (a claimed result) but no
        # tool-run ever finished successfully, the run claims work it did not
        # do -- the deterministic fabrication / silent-failure check.
        produced_output = any(
            f.kind == "tool-run" and not f.is_error for f in features
        )
        has_final_answer = any(f.kind == "agent-message" for f in features)
        if has_final_answer and features and not produced_output:
            alarms.append(
                Alarm(
                    kind="verification",
                    severity="high",
                    message=(
                        "Agent produced a final message but no tool run finished "
                        "successfully -- claimed result is unsupported by tool "
                        "output."
                    ),
                    step_indices=[f.index for f in features if f.kind == "agent-message"],
                    evidence={"produced_output": produced_output},
                )
            )

        return alarms


class AsyncFailureMonitor(FailureMonitor):
    """Async variant of :class:`FailureMonitor` for ``AsyncAgent`` telemetry."""

    async def analyze_async(self, task_view: TaskView) -> MonitoringReport:
        return self.analyze(task_view)

    async def monitor(self, agent: Agent, conversation_id: str) -> MonitoringReport:
        task_view = await agent.view_task_steps(conversation_id)
        return self.analyze(task_view)

    async def repair(self, agent: Agent, conversation_id: str) -> Any:
        return await agent.rerun_task(conversation_id)


__all__ = [
    "Alarm",
    "AsyncFailureMonitor",
    "FailureMonitor",
    "MonitoringReport",
    "StepFeatures",
    "extract_step_features",
]
