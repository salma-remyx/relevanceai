"""Deterministic trajectory verification for Relevance AI agent runs.

Adapted from "Real-Time Detection and Repair of LLM Agent Failures"
(arXiv:2608.02464). The paper stacks two detectors over an agent's
observable step telemetry:

  1. A *trained* one-class monitor (echo-state-network ensemble with CUSUM
     alarms). It performs well but carries two burdens: it needs a
     per-deployment "healthy" null to calibrate against (it does not
     transfer cold), and it has a residual false-alarm rate.
  2. A *parameter-free* deterministic-verification layer that recomputes a
     run's stated result from the tool outputs it actually received and
     confirms every required call was made. It carries neither burden --
     in the paper it trips on 0 of 1825 healthy episodes.

This module implements layer (2) at full fidelity and replaces layer (1)
(the learned estimator) with parameter-free trajectory-shape detectors
that need no healthy-run training -- a target-native substitute, since
this control-plane client has no model-training surface. The detectors
consume the ``AsyncAgent.view_task_steps`` telemetry
(``TaskView.results: List[TaskStep]``) and surface the same failure
classes the paper targets -- loops, cascading tool errors, missing
coverage, and fabricated results -- with no second model call.

Detection is closed into repair via :func:`verify_and_repair`, which
re-triggers a flagged run through the agent's native ``rerun_task``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from ..types.task import TaskView


class VerificationFinding(BaseModel):
    """A single deterministic failure signal raised over a trajectory."""

    code: str
    severity: str = "error"
    message: str
    evidence: Dict[str, Any] = Field(default_factory=dict)


class VerificationReport(BaseModel):
    """The result of deterministically verifying an agent run."""

    findings: List[VerificationFinding] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when no failure signal was raised (the run is healthy)."""
        return not self.findings

    @property
    def codes(self) -> List[str]:
        return [finding.code for finding in self.findings]

    def __bool__(self) -> bool:
        # Truthy when the run PASSED, so ``if report:`` reads naturally.
        return self.passed

    def __repr__(self) -> str:
        if self.passed:
            return "VerificationReport(passed)"
        return "VerificationReport(failed: " + ", ".join(self.codes) + ")"


# --- telemetry accessors (duck-typed over TaskStep.content) ---------------


def _as_steps(trajectory: Any) -> List[Any]:
    """Normalize a TaskView, a list of steps, or any iterable of steps."""
    if isinstance(trajectory, TaskView):
        return list(trajectory.results)
    results = getattr(trajectory, "results", None)
    if results is not None:
        return list(results)
    return list(trajectory)


def _content(step: Any) -> Any:
    return getattr(step, "content", None)


def _content_type(step: Any) -> Optional[str]:
    content = _content(step)
    return getattr(content, "type", None)


def _tool_id(content: Any) -> Optional[str]:
    tool_config = getattr(content, "tool_config", None)
    return getattr(tool_config, "id", None)


def _resolved_params(content: Any) -> Any:
    """The params the tool was actually run with, falling back to the declared json."""
    params = getattr(content, "params", None)
    if params is None:
        return None
    resolved = getattr(params, "resolved", None)
    if resolved is not None:
        return resolved
    return getattr(params, "json_", None) or getattr(params, "json", None)


def _run_errored(content: Any) -> bool:
    """A tool-run errored if its state is 'error' or it carries error bodies."""
    state = getattr(content, "tool_run_state", None)
    if getattr(state, "value", state) == "error":
        return True
    return bool(getattr(content, "errors", None))


def _signature(value: Any) -> str:
    """A stable string signature for resolved params (loop detection)."""
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(value)


def _walk_numbers(value: Any) -> Iterable[float]:
    """Yield every numeric leaf (int/float, not bool) in a nested structure."""
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield float(value)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_numbers(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_numbers(item)


def recompute_total(steps: Sequence[Any]) -> float:
    """Sum every number the (non-errored) tool-run steps received as output.

    This is the deterministic-verification recompute from the paper: a run's
    'total' is recomputed from the tool results it really got, with no model
    in the loop.
    """
    total = 0.0
    for step in steps:
        if _content_type(step) != "tool-run":
            continue
        content = _content(step)
        if _run_errored(content):
            continue
        total += sum(_walk_numbers(getattr(content, "output", None)))
    return total


def verify_trajectory(
    steps: Any,
    *,
    required_tools: Optional[Sequence[str]] = None,
    repeat_threshold: int = 3,
    error_run_threshold: int = 3,
    stated_total: Optional[float] = None,
    total_tolerance: float = 1e-9,
) -> VerificationReport:
    """Deterministically verify an agent trajectory from its step telemetry.

    Raises (with no model call and no per-deployment calibration):

    * ``loop`` -- a tool call repeated with identical resolved params
      (``>= repeat_threshold`` times).
    * ``cascading_tool_errors`` -- ``>= error_run_threshold`` consecutive
      tool calls that errored.
    * ``agent_error`` -- a terminal agent-error step.
    * ``missing_required_tools`` -- a tool in ``required_tools`` that was
      never called (the paper's coverage check).
    * ``result_mismatch`` -- a ``stated_total`` that the non-errored tool
      outputs do not sum to (the paper's recompute check).

    ``steps`` may be a :class:`~relevanceai.types.task.TaskView` or any
    iterable of ``TaskStep``.
    """
    findings: List[VerificationFinding] = []
    step_list = _as_steps(steps)
    required_tools = list(required_tools or [])

    tool_run_indices = [
        index for index, step in enumerate(step_list)
        if _content_type(step) == "tool-run"
    ]

    # agent-error (terminal) ------------------------------------------------
    for index, step in enumerate(step_list):
        if _content_type(step) == "agent-error":
            findings.append(VerificationFinding(
                code="agent_error",
                message="Trajectory contains a terminal agent-error step.",
                evidence={"index": index},
            ))

    # loop (identical tool call repeated) -----------------------------------
    signatures: Dict[Tuple[Optional[str], str], List[int]] = {}
    for index in tool_run_indices:
        content = _content(step_list[index])
        key = (_tool_id(content), _signature(_resolved_params(content)))
        signatures.setdefault(key, []).append(index)
    for (tool_id, _sig), indices in signatures.items():
        if len(indices) >= repeat_threshold:
            findings.append(VerificationFinding(
                code="loop",
                message=(
                    f"Tool '{tool_id}' called {len(indices)} times with "
                    f"identical parameters (>= {repeat_threshold})."
                ),
                evidence={
                    "tool_id": tool_id,
                    "count": len(indices),
                    "indices": indices,
                },
            ))

    # cascading tool errors (longest run of consecutive errored tool calls) -
    errored = [_run_errored(_content(step_list[index])) for index in tool_run_indices]
    streak = 0
    run_start = 0
    best_len = 0
    best_start = -1
    for position, bad in enumerate(errored):
        if bad:
            if streak == 0:
                run_start = position
            streak += 1
            if streak > best_len:
                best_len = streak
                best_start = run_start
        else:
            streak = 0
    if best_len >= error_run_threshold:
        findings.append(VerificationFinding(
            code="cascading_tool_errors",
            message=(
                f"{best_len} consecutive tool calls errored "
                f"(>= {error_run_threshold})."
            ),
            evidence={
                "count": best_len,
                "start_index": (
                    tool_run_indices[best_start] if best_start >= 0 else None
                ),
            },
        ))

    # coverage (every required tool was actually called) -------------------
    if required_tools:
        seen = {
            _tool_id(_content(step_list[index]))
            for index in tool_run_indices
        }
        for required in required_tools:
            if required not in seen:
                findings.append(VerificationFinding(
                    code="missing_required_tools",
                    message=f"Required tool '{required}' was never called.",
                    evidence={"tool_id": required},
                ))

    # recompute (stated total vs. sum of tool outputs) ---------------------
    if stated_total is not None:
        recomputed = recompute_total(step_list)
        if abs(recomputed - stated_total) > total_tolerance:
            findings.append(VerificationFinding(
                code="result_mismatch",
                message=(
                    f"Stated total {stated_total} does not match the "
                    f"{recomputed} recomputed from tool outputs."
                ),
                evidence={
                    "stated_total": stated_total,
                    "recomputed_total": recomputed,
                    "difference": recomputed - stated_total,
                },
            ))

    return VerificationReport(findings=findings)


# --- async wrappers composing the AsyncAgent path -------------------------


async def verify_task(agent: Any, conversation_id: str, **kwargs: Any) -> VerificationReport:
    """Fetch a run's step telemetry and verify it deterministically.

    Wires ``AsyncAgent.view_task_steps`` -> ``TaskView.results`` into
    :func:`verify_trajectory`.
    """
    task_view: TaskView = await agent.view_task_steps(conversation_id)
    return verify_trajectory(task_view, **kwargs)


# Finding codes that, by default, justify a live re-run (the paper's repair).
DEFAULT_RERUN_CODES = frozenset({
    "loop",
    "cascading_tool_errors",
    "agent_error",
    "result_mismatch",
    "missing_required_tools",
})


async def verify_and_repair(
    agent: Any,
    conversation_id: str,
    *,
    rerun_codes: Optional[Iterable[str]] = None,
    max_reruns: int = 1,
    **kwargs: Any,
) -> Tuple[VerificationReport, Any]:
    """Verify a run and, if it is flagged, re-run it via ``agent.rerun_task``.

    Closes detection into repair as in the paper: a flagged run is
    re-triggered using the agent's native rerun primitive. Returns
    ``(report, rerun)`` where ``rerun`` is the value returned by
    ``rerun_task`` (e.g. a ``TriggeredTask``) or ``None`` when the run
    passed, the flagged codes are outside ``rerun_codes``, or the re-run
    budget is zero.
    """
    report = await verify_task(agent, conversation_id, **kwargs)
    rerun = None
    if not report.passed and max_reruns > 0:
        codes = set(rerun_codes) if rerun_codes is not None else DEFAULT_RERUN_CODES
        if any(code in codes for code in report.codes):
            rerun = await agent.rerun_task(conversation_id)
    return report, rerun
