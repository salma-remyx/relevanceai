"""Lightweight, deterministic failure monitors for agent task steps.

Adapted from "Real-Time Detection and Repair of LLM Agent Failures"
(arXiv:2608.02464). The paper shows that a *deterministic verification*
layer over per-step agent telemetry -- recomputing a run's stated results
from the tool outputs it actually received and confirming every required
call was made -- catches the majority of agent failures at zero false
positives, without the per-deployment "healthy null" a learned monitor
needs. This module ports that deterministic layer against the SDK's own
``TaskView`` step telemetry.

Scope (Mode 2 adapted port). The paper's deterministic checks are
implemented at full fidelity. The auxiliary components that require
infrastructure the SDK does not host are intentionally *not* ported:

* the learned one-class echo-state-network ensemble with CUSUM alarms is
  replaced by parameter-free deterministic detectors (the paper's own
  deterministic layer is parameter-free and is its strongest result);
* the live repair loop (rollback + re-run) is reduced to a recommended
  repair action rather than an executor -- the SDK already exposes
  ``agent.rerun_task`` for the actual re-run.

All detectors consume a ``TaskView`` (as returned by
``agent.view_task_steps``) and return structured findings.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from ..types.task import TaskView

__all__ = [
    "Finding",
    "MonitorConfig",
    "MonitorReport",
    "monitor_task_view",
    "detect_loops",
    "detect_error_cascade",
    "check_required_coverage",
    "verify_stated_total",
]


# Repair hint is the only "action" the paper's repair loop collapses to here.
_REPAIR_HINT = (
    "rerun the flagged task -- the paper recovers ~45% of failures by "
    "rollback + re-run; the SDK exposes agent.rerun_task(conversation_id)"
)


@dataclass
class Finding:
    """A single failure signal raised by one detector."""

    detector: str
    severity: str  # "high" | "medium"
    message: str
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitorConfig:
    """Toggle and threshold the detectors run by :func:`monitor_task_view`."""

    enable_loops: bool = True
    max_identical_calls: int = 3
    enable_error_cascade: bool = True
    max_consecutive_errors: int = 3
    enable_coverage: bool = False
    required_tools: Set[str] = field(default_factory=set)
    required_actions: Set[str] = field(default_factory=set)
    enable_total_verification: bool = False
    stated_total: Optional[float] = None
    total_key: str = "count"
    total_agg: str = "sum"  # "sum" | "max"
    total_tolerance: float = 0.0


@dataclass
class MonitorReport:
    """Aggregate verdict over a task's step telemetry."""

    failed: bool
    findings: List[Finding]
    step_count: int
    tool_calls: List[Dict[str, Any]]
    recommended_repair: Optional[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "failed": self.failed,
            "step_count": self.step_count,
            "findings": [asdict(f) for f in self.findings],
            "tool_calls": self.tool_calls,
            "recommended_repair": self.recommended_repair,
        }


def _iter_tool_runs(steps: Any) -> Iterator[Tuple[int, Any]]:
    """Yield (index, content) for each ``tool-run`` step."""
    for idx, step in enumerate(steps or []):
        content = getattr(step, "content", None)
        if getattr(content, "type", None) == "tool-run":
            yield idx, content


def _stringify(obj: Any) -> str:
    """Stable serialization of a tool-call's arguments for dedup."""
    try:
        return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(obj)


def _call_signature(content: Any) -> Tuple[str, str]:
    """Identity of a tool call: (tool_id, serialized resolved arguments)."""
    tool_config = getattr(content, "tool_config", None)
    tool_id = getattr(tool_config, "id", "") or ""
    params = getattr(content, "params", None)
    resolved = getattr(params, "resolved", None)
    if not resolved:
        # Params.json_ is the declared arg map; Params1.json_ is an error string.
        resolved = getattr(params, "json_", None)
    return (tool_id, _stringify(resolved))


def _call_summary(content: Any) -> Dict[str, Any]:
    errors = getattr(content, "errors", None) or []
    state = getattr(getattr(content, "tool_run_state", None), "value", None)
    return {
        "tool_id": getattr(getattr(content, "tool_config", None), "id", None),
        "action": getattr(getattr(content, "action_details", None), "action", None),
        "state": state,
        "errored": bool(errors) or state == "error",
    }


def _step_errored(step: Any) -> bool:
    content = getattr(step, "content", None)
    ctype = getattr(content, "type", None)
    if ctype == "agent-error":
        return True
    if ctype == "tool-run":
        state = getattr(getattr(content, "tool_run_state", None), "value", None)
        if state == "error":
            return True
        if getattr(content, "errors", None):
            return True
    return False


def detect_loops(steps: Any, *, max_identical: int) -> List[Finding]:
    """Flag a tool called more than ``max_identical`` times with identical args.

    Corresponds to the paper's "they loop" failure mode.
    """
    if max_identical <= 0:
        return []
    counts: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for idx, content in _iter_tool_runs(steps):
        sig = _call_signature(content)
        rec = counts.setdefault(sig, {"count": 0, "tool_id": sig[0], "first": idx})
        rec["count"] += 1
    findings: List[Finding] = []
    for rec in counts.values():
        if rec["count"] > max_identical:
            findings.append(
                Finding(
                    detector="loop",
                    severity="high",
                    message=(
                        f"Tool '{rec['tool_id']}' called {rec['count']} times with "
                        f"identical arguments (threshold {max_identical})."
                    ),
                    evidence={
                        "tool_id": rec["tool_id"],
                        "count": rec["count"],
                        "threshold": max_identical,
                        "first_index": rec["first"],
                    },
                )
            )
    return findings


def detect_error_cascade(steps: Any, *, max_consecutive_errors: int) -> List[Finding]:
    """Flag a run of ``max_consecutive_errors`` consecutive errored steps.

    Corresponds to the paper's "cascade tool errors" failure mode.
    """
    if max_consecutive_errors <= 0:
        return []
    run = 0
    start: Optional[int] = None
    best_len = 0
    best_start: Optional[int] = None
    for idx, step in enumerate(steps or []):
        if _step_errored(step):
            if run == 0:
                start = idx
            run += 1
            if run > best_len:
                best_len, best_start = run, start
        else:
            run, start = 0, None
    if best_len >= max_consecutive_errors:
        return [
            Finding(
                detector="error_cascade",
                severity="high",
                message=(
                    f"{best_len} consecutive errored steps "
                    f"(threshold {max_consecutive_errors})."
                ),
                evidence={
                    "run_length": best_len,
                    "start_index": best_start,
                    "threshold": max_consecutive_errors,
                },
            )
        ]
    return []


def check_required_coverage(
    steps: Any, required_tools: Set[str], required_actions: Set[str]
) -> List[Finding]:
    """Confirm every required tool/action was actually called.

    The paper's coverage check lifted deterministic detection from ~60% to
    ~96% of failures at zero false positives ("confirms every required call
    was made").
    """
    seen_tools: Set[str] = set()
    seen_actions: Set[str] = set()
    for _, content in _iter_tool_runs(steps):
        tid = getattr(getattr(content, "tool_config", None), "id", None)
        if tid:
            seen_tools.add(tid)
        act = getattr(getattr(content, "action_details", None), "action", None)
        if act:
            seen_actions.add(act)
    missing_tools = sorted(set(required_tools) - seen_tools)
    missing_actions = sorted(set(required_actions) - seen_actions)
    if not (missing_tools or missing_actions):
        return []
    return [
        Finding(
            detector="coverage",
            severity="medium",
            message="Required tool call(s) missing from the run.",
            evidence={
                "missing_tools": missing_tools,
                "missing_actions": missing_actions,
                "seen_tools": sorted(seen_tools),
            },
        )
    ]


def verify_stated_total(
    steps: Any,
    stated_total: float,
    *,
    key: str,
    agg: str = "sum",
    tolerance: float = 0.0,
) -> List[Finding]:
    """Recompute a run's stated total from the tool outputs it received.

    The paper's flagship deterministic check: "recomputes a run's stated
    total from the tool results it actually received". With
    ``tolerance == 0`` this is exact arithmetic and produces no false
    positives on healthy runs.
    """
    values: List[float] = []
    for _, content in _iter_tool_runs(steps):
        output = getattr(content, "output", None) or {}
        if isinstance(output, dict) and key in output:
            try:
                values.append(float(output[key]))
            except (TypeError, ValueError):
                continue
    if not values:
        return []  # nothing observed to reconcile against -- not a failure
    recomputed = sum(values) if agg == "sum" else max(values)
    if abs(recomputed - stated_total) <= tolerance:
        return []
    return [
        Finding(
            detector="total_mismatch",
            severity="high",
            message=(
                f"Stated total {stated_total} does not reconcile with tool outputs "
                f"({agg} of '{key}' = {recomputed})."
            ),
            evidence={
                "stated_total": stated_total,
                "recomputed": recomputed,
                "agg": agg,
                "key": key,
                "n_values": len(values),
            },
        )
    ]


def monitor_task_view(
    task_view: TaskView, config: Optional[MonitorConfig] = None
) -> MonitorReport:
    """Run the enabled detectors over a ``TaskView`` and return a verdict."""
    cfg = config or MonitorConfig()
    steps = list(getattr(task_view, "results", []) or [])

    findings: List[Finding] = []
    if cfg.enable_loops:
        findings += detect_loops(steps, max_identical=cfg.max_identical_calls)
    if cfg.enable_error_cascade:
        findings += detect_error_cascade(
            steps, max_consecutive_errors=cfg.max_consecutive_errors
        )
    if cfg.enable_coverage:
        findings += check_required_coverage(
            steps, cfg.required_tools, cfg.required_actions
        )
    if cfg.enable_total_verification and cfg.stated_total is not None:
        findings += verify_stated_total(
            steps,
            cfg.stated_total,
            key=cfg.total_key,
            agg=cfg.total_agg,
            tolerance=cfg.total_tolerance,
        )

    return MonitorReport(
        failed=bool(findings),
        findings=findings,
        step_count=len(steps),
        tool_calls=[_call_summary(c) for _, c in _iter_tool_runs(steps)],
        recommended_repair=_REPAIR_HINT if findings else None,
    )
