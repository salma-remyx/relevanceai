"""
Task harness — deterministic scaffolding around an agent's task execution.

Inspired by (Mode 3 - inspired experiment):

    "Harnessing LLMs for Reliable Academic Supervision: A Comparative Study"
    https://arxiv.org/abs/2607.14707v1

The paper's portable contribution is *harness engineering*: deliberately
composing deterministic scaffolding (schema-typed checks, an LLM-as-judge
loop with bounded retry, a human-in-the-loop confirmation gate, and a
per-node audit trail) around an LLM core, so a much smaller model behaves
reliably and traceably. Its thesis is that where reliability and
traceability matter, this scaffolding beats "use a bigger model".

This SDK is a thin httpx client with no model/inference/runtime, so the
paper's full LangGraph graph + SQLite audit store + academic-supervision
domain scoring cannot be hosted here. But the SDK already exposes every
primitive that harness needs on the Agent resource:

    trigger_task            -> start the LLM-core run
    get_task_output_preview -> read the produced answer
    rerun_task              -> the bounded-retry primitive a judge loop needs
    approve_task            -> the ``requires_confirmation`` HITL gate
    view_task_steps         -> yields the TaskView (the audit source)

``TaskHarness`` composes those *existing* methods into the paper's shape:
judge each output, retry on rejection up to a bound, honour the HITL gate,
and record every node to an audit trail for traceability.

Mode 2 substitutions (auxiliaries the SDK cannot host, swapped for
target-native equivalents - the harness *shape* is unchanged):

  * LLM-as-judge  -> a caller-supplied callable ``output -> (accepted, reason)``
    defaulting to a parameter-free non-empty / schema predicate (the paper's
    symbolic-semantic judge replaced by a proxy). Pass a real LLM call to
    reproduce that judge at full fidelity.
  * per-node SQLite audit trail -> an in-memory, JSON-serialisable list
    (the SDK has no DB runtime; persistent storage is a downstream concern).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, List, Optional, Tuple

# NON-NEW call-site module: the Agent resource this harness wraps. The harness
# drives the agent's existing task-execution methods rather than replacing them.
from relevanceai.resources.agent import Agent

# A judge inspects a completed task's output and returns whether it is
# acceptable plus a short reason. The reason lets the audit trail explain
# *why* a node passed or failed (the paper's explainability axis).
JudgeResult = Tuple[bool, str]
Judge = Callable[[Any], JudgeResult]


def default_judge(output: Any) -> JudgeResult:
    """Parameter-free acceptance predicate (Mode 2 judge substitution).

    Accepts any non-empty, non-false output. Swap for an LLM-as-judge call
    to reproduce the paper's symbolic-semantic judge at full fidelity; the
    harness shape is identical either way.
    """
    if not output:
        return False, "empty output"
    if isinstance(output, str) and not output.strip():
        return False, "blank output"
    return True, "non-empty output"


@dataclass
class AuditNode:
    """One recorded step of harness execution (the paper's per-node audit)."""

    node: str
    status: str  # "ok" | "rejected" | "hitl" | "error"
    detail: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AuditTrail:
    """In-memory, JSON-serialisable per-node audit trail.

    Target-native stand-in for the paper's per-node SQLite audit trail: the
    SDK has no DB runtime, so the trail is a list that can be dumped to JSON
    for offline inspection. Persistent storage is intentionally out of scope.
    """

    nodes: List[AuditNode] = field(default_factory=list)

    def record(
        self,
        node: str,
        status: str,
        detail: str = "",
    ) -> AuditNode:
        entry = AuditNode(
            node=node,
            status=status,
            detail=detail,
            timestamp=datetime.now().isoformat(),
        )
        self.nodes.append(entry)
        return entry

    def to_list(self) -> List[dict]:
        return [n.to_dict() for n in self.nodes]

    def to_json(self) -> str:
        return json.dumps(self.to_list(), indent=2)

    def statuses(self) -> List[str]:
        return [n.status for n in self.nodes]


class TaskHarness:
    """Wrap an Agent in deterministic scaffolding: judge loop + HITL + audit.

    Parameters
    ----------
    agent:
        A relevanceai ``Agent`` (sync) - the LLM-core proxy the harness drives.
    judge:
        Callable ``output -> (accepted, reason)``. Defaults to a
        parameter-free non-empty check; pass an LLM-as-judge for the paper's
        symbolic-semantic judge.
    """

    def __init__(self, agent: Agent, judge: Optional[Judge] = None):
        self.agent = agent
        self.judge = judge or default_judge
        self.trail = AuditTrail()

    def run(
        self,
        message: str,
        max_retries: int = 2,
        poll_sleep: float = 0.0,
        approve_confirmation: bool = True,
    ) -> Tuple[Any, AuditTrail]:
        """Trigger a task, judge its output, and retry on rejection.

        Honours the agent's HITL confirmation gate (at most once per run) and
        records every node to the audit trail. Returns ``(final_output, trail)``;
        the final output is ``None`` only if the task never completes.
        """
        triggered = self.agent.trigger_task(message=message)
        conversation_id = getattr(triggered, "conversation_id", None)
        self.trail.record("trigger", "ok", conversation_id or "")

        output = self._await_output(conversation_id, poll_sleep)
        if output is False:
            self.trail.record("output", "error", "task did not complete")
            return None, self.trail

        # HITL gate: if the task is awaiting confirmation, approve it before
        # judging the final output (the paper's human-in-the-loop pattern).
        if approve_confirmation and self._needs_confirmation(conversation_id):
            self.agent.approve_task(conversation_id)
            self.trail.record("hitl", "hitl", "confirmation gate approved")
            output = self._await_output(conversation_id, poll_sleep)

        attempts = 0
        while True:
            accepted, reason = self.judge(output)
            if accepted:
                self.trail.record("judge", "ok", reason)
                return output, self.trail
            attempts += 1
            self.trail.record(
                "judge", "rejected", f"{reason} (attempt {attempts})"
            )
            if attempts > max_retries:
                self.trail.record(
                    "retry", "error", f"giving up after {attempts} attempt(s)"
                )
                return output, self.trail
            self.agent.rerun_task(conversation_id=conversation_id)
            self.trail.record("retry", "ok", f"rerun attempt {attempts}")
            output = self._await_output(conversation_id, poll_sleep)

    # -- helpers --------------------------------------------------------

    def _await_output(self, conversation_id: str, poll_sleep: float) -> Any:
        """Fetch the task output, optionally polling like trigger_and_poll.

        With ``poll_sleep <= 0`` (the default, and what tests use) this is a
        single fetch. With ``poll_sleep > 0`` it mirrors
        ``examples/trigger_and_poll_tasks.py``, sleeping between polls until
        the task reports a complete (truthy) output.
        """
        while True:
            output = self.agent.get_task_output_preview(conversation_id)
            if output:
                return output
            if poll_sleep <= 0:
                return output
            time.sleep(poll_sleep)

    def _needs_confirmation(self, conversation_id: str) -> bool:
        """True if any task step is flagged ``requires_confirmation``."""
        try:
            task_view = self.agent.view_task_steps(conversation_id)
        except Exception:
            return False
        for step in getattr(task_view, "results", []) or []:
            content = getattr(step, "content", None)
            if getattr(content, "requires_confirmation", False):
                return True
        return False


def _accepted(judge: Judge, output: Any) -> bool:
    accepted, _ = judge(output)
    return accepted


def run_reliability_comparison(
    agent: Agent,
    scenarios: List[str],
    judge: Optional[Judge] = None,
    max_retries: int = 2,
    poll_sleep: float = 0.0,
) -> dict:
    """Run scenarios with the harness ON vs OFF and compare reliability.

    Baseline (harness OFF): a single trigger + output fetch, no judge, no
    retry, no HITL - the paper's unscaffolded "ASA" analogue. Harness ON:
    ``TaskHarness.run`` with judge loop, bounded retry, HITL gate, and audit
    trail.

    Returns per-mode pass-rate and the harness audit trail. This mirrors the
    paper's within-subject harness-vs-no-harness comparison without
    reproducing its ten-rater rubric; deeper evaluation is a downstream PR.
    """
    judge_fn = judge or default_judge
    harness = TaskHarness(agent, judge=judge_fn)

    baseline_results: List[bool] = []
    harness_results: List[bool] = []

    for message in scenarios:
        # Baseline: one trigger + one fetch, no scaffolding.
        try:
            trig = agent.trigger_task(message=message)
            out = agent.get_task_output_preview(trig.conversation_id)
            baseline_results.append(_accepted(judge_fn, out))
        except Exception:
            baseline_results.append(False)

        # Harness: judge loop + bounded retry + HITL gate + audit trail.
        try:
            out, _ = harness.run(
                message=message,
                max_retries=max_retries,
                poll_sleep=poll_sleep,
            )
            harness_results.append(_accepted(judge_fn, out))
        except Exception:
            harness_results.append(False)

    def pass_rate(results: List[bool]) -> float:
        return sum(1 for x in results if x) / len(results) if results else 0.0

    return {
        "n": len(scenarios),
        "baseline_pass_rate": pass_rate(baseline_results),
        "harness_pass_rate": pass_rate(harness_results),
        "trail": harness.trail.to_list(),
    }
