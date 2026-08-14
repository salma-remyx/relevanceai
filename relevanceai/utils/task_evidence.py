"""
Task evidence-graph grounding checks.

Represents an agent task's reasoning trace as a typed evidence graph and
validates that each claim the agent makes is grounded in the data its tools
actually produced.

Adapted from the EviGraph framework ("EviGraph: Evidence-Guided Autonomous
Research Agents"), which models a research process as a typed evidence graph
(Problem / Gap / Hypothesis / Experiment / Finding / Claim nodes) that the
agent maintains as operational state, inspects for missing dependencies and
result-claim inconsistencies, and uses to localize its earliest weak node.

This module ports that core mechanism onto the Relevance AI task-trace
surface. A ``TaskView`` returned by ``Agent.view_task_steps`` is walked and
each ``TaskStep`` is classified into one of three evidence-node types that map
onto the trace's real content types:

  * ``user-message``  -> ``PROBLEM``  (the request the agent must answer)
  * ``tool-run``      -> ``EVIDENCE`` (structured data a tool produced)
  * ``agent-message`` -> ``CLAIM``    (an assertion that must be grounded)

Mode 2 (adapted port). EviGraph's learned semantic-alignment estimator is
substituted here with a parameter-free vocab-overlap proxy (claim coverage),
because the SDK hosts no embedding model. EviGraph's graph checkpointing,
downstream-subgraph regeneration, and manuscript generation are intentionally
out of scope: they require an autonomous-research runtime this library does
not provide. The delivered slice is the paper's stated core -- explicit
evidence-state maintenance and claim-evidence consistency checking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from relevanceai.types.task import TaskView


class NodeType(str, Enum):
    """Evidence-node types projected from a task trace's content types."""

    PROBLEM = "problem"
    EVIDENCE = "evidence"
    CLAIM = "claim"


# Small stopword set so the vocab-overlap signal reflects content rather than
# glue words. Kept tiny and dependency-free on purpose.
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "that", "this", "with", "from", "have", "has",
        "are", "was", "were", "but", "not", "you", "all", "can", "her",
        "was", "one", "our", "out", "his", "its", "they", "their", "them",
        "been", "will", "would", "could", "should", "into", "than", "then",
        "based", "using", "used",
    }
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class EvidenceNode:
    """A single node in the evidence graph projected from one task step."""

    item_id: str
    node_type: NodeType
    text: str
    tokens: Set[str] = field(default_factory=set)
    source_type: str = ""
    grounding_score: Optional[float] = None
    weak_reason: Optional[str] = None


@dataclass
class EvidenceGraph:
    """Ordered typed-evidence view of a task trace (trace order preserved)."""

    nodes: List[EvidenceNode] = field(default_factory=list)

    @property
    def claims(self) -> List[EvidenceNode]:
        return [n for n in self.nodes if n.node_type == NodeType.CLAIM]

    @property
    def evidence(self) -> List[EvidenceNode]:
        return [n for n in self.nodes if n.node_type == NodeType.EVIDENCE]

    def preceding_evidence(self, index: int) -> List[EvidenceNode]:
        """Evidence nodes that appear before ``index`` in the trace."""
        return [
            n
            for n in self.nodes[:index]
            if n.node_type == NodeType.EVIDENCE
        ]


@dataclass
class WeakNode:
    """A claim that failed the grounding check, with the reason."""

    item_id: str
    text: str
    reason: str
    grounding_score: Optional[float] = None


@dataclass
class GroundingReport:
    """Result of checking that every claim in a trace is grounded."""

    task_consistent: bool
    total_claims: int
    grounded_claims: int
    weak_nodes: List[WeakNode] = field(default_factory=list)
    earliest_weak_node: Optional[WeakNode] = None
    scores: Dict[str, float] = field(default_factory=dict)


def _tokenize(text: str) -> Set[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return {t for t in tokens if len(t) >= 3 and t not in _STOPWORDS}


def _flatten_output(value: Any) -> str:
    """Flatten a (possibly nested) tool output into a single tokenizable string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    parts: List[str] = []
    if isinstance(value, dict):
        for chunk in value.values():
            parts.append(_flatten_output(chunk))
    elif isinstance(value, (list, tuple)):
        for chunk in value:
            parts.append(_flatten_output(chunk))
    else:
        parts.append(str(value))
    return " ".join(part for part in parts if part)


def _node_from_step(step: Any) -> Optional[EvidenceNode]:
    """Project a single ``TaskStep`` into an evidence node, or ``None`` to skip."""
    content = getattr(step, "content", None)
    if content is None:
        return None
    content_type = getattr(content, "type", "")
    item_id = getattr(step, "item_id", content_type)
    if content_type == "user-message":
        text = getattr(content, "text", "") or ""
        return EvidenceNode(item_id, NodeType.PROBLEM, text, _tokenize(text), content_type)
    if content_type == "tool-run":
        text = _flatten_output(getattr(content, "output", None))
        if not text.strip():
            # A tool run that produced no data carries no evidence -> skip.
            return None
        return EvidenceNode(item_id, NodeType.EVIDENCE, text, _tokenize(text), content_type)
    if content_type == "agent-message":
        text = getattr(content, "text", "") or ""
        return EvidenceNode(item_id, NodeType.CLAIM, text, _tokenize(text), content_type)
    return None


def build_evidence_graph(task_view: TaskView) -> EvidenceGraph:
    """Walk a ``TaskView`` trace and project it into a typed evidence graph."""
    graph = EvidenceGraph()
    for step in getattr(task_view, "results", []) or []:
        node = _node_from_step(step)
        if node is not None:
            graph.nodes.append(node)
    return graph


def _claim_coverage(claim_tokens: Set[str], evidence_tokens: Set[str]) -> float:
    """Fraction of the claim's content tokens that are backed by evidence.

    Parameter-free proxy for EviGraph's learned semantic-alignment signal: a
    grounded claim shares most of its content vocabulary with the data its
    tools produced, while a hallucinated claim introduces entities/numbers the
    evidence never contained.
    """
    if not claim_tokens:
        return 0.0
    if not evidence_tokens:
        return 0.0
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


def check_grounding(
    task_view: TaskView,
    *,
    min_overlap: float = 0.3,
) -> GroundingReport:
    """Validate that every claim in a task trace is grounded in its evidence.

    For each ``CLAIM`` node, computes vocab-overlap coverage against the union
    of all preceding ``EVIDENCE`` nodes. Claims below ``min_overlap`` are weak;
    the earliest weak node is localized (mirroring EviGraph's repair entry
    point). A claim with no preceding evidence is flagged
    ``missing-dependency``; a claim with evidence but low overlap is flagged
    ``semantic-misalignment``.
    """
    graph = build_evidence_graph(task_view)
    weak_nodes: List[WeakNode] = []
    scores: Dict[str, float] = {}
    grounded = 0
    earliest_weak: Optional[WeakNode] = None

    for index, node in enumerate(graph.nodes):
        if node.node_type != NodeType.CLAIM:
            continue
        preceding = graph.preceding_evidence(index)
        evidence_tokens: Set[str] = set()
        for evidence in preceding:
            evidence_tokens |= evidence.tokens
        score = _claim_coverage(node.tokens, evidence_tokens)
        node.grounding_score = score
        scores[node.item_id] = score

        if score >= min_overlap:
            grounded += 1
        else:
            reason = (
                "missing-dependency" if not evidence_tokens else "semantic-misalignment"
            )
            node.weak_reason = reason
            weak = WeakNode(node.item_id, node.text, reason, score)
            weak_nodes.append(weak)
            if earliest_weak is None:
                earliest_weak = weak

    return GroundingReport(
        task_consistent=not weak_nodes,
        total_claims=len(graph.claims),
        grounded_claims=grounded,
        weak_nodes=weak_nodes,
        earliest_weak_node=earliest_weak,
        scores=scores,
    )


def format_report(report: GroundingReport) -> str:
    """Render a short human-readable summary of a grounding report."""
    status = "consistent" if report.task_consistent else "inconsistent"
    lines = [
        f"[evidence] task {status}: "
        f"{report.grounded_claims}/{report.total_claims} claims grounded",
    ]
    for weak in report.weak_nodes:
        score = "" if weak.grounding_score is None else f" (overlap={weak.grounding_score:.2f})"
        snippet = weak.text.strip().replace("\n", " ")[:80]
        lines.append(f"  - weak claim [{weak.reason}]{score}: {snippet}")
    return "\n".join(lines)
