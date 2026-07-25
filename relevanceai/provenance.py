"""Agent provenance — turn a task trajectory into a dataflow graph.

Core idea adapted from *AgentTrails: Towards Trust and Reuse for Agentic
Tasks* (arXiv:2607.18816). AgentTrails converts raw, chronological agent
trajectories into structured *provenance graphs*: tool calls become
computational actions, their inputs/outputs become data artifacts, and the
dependencies between them become edges — so dataflow, hidden dependencies and
recurring tool-use patterns become visible instead of being buried in a log.

This module ports that idea onto the Relevance AI SDK. It consumes the
``TaskView`` returned by
:meth:`relevanceai.resources.agent.Agent.view_task_steps` and produces a
provenance graph, plus a joined *quotient graph* for comparing several
executions of the same agent. The construction is a deterministic, purely
client-side transform — it needs no model, training, or external service.

Edge semantics
--------------
* ``produces``    action -> an artifact it outputs
* ``consumed-by`` an upstream artifact -> the action that consumes it
* ``feeds``       a user/agent message -> the action it drives
* ``sequential``  fallback ordering edge used when no explicit data reference
                  is found (the trajectory-to-DAG default AgentTrails describes)
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, Iterator, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .types.task import TaskView

__all__ = [
    "ProvenanceNode",
    "ProvenanceEdge",
    "ProvenanceGraph",
    "build_provenance",
    "quotient_graph",
    "recurring_patterns",
]

_TEMPLATE_RE = re.compile(r"\$\{\{\s*(.*?)\s*\}\}")
_STEP_OUTPUT_RE = re.compile(r"steps?\.([A-Za-z0-9_\- ]+?)\.output", re.IGNORECASE)
_BARE_OUTPUT_RE = re.compile(r"output\.([A-Za-z0-9_]+)", re.IGNORECASE)


class ProvenanceNode(BaseModel):
    """A node in a provenance graph: an action, an artifact, or a message."""

    model_config = ConfigDict(extra="allow")

    id: str
    kind: str  # "action" | "artifact" | "message"
    node_type: str  # "tool-run" | "user-message" | "agent-message" | "agent-error" | "tool-output"
    label: str
    attributes: Dict[str, Any] = Field(default_factory=dict)


class ProvenanceEdge(BaseModel):
    """A directed dataflow edge between two provenance nodes."""

    model_config = ConfigDict(extra="allow")

    source: str
    target: str
    relation: str
    attributes: Dict[str, Any] = Field(default_factory=dict)


class ProvenanceGraph(BaseModel):
    """A provenance / dataflow graph for one agent task trajectory."""

    model_config = ConfigDict(extra="allow")

    conversation_id: Optional[str] = None
    nodes: Dict[str, ProvenanceNode] = Field(default_factory=dict)
    edges: List[ProvenanceEdge] = Field(default_factory=list)

    def actions(self) -> List[ProvenanceNode]:
        return [n for n in self.nodes.values() if n.kind == "action"]

    def artifacts(self) -> List[ProvenanceNode]:
        return [n for n in self.nodes.values() if n.kind == "artifact"]

    def add_node(self, node: ProvenanceNode) -> ProvenanceNode:
        self.nodes[node.id] = node
        return node

    def add_edge(self, source: str, target: str, relation: str, **attrs: Any) -> None:
        """Record an edge, dropping it if either endpoint is not yet a node."""
        if source not in self.nodes or target not in self.nodes:
            return
        self.edges.append(
            ProvenanceEdge(
                source=source, target=target, relation=relation, attributes=dict(attrs)
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "nodes": {nid: n.model_dump() for nid, n in self.nodes.items()},
            "edges": [e.model_dump() for e in self.edges],
        }

    def to_dot(self) -> str:
        """Render the graph as Graphviz DOT for quick visualisation."""
        lines = ["digraph provenance {"]
        for n in self.nodes.values():
            label = n.label.replace('"', "'").replace("\n", " ")
            lines.append(
                f'  "{n.id}" [label="{label}", kind={n.kind}, type={n.node_type}];'
            )
        for e in self.edges:
            lines.append(f'  "{e.source}" -> "{e.target}" [label={e.relation}];')
        lines.append("}")
        return "\n".join(lines)


def _iter_strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_strings(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_strings(value)


def _extract_refs(blob: Any) -> List[tuple]:
    """Return ``(kind, token)`` data references found in ``${{ ... }}`` templates.

    ``kind`` is ``"step"`` (a named prior tool) or ``"output"`` (a bare output key).
    """
    refs: List[tuple] = []
    for text in _iter_strings(blob):
        for expr in _TEMPLATE_RE.findall(text):
            refs.extend(("step", s.strip().lower()) for s in _STEP_OUTPUT_RE.findall(expr))
            refs.extend(("output", k.strip().lower()) for k in _BARE_OUTPUT_RE.findall(expr))
    return refs


def _params_blob(content: Any) -> Dict[str, Any]:
    """Pull the declared (templated) and resolved params off a ``tool-run`` step."""
    params = getattr(content, "params", None)
    if not getattr(params, "valid", False):
        return {}
    declared = getattr(params, "json_", None) or {}
    resolved = getattr(params, "resolved", None) or {}
    return {"declared": declared, "resolved": resolved}


def build_provenance(
    task_view: TaskView, conversation_id: Optional[str] = None
) -> ProvenanceGraph:
    """Build a provenance graph from a ``TaskView`` trajectory.

    Wire this directly after ``Agent.view_task_steps``: pass its return value
    straight in.
    """
    graph = ProvenanceGraph(conversation_id=conversation_id)

    output_key_to_actions: Dict[str, List[str]] = {}
    tool_token_to_action: Dict[str, str] = {}
    last_action: Optional[str] = None
    last_message: Optional[str] = None

    def _short(text: str, limit: int = 60) -> str:
        return (text[:limit] + "…") if len(text) > limit else text

    for step in getattr(task_view, "results", None) or []:
        content = getattr(step, "content", None)
        ctype = getattr(content, "type", None)
        item_id = getattr(step, "item_id", "") or ""

        if ctype == "user-message":
            text = getattr(content, "text", "") or ""
            node_id = f"message:{item_id}"
            graph.add_node(
                ProvenanceNode(
                    id=node_id,
                    kind="message",
                    node_type="user-message",
                    label=_short(text),
                    attributes={
                        "text": text,
                        "trigger": getattr(content, "is_trigger_message", None),
                    },
                )
            )
            last_message = node_id

        elif ctype == "agent-message":
            text = getattr(content, "text", "") or ""
            node_id = f"message:{item_id}"
            graph.add_node(
                ProvenanceNode(
                    id=node_id,
                    kind="artifact",
                    node_type="agent-message",
                    label=_short(text),
                    attributes={"text": text},
                )
            )
            if last_action is not None:
                graph.add_edge(last_action, node_id, "produces")

        elif ctype == "agent-error":
            node_id = f"error:{item_id}"
            errors = getattr(content, "errors", None) or []
            graph.add_node(
                ProvenanceNode(
                    id=node_id,
                    kind="artifact",
                    node_type="agent-error",
                    label="agent-error",
                    attributes={"errors": [getattr(e, "body", str(e)) for e in errors]},
                )
            )
            if last_action is not None:
                graph.add_edge(last_action, node_id, "produces")

        elif ctype == "tool-run":
            tool_config = getattr(content, "tool_config", None)
            action_details = getattr(content, "action_details", None)
            action_request_id = getattr(action_details, "action_request_id", None) or item_id
            action_id = f"action:{action_request_id}"
            tool_id = getattr(tool_config, "id", "") or ""
            tool_title = getattr(tool_config, "title", "") or tool_id
            action_name = getattr(action_details, "action", "") or ""

            param_blob = _params_blob(content)
            graph.add_node(
                ProvenanceNode(
                    id=action_id,
                    kind="action",
                    node_type="tool-run",
                    label=f"{tool_title}#{action_name}" if action_name else tool_title,
                    attributes={
                        "tool_id": tool_id,
                        "tool_title": tool_title,
                        "action": action_name,
                        "state": str(getattr(content, "tool_run_state", "")),
                        "requires_confirmation": getattr(content, "requires_confirmation", False),
                        "params": param_blob.get("resolved") or param_blob.get("declared"),
                    },
                )
            )

            for token in {tool_id.lower(), tool_title.lower()}:
                if token:
                    tool_token_to_action.setdefault(token, action_id)

            if last_message is not None:
                graph.add_edge(last_message, action_id, "feeds")

            # explicit data references inside the declared (templated) params
            refs = _extract_refs(param_blob.get("declared"))
            connected = False
            for kind, token in refs:
                if kind == "step":
                    producer = tool_token_to_action.get(token)
                else:  # bare output key -> most recent producer of that key
                    producers = output_key_to_actions.get(token, [])
                    producer = producers[-1] if producers else None
                if not producer or producer == action_id:
                    continue
                producer_req = producer.split(":", 1)[1]
                artifacts = [
                    nid for nid in graph.nodes if nid.startswith(f"artifact:{producer_req}:")
                ]
                graph.add_edge(artifacts[0] if artifacts else producer, action_id, "consumed-by")
                connected = True

            # sequential fallback: prior action's output flows forward in order
            if not connected and last_action is not None:
                graph.add_edge(last_action, action_id, "sequential")

            output = getattr(content, "output", None) or {}
            if isinstance(output, dict):
                for key, value in output.items():
                    art_id = f"artifact:{action_request_id}:{key}"
                    if isinstance(value, (str, int, float, bool)):
                        shown: Any = value
                    else:
                        shown = type(value).__name__
                    graph.add_node(
                        ProvenanceNode(
                            id=art_id,
                            kind="artifact",
                            node_type="tool-output",
                            label=f"{tool_title}.{key}",
                            attributes={"key": key, "value": shown},
                        )
                    )
                    graph.add_edge(action_id, art_id, "produces")
                    output_key_to_actions.setdefault(key.lower(), []).append(action_id)

            last_action = action_id
            last_message = None

    return graph


def quotient_graph(graphs: Iterable[ProvenanceGraph]) -> ProvenanceGraph:
    """Join several provenance graphs on a shared canvas as a quotient graph.

    Actions are merged by ``(tool_id, action)``; artifacts/messages by
    ``(kind, node_type, label-stem)``; edges by their quotient endpoints.
    Each merged node records how many trajectories it appeared in, so recurring
    tools and dependency structures surface as high ``occurrences`` counts.
    """
    merged = ProvenanceGraph(conversation_id="<quotient>")

    def action_qkey(node: ProvenanceNode) -> str:
        attr = node.attributes
        return f"action|{attr.get('tool_id', '')}|{attr.get('action', '')}"

    def artifact_qkey(node: ProvenanceNode) -> str:
        return f"{node.kind}|{node.node_type}|{node.label.split('.')[0]}"

    qkey_to_node: Dict[str, ProvenanceNode] = {}
    occurrence: Dict[str, set] = {}
    edge_counts: Dict[tuple, int] = {}

    for graph in graphs:
        original_to_qkey: Dict[str, str] = {}
        for node in graph.nodes.values():
            qkey = action_qkey(node) if node.kind == "action" else artifact_qkey(node)
            original_to_qkey[node.id] = qkey
            if qkey not in qkey_to_node:
                representative = node.model_copy()
                representative.id = qkey
                representative.attributes = dict(representative.attributes)
                representative.attributes["occurrences"] = 0
                qkey_to_node[qkey] = representative
                occurrence[qkey] = set()
            occurrence[qkey].add(graph.conversation_id or "")

        seen_edges: set = set()
        for edge in graph.edges:
            src = original_to_qkey.get(edge.source)
            tgt = original_to_qkey.get(edge.target)
            if not src or not tgt:
                continue
            key = (src, tgt, edge.relation)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            edge_counts[key] = edge_counts.get(key, 0) + 1

    for qkey, node in qkey_to_node.items():
        node.attributes["occurrences"] = len(occurrence[qkey])
        merged.nodes[qkey] = node

    for (src, tgt, relation), count in edge_counts.items():
        if src in merged.nodes and tgt in merged.nodes:
            merged.edges.append(
                ProvenanceEdge(
                    source=src, target=tgt, relation=relation, attributes={"count": count}
                )
            )

    return merged


def recurring_patterns(
    graphs: Iterable[ProvenanceGraph], n: int = 2
) -> List[Dict[str, Any]]:
    """Surface recurring action-tool sequences of length ``n`` across trajectories.

    This is AgentTrails' "recurring tool-use patterns" signal: a frequent
    ``A -> B`` adjacency hints at a reusable skill worth abstracting.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    counter: Counter = Counter()
    for graph in graphs:
        sequence = [a.attributes.get("tool_id") or a.label for a in graph.actions()]
        for index in range(len(sequence) - n + 1):
            counter[tuple(sequence[index : index + n])] += 1
    return [
        {"pattern": list(pattern), "count": count}
        for pattern, count in counter.most_common()
        if count >= 1
    ]
