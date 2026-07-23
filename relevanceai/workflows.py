"""
Stateful workflow graphs for long-running agent pipelines.

Adapted (Mode 3 -- inspired experiment) from the recipe patterns in
"Graph-Based Agentic AI with LangGraph: Workflow Pathways for Long-Running
Stateful Business Processes" (arXiv:2607.19297). This is a *target-native*
re-implementation of that paper's core workflow insight -- typed state,
conditional routing, retries, checkpoints / interrupt-and-resume, and an
execution trace -- built directly on the Relevance AI SDK's existing async
agent/task primitives. It does not depend on, port, or reproduce LangGraph or
the paper's three example recipes; it applies the paper's *idea* (make the
routes, retries, pauses and audit trail of a long-running pipeline explicit
graph behaviour instead of hidden imperative logic) to the SDK's real surface.

The motivating problem is the ad-hoc "trigger then ``while not done:
time.sleep``" poll loop used today (see ``examples/trigger_and_poll_tasks.py``).
For stateful or failure-prone multi-step pipelines that loop buries routing,
retry and audit logic in imperative code; a state graph makes those explicit.
See :func:`trigger_poll_workflow` for the SDK-backed convenience that replaces
it.
"""
from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

__all__ = [
    "START",
    "END",
    "StateGraph",
    "MemorySaver",
    "RetryPolicy",
    "GraphInterrupt",
    "GraphRecursionError",
    "RunResult",
    "TraceEntry",
    "trigger_poll_workflow",
]

START = "__start__"
END = "__end__"

State = Dict[str, Any]
NodeFunc = Callable[[State], Union[State, Awaitable[State]]]
Router = Callable[[State], str]


class GraphRecursionError(RuntimeError):
    """A run exceeded its step budget without reaching the END node."""


class GraphInterrupt(Exception):
    """Pause a run for external input (human-in-the-loop review).

    A node raises this to yield control back to the caller. The state at the
    point of interrupt is checkpointed, so the same thread can be resumed --
    re-running the interrupted node -- once the external decision is made.
    """


@dataclass
class RetryPolicy:
    """Retry a node's body on failure (LangGraph's retry-on-node analogue)."""

    max_attempts: int = 1
    retry_on: Union[type, Tuple[type, ...]] = Exception


@dataclass
class TraceEntry:
    """One executed node in the audit trail."""

    node: str
    attempts: int
    output: Any = None
    interrupted: bool = False


@dataclass
class RunResult:
    """Outcome of a graph run: final state, audit trail, and pause flag."""

    state: State
    trace: List[TraceEntry]
    interrupted: bool


class MemorySaver:
    """In-memory checkpointer (LangGraph's ``MemorySaver`` analogue).

    Snapshots are keyed by thread id + step, supporting interrupt/resume and
    :meth:`StateGraph.get_state` inspection. Swap in any object exposing
    ``put``/``get``/``list`` with the same signatures for durable storage.
    """

    def __init__(self) -> None:
        self._store: Dict[str, dict] = {}
        self._latest: Dict[str, str] = {}

    def put(self, thread_id: str, state: State, next_node: str, step: int) -> str:
        checkpoint_id = f"{thread_id}:{step}"
        self._store[checkpoint_id] = {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
            "state": dict(state),
            "next_node": next_node,
            "step": step,
        }
        self._latest[thread_id] = checkpoint_id
        return checkpoint_id

    def get(self, thread_id: str) -> Optional[dict]:
        checkpoint_id = self._latest.get(thread_id)
        return self._store.get(checkpoint_id) if checkpoint_id else None

    def list(self, thread_id: str) -> List[dict]:
        return [
            self._store[cid]
            for cid in sorted(self._store, key=lambda c: self._store[c]["step"])
            if self._store[cid]["thread_id"] == thread_id
        ]


class StateGraph:
    """A small state-graph runner: nodes, edges, conditional routing, retries.

    State is a plain dict; each node returns a partial dict that is merged in
    (shallow overwrite, matching LangGraph's default reducer-less merge). Nodes
    may be sync or async.
    """

    def __init__(
        self,
        *,
        checkpointer: Optional[MemorySaver] = None,
        recursion_limit: int = 25,
    ) -> None:
        self.checkpointer = checkpointer
        self.recursion_limit = recursion_limit
        self._nodes: Dict[str, Tuple[NodeFunc, RetryPolicy]] = {}
        self._edges: Dict[str, str] = {}
        self._conditional: Dict[str, Router] = {}

    # -- graph construction -------------------------------------------------
    def add_node(
        self,
        name: str,
        func: NodeFunc,
        *,
        retry: Optional[RetryPolicy] = None,
    ) -> "StateGraph":
        if name in (START, END):
            raise ValueError(f"reserved node name: {name!r}")
        self._nodes[name] = (func, retry or RetryPolicy())
        return self

    def add_edge(self, source: str, target: str) -> "StateGraph":
        self._edges[source] = target
        return self

    def add_conditional_edges(self, source: str, router: Router) -> "StateGraph":
        self._conditional[source] = router
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        return self.add_edge(START, name)

    # -- introspection ------------------------------------------------------
    def get_state(self, thread_id: str = "default") -> Optional[dict]:
        return self.checkpointer.get(thread_id) if self.checkpointer else None

    # -- execution ----------------------------------------------------------
    async def ainvoke(
        self,
        state: Optional[State] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> RunResult:
        cfg = config or {}
        thread_id = cfg.get("thread_id", "default")

        checkpoint = self.checkpointer.get(thread_id) if self.checkpointer else None
        if checkpoint is not None and state is None:
            current_state: State = dict(checkpoint["state"])
            node = checkpoint["next_node"]
            step = checkpoint["step"]
        else:
            if START not in self._edges:
                raise ValueError("graph has no entry point; call set_entry_point(...)")
            current_state = dict(state or {})
            node = self._edges[START]
            step = 0

        trace: List[TraceEntry] = []
        while node != END:
            if step >= self.recursion_limit:
                raise GraphRecursionError(
                    f"exceeded recursion_limit={self.recursion_limit} at node {node!r}"
                )
            if node not in self._nodes:
                raise ValueError(f"unknown node {node!r}")

            try:
                output, attempts = await self._invoke_node(node, current_state)
            except GraphInterrupt as interrupt:
                trace.append(
                    TraceEntry(
                        node=node,
                        attempts=1,
                        output=interrupt.args[0] if interrupt.args else None,
                        interrupted=True,
                    )
                )
                if self.checkpointer is not None:
                    self.checkpointer.put(thread_id, current_state, node, step)
                return RunResult(state=current_state, trace=trace, interrupted=True)

            current_state = {**current_state, **output}
            trace.append(TraceEntry(node=node, attempts=attempts, output=output))
            node = self._next(node, current_state)
            step += 1
            if self.checkpointer is not None:
                self.checkpointer.put(thread_id, current_state, node, step)

        return RunResult(state=current_state, trace=trace, interrupted=False)

    def invoke(
        self,
        state: Optional[State] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> RunResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(state, config=config))
        raise RuntimeError(
            "StateGraph.invoke() cannot run inside an event loop; "
            "use 'await graph.ainvoke(...)' instead"
        )

    # -- internals ----------------------------------------------------------
    async def _invoke_node(self, name: str, state: State) -> Tuple[State, int]:
        func, policy = self._nodes[name]
        attempts = 0
        while True:
            attempts += 1
            try:
                result = func(state)
                if inspect.isawaitable(result):
                    result = await result
                return dict(result or {}), attempts
            except GraphInterrupt:
                raise
            except policy.retry_on:
                if attempts >= policy.max_attempts:
                    raise
                continue

    def _next(self, node: str, state: State) -> str:
        if node in self._conditional:
            return self._conditional[node](state)
        if node in self._edges:
            return self._edges[node]
        raise ValueError(f"node {node!r} has no outgoing edge to END")


def trigger_poll_workflow(
    agent: Any,
    message: str,
    *,
    poll_interval: float = 5.0,
    retry_max_attempts: int = 3,
    recursion_limit: int = 50,
    checkpointer: Optional[MemorySaver] = None,
) -> StateGraph:
    """Build a trigger -> poll state graph over an ``AsyncAgent``.

    This is the SDK-backed replacement for the ad-hoc
    ``while not output: time.sleep(5)`` loop in
    ``examples/trigger_and_poll_tasks.py``. Transient failures on trigger/poll
    are retried, conditional routing drives pending vs. complete, and every step
    is checkpointed and traced::

        START -> trigger -> poll --(pending)--> poll
                                 --(complete)-> END

    Run it with::

        graph = trigger_poll_workflow(my_agent, "Summarise this lead ...")
        result = await graph.ainvoke({}, config={"thread_id": "run-1"})
        if not result.interrupted:
            print(result.state["output"])
    """
    graph = StateGraph(
        checkpointer=checkpointer or MemorySaver(),
        recursion_limit=recursion_limit,
    )

    async def trigger(state: State) -> State:
        task = await agent.trigger_task(message=message)
        await asyncio.sleep(poll_interval)
        return {"conversation_id": task.conversation_id, "output": None}

    async def poll(state: State) -> State:
        await asyncio.sleep(poll_interval)
        output = await agent.get_task_output_preview(state["conversation_id"])
        return {"output": output}

    def route_poll(state: State) -> str:
        return "poll" if not state.get("output") else END

    retry = RetryPolicy(max_attempts=retry_max_attempts, retry_on=Exception)
    graph.add_node("trigger", trigger, retry=retry)
    graph.add_node("poll", poll, retry=retry)
    graph.set_entry_point("trigger")
    graph.add_edge("trigger", "poll")
    graph.add_conditional_edges("poll", route_poll)
    return graph
