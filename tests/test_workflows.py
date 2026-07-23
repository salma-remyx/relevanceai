import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

# Imports a NON-NEW module (relevanceai.resources.agent) to ground the wiring on
# the SDK's real long-running surface, and the new capability under test.
from relevanceai.resources.agent import AsyncAgent
from relevanceai.workflows import (
    END,
    GraphInterrupt,
    GraphRecursionError,
    MemorySaver,
    StateGraph,
    trigger_poll_workflow,
)


class TestTriggerPollWorkflow:
    def _agent(self):
        """An AsyncAgent-shaped mock whose async methods are AsyncMocks."""
        agent = MagicMock(spec=AsyncAgent)
        agent.trigger_task = AsyncMock()
        agent.get_task_output_preview = AsyncMock()
        return agent

    def test_completes_and_polls_until_output(self):
        agent = self._agent()
        task = MagicMock(conversation_id="conv-123")
        agent.trigger_task = AsyncMock(side_effect=[task])
        # pending (False) twice, then a completed output
        agent.get_task_output_preview = AsyncMock(
            side_effect=[False, False, {"answer": "ok"}]
        )

        graph = trigger_poll_workflow(agent, "hello", poll_interval=0)
        result = asyncio.run(graph.ainvoke({}, config={"thread_id": "t1"}))

        assert not result.interrupted
        assert result.state["conversation_id"] == "conv-123"
        assert result.state["output"] == {"answer": "ok"}
        agent.trigger_task.assert_awaited_once()
        assert agent.get_task_output_preview.await_count == 3
        # audit trail: trigger, then three polls (two pending + one complete)
        assert [e.node for e in result.trace] == ["trigger", "poll", "poll", "poll"]
        assert all(e.interrupted is False for e in result.trace)

    def test_retries_transient_trigger_failure(self):
        agent = self._agent()
        task = MagicMock(conversation_id="conv-1")
        agent.trigger_task = AsyncMock(side_effect=[RuntimeError("boom"), task])
        agent.get_task_output_preview = AsyncMock(return_value={"answer": "done"})

        graph = trigger_poll_workflow(
            agent, "hi", poll_interval=0, retry_max_attempts=3
        )
        result = asyncio.run(graph.ainvoke({}, config={"thread_id": "t2"}))

        assert not result.interrupted
        assert result.state["output"] == {"answer": "done"}
        # the trace records that trigger succeeded on the second attempt
        assert result.trace[0].node == "trigger"
        assert result.trace[0].attempts == 2
        assert agent.trigger_task.await_count == 2

    def test_state_is_checkpointed_for_resume(self):
        agent = self._agent()
        task = MagicMock(conversation_id="conv-9")
        agent.trigger_task = AsyncMock(side_effect=[task])
        agent.get_task_output_preview = AsyncMock(return_value={"answer": "x"})

        checkpointer = MemorySaver()
        graph = trigger_poll_workflow(
            agent, "hi", poll_interval=0, checkpointer=checkpointer
        )
        asyncio.run(graph.ainvoke({}, config={"thread_id": "t3"}))

        # the graph recorded its progress against the thread
        assert graph.get_state("t3") is not None
        history = checkpointer.list("t3")
        assert history, "expected at least one checkpoint"
        assert history[-1]["next_node"] == END


class TestStateGraph:
    def test_conditional_routing_loops_then_ends(self):
        g = StateGraph()
        seen = []

        def inc(state):
            n = state.get("n", 0) + 1
            seen.append(n)
            return {"n": n}

        g.add_node("inc", inc)
        g.set_entry_point("inc")
        g.add_conditional_edges("inc", lambda s: "inc" if s["n"] < 3 else END)

        result = asyncio.run(g.ainvoke({}))

        assert result.state["n"] == 3
        assert seen == [1, 2, 3]

    def test_interrupt_then_resume_from_checkpoint(self):
        g = StateGraph(checkpointer=MemorySaver())
        approved = {"value": False}

        async def review(state):
            if not approved["value"]:
                raise GraphInterrupt("awaiting human approval")
            return {"approved": True}

        g.add_node("review", review)
        g.set_entry_point("review")
        g.add_edge("review", END)

        # first run pauses for human review and checkpoints its state
        first = asyncio.run(g.ainvoke({}, config={"thread_id": "approval"}))
        assert first.interrupted
        assert first.trace[-1].interrupted is True
        assert first.trace[-1].output == "awaiting human approval"
        assert g.get_state("approval") is not None

        # human approves; resume from the checkpointed thread (state omitted)
        approved["value"] = True
        second = asyncio.run(g.ainvoke(config={"thread_id": "approval"}))

        assert not second.interrupted
        assert second.state["approved"] is True

    def test_recursion_limit_raises(self):
        g = StateGraph(recursion_limit=4)
        g.add_node("loop", lambda s: {"n": s.get("n", 0) + 1})
        g.set_entry_point("loop")
        g.add_edge("loop", "loop")  # never reaches END

        with pytest.raises(GraphRecursionError):
            asyncio.run(g.ainvoke({}))
