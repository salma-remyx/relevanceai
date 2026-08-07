"""
Monitor a running agent task for mid-episode failures.

Demonstrates the deterministic failure-detection layer adapted from
"Real-Time Detection and Repair of LLM Agent Failures" (arXiv:2608.02464).
After triggering a task, each poll fetches the per-step telemetry
(``view_task_steps``) and runs lightweight, parameter-free detectors that
flag looping tool calls, cascading tool errors, missing required tool
calls, and totals that do not reconcile with the tool results actually
received. Run with a real AGENT_ID, or inspect ``monitor_task_view`` on a
fetched ``TaskView`` directly.
"""

from __future__ import annotations

import asyncio
import os
import time

from dotenv import load_dotenv

from relevanceai import AsyncRelevanceAI, RelevanceAI
from relevanceai.utils.agent_failure_monitor import (
    MonitorConfig,
    monitor_task_view,
)

load_dotenv()

AGENT_ID = os.getenv("AGENT_ID", "xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
POLL_INTERVAL = 5


def run_sync(agent_id: str) -> None:
    """Trigger a task, poll to completion, then monitor its step telemetry."""
    client = RelevanceAI()
    agent = client.agents.retrieve_agent(agent_id=agent_id)
    task = agent.trigger_task(
        message="Search the site, then report the total number of results found."
    )

    while not agent.get_task_output_preview(task.conversation_id):
        print("polling...\n")
        time.sleep(POLL_INTERVAL)

    task_view = agent.view_task_steps(task.conversation_id)
    report = monitor_task_view(
        task_view,
        config=MonitorConfig(
            enable_coverage=True,
            required_tools={"search"},
            enable_total_verification=True,
            stated_total=10,
            total_key="count",
        ),
    )
    print(report.as_dict())
    if report.failed:
        print(
            f"Detected {len(report.findings)} failure signal(s); "
            f"recommended repair: {report.recommended_repair}"
        )


async def run_async(agent_id: str) -> None:
    """Async variant -- same monitor wired into the async polling loop."""
    client = AsyncRelevanceAI()
    agent = await client.agents.retrieve_agent(agent_id=agent_id)
    task = await agent.trigger_task(
        message="Search the site, then report the total number of results found."
    )

    while not await agent.get_task_output_preview(task.conversation_id):
        print("polling...\n")
        await asyncio.sleep(POLL_INTERVAL)

    task_view = await agent.view_task_steps(task.conversation_id)
    report = monitor_task_view(
        task_view,
        config=MonitorConfig(enable_coverage=True, required_tools={"search"}),
    )
    print(report.as_dict())


if __name__ == "__main__":
    run_sync(AGENT_ID)
