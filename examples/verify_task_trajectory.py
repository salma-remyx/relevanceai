"""
Trigger an agent task, poll its step telemetry, and run deterministic
trajectory verification over the result -- detecting loops, cascading
tool errors, missing coverage, and fabricated totals without a second
LLM call. When a run is flagged, optionally re-run it (the paper's
detection -> repair closure) using the agent's native rerun_task.

Adapted from "Real-Time Detection and Repair of LLM Agent Failures"
(arXiv:2608.02464): the deterministic-verification layer at full
fidelity; the paper's trained monitor is substituted with parameter-free
trajectory-shape detectors (see relevanceai.utils.trajectory_verifier).

Requires RAI_API_KEY / RAI_REGION / RAI_PROJECT env vars (see .env.example).
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from relevanceai import AsyncRelevanceAI
from relevanceai.utils.trajectory_verifier import verify_and_repair


AGENT_ID = os.getenv("EXAMPLE_AGENT_ID", "xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
MESSAGE = "Sum the invoice totals you can find and report the grand total."

# Optional verification knobs. Leave empty/None to skip that check.
REQUIRED_TOOLS = []  # e.g. ["invoice-search"] -- tools that MUST be called
STATED_TOTAL = None  # e.g. 412.5 -- the total the agent claims in its answer


async def main() -> None:
    async with AsyncRelevanceAI() as client:
        agent = await client.agents.retrieve_agent(agent_id=AGENT_ID)
        task = await agent.trigger_task(message=MESSAGE)

        # Poll until the run settles (same shape as trigger_and_poll_tasks,
        # but over the async path), then verify the committed trajectory.
        while not await agent.get_task_output_preview(task.conversation_id):
            print("polling...\n")
            await asyncio.sleep(5)

        report, rerun = await verify_and_repair(
            agent,
            task.conversation_id,
            required_tools=REQUIRED_TOOLS or None,
            stated_total=STATED_TOTAL,
        )

        print(report)
        if rerun is not None:
            print(f"Flagged run re-triggered: {rerun.conversation_id}")


if __name__ == "__main__":
    asyncio.run(main())
