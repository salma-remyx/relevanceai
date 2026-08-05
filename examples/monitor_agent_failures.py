"""
Monitor a running agent for mid-episode failures and repair flagged runs.

Demonstrates the FailureMonitor (relevanceai/monitoring.py), which inspects
the step telemetry returned by Agent.view_task_steps() and flags looping,
tool-error cascades, and deterministic verification failures (terminal
agent-error, missing required tools, claimed results with no tool output).
Flagged runs are then rolled back and re-run via rerun_task.
"""

import time
from dotenv import load_dotenv

load_dotenv()

from relevanceai import RelevanceAI
from relevanceai.monitoring import FailureMonitor

client = RelevanceAI()

agent_id = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

my_agent = client.agents.retrieve_agent(agent_id=agent_id)

# Trigger a task so we have a conversation to monitor.
task = my_agent.trigger_task(message="Summarise the latest news on AI agents.")

# Required tools the run must invoke at least once (optional coverage check).
monitor = FailureMonitor(loop_threshold=3, error_cascade_threshold=3)

conversation_id = task.conversation_id

# Give the agent a moment to produce steps, then inspect the telemetry.
time.sleep(5)

report = monitor.monitor(my_agent, conversation_id)
print(report.summary())

if report:
    # Roll back and re-run the flagged run, mirroring the paper's repair loop.
    rerun = monitor.repair(my_agent, conversation_id)
    print(f"Re-run triggered: {getattr(rerun, 'conversation_id', rerun)}")
