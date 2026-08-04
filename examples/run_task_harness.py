"""
Run an agent task inside the task harness.

Demonstrates wrapping an existing Agent in deterministic scaffolding - an
LLM-as-judge loop with bounded retry, a human-in-the-loop confirmation
gate, and a per-node audit trail - so a small model behaves reliably and
traceably. Adapted from the harness-engineering patterns in
arxiv:2607.14707v1. See ``examples/task_harness.py`` for the mechanism.
"""

from dotenv import load_dotenv

load_dotenv()

from relevanceai import RelevanceAI

try:  # works as ``python -m examples.run_task_harness`` or after install
    from examples.task_harness import TaskHarness, default_judge
except ImportError:  # works as ``python examples/run_task_harness.py``
    from task_harness import TaskHarness, default_judge

client = RelevanceAI()

agent = client.agents.retrieve_agent(agent_id="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")

harness = TaskHarness(agent, judge=default_judge)

output, trail = harness.run(
    message="Summarise the latest activity on this account.",
    max_retries=2,
    poll_sleep=5.0,
)

print("Final output:", output)
print("Audit trail:")
print(trail.to_json())
