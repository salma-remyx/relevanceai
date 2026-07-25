
"""
Visualise an agent task as a provenance / dataflow graph.

Turns the chronological TaskView returned by ``Agent.view_task_steps`` into a
provenance graph — tool calls become computational actions, their inputs and
outputs become data artifacts, and the dependencies between them become edges.
The graph is printed as Graphviz DOT. Two conversations are then compared on a
shared canvas via a joined quotient graph that aligns recurring tools across
executions.

Idea adapted from *AgentTrails: Towards Trust and Reuse for Agentic Tasks*
(arXiv:2607.18816).
"""

from dotenv import load_dotenv

load_dotenv()

from relevanceai import RelevanceAI
from relevanceai.provenance import (
    build_provenance,
    quotient_graph,
    recurring_patterns,
)

client = RelevanceAI()

agent_id = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

my_agent = client.agents.retrieve_agent(agent_id=agent_id)

# 1. Build a provenance graph from a single task trajectory, wired directly
#    after Agent.view_task_steps, and render it as DOT.
conversation_id = "yyyyyyyy-yyyy-yyyy-yyyy-yyyyyyyyyyyy"
task_view = my_agent.view_task_steps(conversation_id=conversation_id)
graph = build_provenance(task_view, conversation_id=conversation_id)

print(graph.to_dot())

# 2. Compare several executions of the same agent: the quotient graph merges
#    recurring tools/artifacts and counts how many trajectories each appears in.
conversation_ids = [
    conversation_id,
    "zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz",
]
graphs = [
    build_provenance(
        my_agent.view_task_steps(conversation_id=cid),
        conversation_id=cid,
    )
    for cid in conversation_ids
]

joined = quotient_graph(graphs)
print(joined.to_dot())

# 3. Surface recurring tool-use patterns across the executions.
for pattern in recurring_patterns(graphs, n=2):
    print(pattern)
