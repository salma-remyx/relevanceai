
"""
Approve tasks of an agent in bulk
"""

import time
from dotenv import load_dotenv
load_dotenv()

from relevanceai import RelevanceAI

client = RelevanceAI()

agent_id = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

message_list = [
    "Research the following company: RelevanceAI relevanceai.com",
    "Research the following company: Vividly govividly.com",
    "Research the following company: Airwallex airwallex.com",
    "Research the following company: Lark larksuite.com",
    "Research the following company: SafetyCulture safetyculture.com",
]

task_ids = [] 

my_agent = client.agents.retrieve_agent(agent_id=agent_id)

# triggers the tasks
for message in message_list: 
    task = my_agent.trigger_task(message)
    task_ids.append(task.conversation_id) # save the ids of the triggered tasks
    time.sleep(1)

tool_id = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

# approving the tasks given a list of task_ids above
for task_id in task_ids:
    approved_task = my_agent.approve_task(task_id)

    # approve only a specific tool (optional) -> use the code below instead
    approved_task = my_agent.approve_task(task_id, tool_id=tool_id)


# --- Audit each task's execution trace (read-only) --------------------------
# Adapted from A2E (Agent Auditing Engine): now that the tasks have run,
# walk the task steps returned by Agent.view_task_steps and compute
# multidimensional metrics (tool use, error recovery, efficiency, planning)
# over the tools the agent invoked. See relevanceai/utils/task_audit.py.
from relevanceai.utils.task_audit import audit_task_view

for task_id in task_ids:
    task_view = my_agent.view_task_steps(task_id)
    report = audit_task_view(task_view)
    print("[audit] {}\n{}\n".format(task_id, report.summary()))







