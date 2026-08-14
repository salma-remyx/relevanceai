
"""
Approve tasks of an agent in bulk.

After approving a batch of tasks this script also runs an evidence-grounding
check over each task's trace (see ``relevanceai.utils.task_evidence``): it
verifies that the findings the agent reported are actually grounded in the
data its tools produced, and flags the earliest weak claim for review.
"""

import time

from relevanceai import RelevanceAI
from relevanceai.utils.task_evidence import check_grounding, format_report


def run_evidence_check(agent, conversation_ids):
    """Check whether each approved task's claims are grounded in its tool data.

    Walks ``agent.view_task_steps`` for every conversation id, builds the typed
    evidence graph for the trace, and reports claim-evidence consistency. The
    map returned is ``{conversation_id: GroundingReport}``.
    """
    results = {}
    for conversation_id in conversation_ids:
        task_view = agent.view_task_steps(conversation_id)
        report = check_grounding(task_view)
        results[conversation_id] = report
        print(format_report(report))
    return results


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

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
        task_ids.append(task.conversation_id)  # save the ids of the triggered tasks
        time.sleep(1)

    tool_id = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

    # approving the tasks given a list of task_ids above
    for task_id in task_ids:
        approved_task = my_agent.approve_task(task_id)

        # approve only a specific tool (optional) -> use the code below instead
        approved_task = my_agent.approve_task(task_id, tool_id=tool_id)

    # Validate that each approved task's findings are grounded in its tool data.
    run_evidence_check(my_agent, task_ids)
