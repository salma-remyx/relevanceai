## Relevance AI - The home of your AI Workforce

🔥 Use Relevance to build AI agents for your AI workforce

[Sign up for a free account ->](https://app.relevanceai.com)

## 🧠 Documentation

| Type      | Link |
| ------------- | ----------- |
| Home Page | [Home Page](https://relevanceai.com/) |
| Platform | [Platform](https://app.relevanceai.com/) |
| Developer Documentation | [Documentation](https://sdk.relevanceai.com/) |

# Relevance AI SDK

Welcome to the Relevance AI SDK! This guide will help you set up and start using the SDK to interact with your AI agents, tools, and knowledge.

## Installation

To get started, you'll need to install the RelevanceAI library in a Python 3 environment. Run the following command in your terminal:

```bash
pip install relevanceai
```

## Create an Account

Before using the SDK, ensure you have an account with Relevance AI.

1. Sign up for a free account at [Relevance AI](https://app.relevanceai.com) and log in.
2. Create a new secret key at [SDK Login](https://app.relevanceai.com/login/sdk). Scroll to the bottom of the integrations page, click on "+ Create new secret key," and select "Admin" permissions.

## Set Up Your Client

To interact with Relevance AI, you'll need to set up a client. Start by importing the library:

```python
from relevanceai import RelevanceAI
client = RelevanceAI()
```

### Validate Client Credentials

You can validate your client credentials by storing them as environment variables and loading them into your project using `python-dotenv` or the `os` library:

```env
RAI_API_KEY=
RAI_REGION=
RAI_PROJECT=
```

```python
from dotenv import load_dotenv
load_dotenv()

from relevanceai import RelevanceAI
client = RelevanceAI()
```

Alternatively, pass the credentials directly to the client:

```python
from relevanceai import RelevanceAI
client = RelevanceAI(
    api_key="your_api_key", 
    region="your_region", 
    project="your_project"
)
```

You are now ready to start using Relevance AI via the Python SDK.

## Quickstart

### Using Agents & Tasks

List all the agents in your project:

```python
from relevanceai import RelevanceAI
client = RelevanceAI()

agents = client.agents.list_agents()
print(agents)
# Example output: [Agent(agent_id="xxxxxxxx", name="Sales Qualifier"), ...]
```

Retrieve and interact with a specific agent:

```python
my_agent = client.agents.retrieve_agent(agent_id="xxxxxxxx")

message = "Let's qualify this lead:\n\nName: Ethan Trang\n\nCompany: Relevance AI\n\nEmail: ethan@relevanceai.com"

# Trigger a task with the agent
task = my_agent.trigger_task(message=message)
print(f"Task started with ID: {task.conversation_id}")

# View task progress
steps = my_agent.view_task_steps(conversation_id=task.conversation_id)
```

### Using Tools

List all the tools in your project:

```python
tools = client.tools.list_tools()
print(tools)
# Example output: [Tool(tool_id="xxxxxxxx", title="Search Website"), ...]
```

Retrieve and interact with a specific tool:

```python
my_tool = client.tools.retrieve_tool(tool_id="xxxxxxxx")

# Check tool parameters schema
params_schema = my_tool.get_params_schema()

# Trigger the tool
result = my_tool.trigger(params={"search_query": "AI automation"})
```

### Managing Knowledge Sets

Work with knowledge sets to store and retrieve data:

```python
# List knowledge sets
knowledge_sets = client.knowledge.list_knowledge()

# Retrieve data from a knowledge set
data = client.knowledge.retrieve_knowledge(knowledge_set="my_dataset")
```

### Managing Tasks

Track and manage ongoing tasks:

```python
# Get task metadata
metadata = client.tasks.get_metadata(conversation_id="xxxxxxxx")

# Delete a completed task
client.tasks.delete_task(conversation_id="xxxxxxxx")
```

## Explore More

Explore all the methods available for agents, tasks, tools, and knowledge with the [documentation](https://sdk.relevanceai.com/)

## Auditing Task Execution

Once an agent has run a task, walk the task steps returned by `view_task_steps` to compute a multidimensional audit of how the agent behaved — tool use, error recovery, execution efficiency, and task planning — instead of a single pass/fail. This is a read-only utility; it never calls back into the API.

```python
from relevanceai.utils.task_audit import audit_task_view

task_view = my_agent.view_task_steps(conversation_id=task.conversation_id)
report = audit_task_view(task_view)

print(report.summary())      # compact per-axis summary
metrics = report.to_dict()   # nested dict for logging / export
```

`report.tool_use` (calls, distinct tools, distribution, confirmations), `report.error_recovery` (tool/agent errors, recovered, recovery rate), `report.efficiency` (success rate, planning overhead, duration), and `report.planning` (user/agent messages, tool-call ratio) characterize the run across the four axes.

> Task audit — adapted from A²E: An End-to-End Agent Auditing Engine (https://arxiv.org/abs/2608.07346v1).
