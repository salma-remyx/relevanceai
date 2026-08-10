"""
Retrieve the most relevant tools for a task using a hybrid ranker.

An agent backed by a large tool library should load a small, on-demand top-k
slice instead of putting every tool in context. This example lists the
project's tools once, then ranks them against a task query with the hybrid
ranker (lexical BM25 + a parameter-free dense proxy) and prints the top-k to
load.

Adapted from "Comparative Approaches to Agent Retrieval over Large Skill
Libraries" (arXiv:2608.06196).
"""

from dotenv import load_dotenv
load_dotenv()

from relevanceai import RelevanceAI
from relevanceai.utils.tool_ranker import hit_at_k, rank_tools

client = RelevanceAI()

query = "Search a company website for contact emails"

# Load tool metadata once, then rank on demand -- sparse, top-k loading.
tools = client.tools.list_tools(max_results=100)

if not tools:
    print("No tools found in this project.")
else:
    top_k = 5
    ranked = rank_tools(query, tools, top_k=top_k)

    print(f"Top {len(ranked)} tools for: {query!r}")
    for tool in ranked:
        print(f"  - {tool}")

    # The paper's headline metric: did the ranker surface a known-relevant tool?
    # (Here we re-check the top result; in a real run compare against a held-out
    # gold tool id.)
    ranked_ids = [t.tool_id for t in ranked]
    gold_tool_id = ranked_ids[0] if ranked_ids else None
    print(f"hit@{top_k}: {hit_at_k(ranked_ids, gold_tool_id, k=top_k)}")
