"""
Calibrated tool retrieval.

Large tool libraries share a "background" of generic descriptive phrasing
("this tool takes ... and returns ...") that inflates ordinary similarity and
hides the tokens that actually tell capabilities apart. This example ranks the
tools returned by ToolsManager.list_tools() with that background calibrated
out, and prints the result next to the raw (background-kept) baseline so the
difference is visible.

Adapted from SkillSight (Xiao et al., arxiv:2607.18785). See
relevanceai.calibrated_tool_retrieval for the adaptation notes.
"""

from dotenv import load_dotenv
load_dotenv()

from relevanceai import RelevanceAI
from relevanceai.calibrated_tool_retrieval import SkillLibrary

client = RelevanceAI()

query = "translate french text into english"

# ToolsManager.list_tools() returns Tool objects whose .metadata carries the
# descriptions the retriever ranks over.
tools = client.tools.list_tools()

# Wrap the live library and retrieve with shared descriptive background
# calibrated out (Semantic + Lexical Background Calibration).
library = SkillLibrary.from_tools(tools)

print(f"Query: {query}\n")
print(f"{'rank':<5}{'calibrated':<12}{'raw':<8}{'title'}")
for hit in library.retrieve(query, top_k=10):
    title = next(
        (t.metadata.title for t in tools if t.metadata.studio_id == hit.key),
        hit.key,
    )
    print(f"{hit.rank:<5}{hit.score:<12.3f}{hit.raw_score:<8.3f}{title}")
