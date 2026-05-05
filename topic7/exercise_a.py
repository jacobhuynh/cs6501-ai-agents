# Exercise A: Discover the Asta MCP Tools
#
# Q: Which tool would you use to find all papers about "transformer attention mechanisms"?
# A: search_papers_by_relevance — it takes a keyword and returns papers ranked by relevance,
#    making it ideal for broad topic searches like this.
#
# Q: Which tool would you use to find who else published in the same area as a specific author?
# A: The best approach combines two tools:
#      1. get_author_papers — retrieve the target author's papers to learn their research area.
#      2. search_papers_by_relevance — search that area keyword; the returned papers include
#         author lists, revealing other researchers working in the same space.
#    There is no single "find similar authors" tool, so this two-step workflow is the right one.

import json
import os

import requests

MCP_URL = "https://asta-tools.allen.ai/mcp/v1"

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"],
}

payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {},
}

resp = requests.post(MCP_URL, headers=headers, json=payload)
resp.raise_for_status()

# The server returns SSE format: lines like "data: {...}"
raw = resp.text
data_line = next(line for line in raw.splitlines() if line.startswith("data:"))
tools = json.loads(data_line[len("data:"):].strip())["result"]["tools"]

for tool in tools:
    schema = tool.get("inputSchema", {})
    required_fields = set(schema.get("required", []))
    properties = schema.get("properties", {})

    # Pull a one-line description from the first non-empty line of the docstring
    raw_desc = tool.get("description", "").strip()
    one_liner = next((ln.strip() for ln in raw_desc.splitlines() if ln.strip()), "No description")

    required_params = []
    optional_params = []
    for param, info in properties.items():
        ptype = info.get("type", "any")
        if isinstance(ptype, list):
            ptype = " | ".join(ptype)
        entry = f"{param} ({ptype})"
        if param in required_fields:
            required_params.append(entry)
        else:
            optional_params.append(entry)

    print(f"Tool: {tool['name']}")
    print(f"  Description: {one_liner}")
    if required_params:
        print(f"  Required: {', '.join(required_params)}")
    if optional_params:
        print(f"  Optional: {', '.join(optional_params)}")
    print()
