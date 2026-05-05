import json
import os

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MCP_URL = "https://asta-tools.allen.ai/mcp/v1"
MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"],
}

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM_PROMPT = (
    "You are a research assistant with access to Semantic Scholar via the Asta API. "
    "Use the available tools to look up papers, authors, citations, and references. "
    "Always base your answers on what the tools return rather than prior knowledge."
)

_req_id = 0


def _mcp_post(payload: dict) -> dict:
    global _req_id
    _req_id += 1
    payload["id"] = _req_id
    resp = requests.post(MCP_URL, headers=MCP_HEADERS, json=payload)
    resp.raise_for_status()
    data_line = next(l for l in resp.text.splitlines() if l.startswith("data:"))
    return json.loads(data_line[5:].strip())


def get_asta_tools() -> list[dict]:
    """Fetch tool schemas from MCP and convert to OpenAI function-calling format."""
    result = _mcp_post({"jsonrpc": "2.0", "method": "tools/list", "params": {}})
    tools = result["result"]["tools"]
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["inputSchema"],
            },
        }
        for tool in tools
    ]


MAX_TOOL_RESULT_CHARS = 8000


def call_asta_tool(name: str, arguments: dict) -> str:
    """Execute a tools/call and return the text content, truncated to avoid context overflow."""
    try:
        result = _mcp_post({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })
        content = result["result"]["content"]
        text = "\n".join(item["text"] for item in content if item.get("type") == "text")
        if len(text) > MAX_TOOL_RESULT_CHARS:
            text = text[:MAX_TOOL_RESULT_CHARS] + "\n...[truncated]"
        return text
    except Exception as e:
        return f"Error calling {name}: {e}"


def chat(user_message: str, messages: list[dict], tools: list[dict]) -> str:
    """Run one full chatbot turn, handling any number of tool call rounds."""
    messages.append({"role": "user", "content": user_message})

    while True:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            # Append the assistant message with tool_calls before tool results
            messages.append(msg)

            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                print(f"  [tool call] {name}({json.dumps(args)})")

                tool_result = call_asta_tool(name, args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })
        else:
            answer = msg.content
            messages.append({"role": "assistant", "content": answer})
            return answer


def main():
    print("Loading Asta tools from MCP server...")
    tools = get_asta_tools()
    print(f"Loaded {len(tools)} tools: {[t['function']['name'] for t in tools]}\n")

    test_queries = [
        "Find recent papers about large language model agents",
        "Who wrote Attention is All You Need and what else have they published?",
        "What papers cite the original BERT paper?",
        "Summarize the references used in the ReAct paper",
    ]

    for query in test_queries:
        print("=" * 60)
        print(f"User: {query}")
        print("-" * 60)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        answer = chat(query, messages, tools)
        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    main()
