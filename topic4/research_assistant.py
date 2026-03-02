import os
from dotenv import load_dotenv, find_dotenv

import operator
from typing import Annotated, TypedDict, List, Union

from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(find_dotenv())

# setup tools
tools = [
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
    DuckDuckGoSearchRun()
]

# define state with message history
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# setup llm with tools
llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

def researcher_node(state: AgentState):
    # system instructions to ensure it uses BOTH tools
    sys_msg = SystemMessage(content=(
        "You are a research assistant. To provide a high-quality report, "
        "you MUST call BOTH Wikipedia and DuckDuckGo search before answering."
    ))
    # combine system message with conversation history
    response = llm.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}

# build the Graph
workflow = StateGraph(AgentState)

# add our custom LLM node
workflow.add_node("researcher", researcher_node)

# add tool node
workflow.add_node("tools", ToolNode(tools))

# create graph
workflow.add_edge(START, "researcher")

# conditional edge for routing
workflow.add_conditional_edges(
    "researcher",
    tools_condition, # This is a pre-built helper that checks for tool_calls
    {"tools": "tools", END: END}
)

# after tools go back to researcher
workflow.add_edge("tools", "researcher")

app = workflow.compile()

# run loop
user_input = input("Research Topic: ")
inputs = {"messages": [HumanMessage(content=user_input)]}

for output in app.stream(inputs):
    for node, state in output.items():
        print(f"\n--- Node: {node} ---")
        last_msg = state["messages"][-1]
        print(last_msg.content if last_msg.content else f"Calling tools: {last_msg.tool_calls}")