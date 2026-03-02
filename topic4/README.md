# Part 3

### 1. What features of Python does ToolNode use to dispatch tools in parallel? What kinds of tools would most benefit from parallel dispatch?

ToolNode uses the asyncio library to execute tools in parallel. Tools that are I/O-bound and spend time waiting on external systems would benefit from parallel dispatch (like querying databases or making requests to external APIs).

### 2. How do the two programs handle special inputs such as "verbose" and "exit"?

In both programs there's a node before the LLM node that checks for "verbose" and "exit". If "verbose" is sent, it updates the command variable in the state which then routes it back to the input. If "exit" is sent, it updates the command variable in state which then routes to the end node.

### 3. Compare the graph diagrams of the two programs. How do they differ if at all?

The graph diagram for the ToolNode agent contains a loop from the agent to the tool node back to the agent and then a path from the agent to the end. This differs from teh ReAct agent diagram which has a linear path from input to agent to output.

### 4. What is an example of a case where the structure imposed by the LangChain react agent is too restrictive and you'd want to pursue the toolnode approach?

One example is if you want to conditionally modify the output of the tool before passing it back to the main agent. For example if your tool pulls a huge document, you might want to route it to a summarizing agent before sending it back to the main agent, but if you use ReAct, you can't do this.
