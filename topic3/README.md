# Topic 3: Tool-Calling Agents with Ollama and OpenAI

AI Assistance: Used ClaudeCode to help complete assignment and generate README. Assignment Answers were AI assisted, but not completely AI generated.

## Files

### Scripts & Notebooks

- **[topic3_ollama.ipynb](topic3_ollama.ipynb)** — Jupyter notebook that benchmarks the Llama 3.2:1B model locally via Ollama on MMLU subjects (abstract_algebra and college_computer_science). Writes two standalone Python scripts (`program1.py`, `program2.py`) that each query the local Ollama server for one subject, then runs them sequentially and in parallel (using the shell `time` command) to compare throughput and GPU resource sharing behavior.

- **[topic3_openai.ipynb](topic3_openai.ipynb)** — Jupyter notebook implementing a tool-calling agent with OpenAI's `gpt-4o-mini`. Part 4 builds a manual agent loop with four tools: `get_weather`, `calculator`, `count_letters`, and `convert_currency`, running 6 progressively complex test cases (single tool, no tools, parallel calls, multi-tool chaining, and a 5-turn limit test). Part 5 refactors the agent into a LangGraph `StateGraph` with a MemorySaver checkpointer, demonstrating multi-turn conversation with persistent memory.

### Other Files

- **[Diagram.png](Diagram.png)** — Flowchart of the LangGraph agent workflow: Start → Agent → decision (tools needed? → Tools node → loop back, or final answer → End).

---

# Assignment Answers

## Part 1

After running both the sequential and parallel execution, I noticed a couple key details.

The first is that the real time is significantly higher than the user time. This is most likely due to the fact that the python scripts are small and don't actually have to do a lot, but still have to wait for the Ollama server to respond (shown by the difference between the user time and real time).

The second is that in sequential mode, each script ran at 3.0 iterations per second while in parallel they only ran at 1.65 iterations per second. This is most likely due to the fact that Ollama has to share the same GPU resources between the parallel scripts, meaning that since Ollama has to process both requests at the same time, it takes longer to complete.

The last is that parallel was faster than sequential, but not by much. This was interesting because I would assume running in parallel should be 50% faster given 2 scripts. However, this is most likely because the GPU is being bottlenecked so the time reduction isn't as significant as it should be in theory.

On a side note, abstract_algebra received a 0/100 which is most likely due to one of two reasons (or maybe both). The first is that the model used simply struggled on the topic, or that the model produced the answer in the wrong format (not just A, B, C, or D) resulting in an incorrect answer every time.

## Part 2.3

We simply create an OpenAI object and run an API call with parameters denoting the model we want and the current message history (preloaded with a user prompt). We also limit the maximum tokens that the API call can use to 5.

## Part 3

My agent was able to accurately determine when to use tools and when not to.

## Part 4

In tests 4 and 6, the agent was able to identify separate tasks and execute them in parallel. In addition, in test 6, the agent was able to successfully follow an ordered sequence of tasks (currency conversion, find weather, then multiply). However, despite efforts to make the agent hit the upper limit of 5 iterations, I was unable to produce a task to reach that limit. The main problem I ran into while trying to do this was the fact that the agent was able to plan and perform multiple tasks in each iteration.

## Part 6

The agent is already running multiple tool calls per iteration, but is running them sequentially. One opportunity for parallelization is to have asyncronous tool execution, allowing all of these tool calls to run in parallel.
