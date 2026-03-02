# Topic 2: LangGraph Agents with Local LLMs

AI Assistance: Used ClaudeCode to help complete assignment and generate README. Assignment Answers were AI assisted, but not completely AI generated.

## Files

### Scripts & Notebooks

- **[langgraph_simple_llama_agent.py](langgraph_simple_llama_agent.py)** — A simple LangGraph agent that creates an interactive chatbot using Llama-3.2-1B-Instruct. Implements device auto-detection (CUDA > MPS > CPU), wraps the model in a `HuggingFacePipeline` for LangChain compatibility, and builds a state graph with nodes for user input, LLM inference, and response printing. The graph routes to END when the user types quit/exit/q, otherwise loops back. Also saves a Mermaid diagram of the graph as `lg_graph.png`.

- **[topic2.ipynb](topic2.ipynb)** — Jupyter notebook containing two implementations: (1) a copy of the simple LangGraph agent above, and (2) a more advanced persistent chat system that supports two models (Llama-3.2-1B-Instruct and Qwen2.5-1.5B-Instruct) in a multi-party conversation. Uses `SqliteSaver` to checkpoint conversation state to `checkpoints.sqlite`, enabling conversation resumption across program restarts. Routes to Qwen when the user types "Hey Qwen", otherwise defaults to Llama.

### Other Files

- **[requirements.txt](requirements.txt)** — Python dependencies: `torch`, `transformers`, `langchain-huggingface`, `langgraph`, `grandalf`.

---

# Assignment Answers

## Part 2

When I typed empty inputs, nothing happened. Though I know that with some smaller models it will hallucinate answers.
