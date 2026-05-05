# Topic 7: MCP & Asta Tools

## Exercise A

**Q: Which tool would you use to find all papers about "transformer attention mechanisms"?**

`search_papers_by_relevance` — takes a keyword and returns papers ranked by relevance.

**Q: Which tool would you use to find who else published in the same area as a specific author?**

No single tool does this directly. Use `get_author_papers` to get their papers and identify their research area, then `search_papers_by_relevance` on that area — the returned papers include author lists revealing others working in the same space.

**Output:**

```
Tool: get_paper
  Description: Get details about a paper by its id.
  Required: paper_id (string)
  Optional: fields (string)

Tool: get_paper_batch
  Description: Get details about a list of papers by their ids.
  Required: ids (array)
  Optional: fields (string)

Tool: get_citations
  Description: Get details about the papers that cite this paper (i.e. papers in whose bibliography this paper appears)
  Required: paper_id (string)
  Optional: fields (string), limit (integer), publication_date_range (string)

Tool: search_authors_by_name
  Description: Search for authors by name.
  Required: name (string)
  Optional: fields (string), limit (integer)

Tool: get_author_papers
  Description: Get papers written by this author.
  Required: author_id (string)
  Optional: paper_fields (string), limit (integer), publication_date_range (string)

Tool: search_papers_by_relevance
  Description: Search for papers by keyword relevance.
  Required: keyword (string)
  Optional: fields (string), limit (integer), publication_date_range (string), venues (string)

Tool: search_paper_by_title
  Description: Search for papers by title.
  Required: title (string)
  Optional: fields (string), publication_date_range (string), venues (string)

Tool: snippet_search
  Description: Search for text snippets that most closely match the query.
  Required: query (string)
  Optional: limit (integer), venues (string), paper_ids (string), inserted_before (string)
```

## Exercise B

**Output:**

```
============================================================
Drill 1: Recent LLM Agent Papers
============================================================
1. InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents (2024)
   Authors: Qiusi Zhan, Zhixiang Liang, Zifan Ying et al. (+1)
   Abstract: Recent work has embodied LLMs as agents, allowing them to access tools, perform actions, and interact with external cont...

2. Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning (2025)
   Authors: Sikuan Yan, Xiufeng Yang, Zuchao Huang et al. (+7)
   Abstract: Large Language Models (LLMs) have demonstrated impressive capabilities across a wide range of NLP tasks, but they remain...

3. A Survey of Large Language Model Agents for Question Answering (2025)
   Authors: Murong Yue
   Abstract: This paper surveys the development of large language model (LLM)-based agents for question answering (QA). Traditional a...

4. TimeCAP: Learning to Contextualize, Augment, and Predict Time Series Events with Large Language Model Agents (2025)
   Authors: Geon Lee, Wenchao Yu, Kijung Shin et al. (+2)
   Abstract: Time series data is essential in various applications, including climate modeling, healthcare monitoring, and financial...

5. Large language model agents can use tools to perform clinical calculations (2025)
   Authors: Alex J. Goodell, Simon N. Chu, D. Rouholiman et al. (+1)
   Abstract: Large language models (LLMs) can answer expert-level questions in medicine but are prone to hallucinations and arithmeti...

============================================================
Drill 2: Papers Citing BERT (2023+)
============================================================
Results returned: 10

1. Mispronunciation detection and diagnosis based on large language models (2026)
   Authors: Yanlu Xie, Huihang Zhong et al.

2. Advanced technology-driven few-shot relation extraction: Challenges, opportunities, and future outlook (2026)
   Authors: Daiyi Li, Yaoyao Liang et al.

3. TactileFormer: A feature-fused CNN-Transformer model for few-shot tactile perception (2026)
   Authors: Tianci Xue, Kaiyan Xie et al.

4. An interpretable multimodal transformer for medical report generation via hierarchical semantics and clinical labeling (2026)
   Authors: Jia Sheng Yang, Chenbo Xia et al.

5. Advancing sustainable development goals through multilingual text summarization: A transformer-based approach (2026)
   Authors: Atul Kumar, Shashi Kant Gupta et al.

============================================================
Drill 3: References of the ReAct Paper (sorted by year)
============================================================
Total references: 40

  [1965] L.S. Vygotsky and the problem of localization of functions
  [1984] Working memory
  [2009] Self-and Social-Regulation Social Interaction and the Development of Social Understanding and Executive Functions
  [2010] Vygotsky, Luria, and the social brain.
  [2013] The ACL Anthology Network Corpus as a Resource for NLP-based Bibliometrics
  [2015] Inner Speech: Development, Cognitive Functions, Phenomenology, and Neurobiology
  [2018] HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering
  [2018] FEVER: a Large-scale Dataset for Fact Extraction and VERification
  [2018] Thinking and Speech
  [2019] ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks
  [2019] ELI5: Long Form Question Answering
  [2020] Imitating Interactive Intelligence
  [2020] ALFWorld: Aligning Text and Embodied Environments for Interactive Learning
  [2020] Keep CALM and Explore: Language Models for Action Generation in Text-based Games
  [2020] Language Models are Few-Shot Learners
  [2020] Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
  [2020] A Simple Language Model for Task-Oriented Dialogue
  [2021] WebGPT: Browser-assisted question-answering with human feedback
  [2021] Show Your Work: Scratchpads for Intermediate Computation with Language Models
  [2021] LILA: Language-Informed Latent Actions
  [2021] Adaptive Information Seeking for Open-Domain Question Answering
  [2021] Language Models are Few-Shot Butlers
  [2022] Improving alignment of dialogue agents via targeted human judgements
  [2022] Text and Patterns: For Effective Chain of Thought, It Takes Two to Tango
  [2022] Faithful Reasoning Using Large Language Models
  [2022] BlenderBot 3: a deployed conversational agent that continually learns to responsibly engage
  [2022] Inner Monologue: Embodied Reasoning through Planning with Language Models
  [2022] WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents
  [2022] Rationale-Augmented Ensembles in Language Models
  [2022] Large Language Models are Zero-Shot Reasoners
  [2022] Least-to-Most Prompting Enables Complex Reasoning in Large Language Models
  [2022] Selection-Inference: Exploiting Large Language Models for Interpretable Logical Reasoning
  [2022] PaLM: Scaling Language Modeling with Pathways
  [2022] Do As I Can, Not As I Say: Grounding Language in Robotic Affordances
  [2022] Language Models that Seek for Knowledge: Modular Search & Generation for Dialogue and Prompt Completion
  [2022] Self-Consistency Improves Chain of Thought Reasoning in Language Models
  [2022] Internet-augmented language models through few-shot prompting for open-domain question answering
  [2022] Pre-Trained Language Models for Interactive Decision-Making
  [2022] Chain of Thought Prompting Elicits Reasoning in Large Language Models
  [2022] Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents
```

**Q: What differences did you notice in the structure of results across the three tools?**

`search_papers_by_relevance` returns multiple `content` items — one per paper — so you iterate over all of them. `get_citations` also returns one item per result but each is wrapped in a `{"citingPaper": {...}}` envelope, adding an extra nesting level. `get_paper` returns a single `content` item with the full paper object; when `references` is requested, those reference objects only contain `paperId` and `title` with no year, requiring a follow-up `get_paper_batch` call to enrich them.

**Q: How did you handle the JSON returned inside the `content[0]["text"]` field?**

Each `content[i]["text"]` is a JSON string, so we called `json.loads(item["text"])` on each item to get a usable dict. For Drill 1 and 2 we looped over all items in `content`; for Drill 3 we only needed `content[0]` since the full paper comes back as one object.

## MCP Closing Discussion

**Q: You wrote tool schemas by hand in concept, then saw MCP provide them dynamically. What does this automation buy you? What does it cost?**

Dynamic schema loading means the client never needs to know what tools exist ahead of time — add or remove a tool on the server and every connected agent picks it up on next startup with no code change. It also eliminates a whole class of bugs where a hardcoded schema drifts from the actual implementation. The cost is a new network dependency at startup (if `tools/list` fails, the agent can't run at all), and the LLM now sees schemas it may not know how to use well — a poorly written server description can cause the model to misuse or ignore a tool in ways that are hard to debug.

**Q: The Asta tools return rich JSON. How did you decide what to include in the context window and what to discard? What happened to quality when you passed everything vs. a summary?**

We only requested fields we actually needed (e.g. `title,abstract,year,authors,citationCount`) rather than the full paper object, and truncated abstracts to ~300 characters in the report prompt. Passing everything — full reference lists, all author metadata, complete abstracts — quickly fills the context window with redundant or low-signal content, which tends to dilute the LLM's attention and produce more generic, less focused summaries. Selective field requests at the MCP call level are cheaper and produce better output than filtering after the fact.

**Q: In Exercise D you controlled the tool-calling order. What would it take to let the LLM decide the order? What could go wrong?**

To let the LLM drive the order you'd wire it up like Exercise C — give it all the tools and a high-level goal and let it emit tool calls iteratively. The main risks: the model might call `get_author_papers` before it has an author ID, causing an error it has to recover from; it might make redundant calls (fetching the same paper twice); and it might stop early if intermediate results look "good enough" without completing the full neighborhood. You'd also need a loop limit to prevent runaway tool chains. The deterministic order in Exercise D sidesteps all of this at the cost of flexibility — it can only do exactly what it was programmed to do.

**Q: MCP is a relatively young standard. What would you want a mature MCP ecosystem to offer that is not available today?**

A few things would meaningfully raise the floor: standardized authentication (right now each server does its own thing — Asta uses `x-api-key`, others use OAuth, bearer tokens, etc.); tool versioning so a client can pin to a schema version and not break on server updates; streaming results for long-running calls instead of blocking on a single SSE response; and a registry or discovery mechanism so an agent could search for "what MCP servers expose paper search?" rather than requiring the URL to be hardcoded. Rate limit and quota metadata in the schema would also help agents plan how many calls they can afford to make.

## Exercise C

**Output:**

```
Loading Asta tools from MCP server...
Loaded 8 tools: ['get_paper', 'get_paper_batch', 'get_citations', 'search_authors_by_name', 'get_author_papers', 'search_papers_by_relevance', 'search_paper_by_title', 'snippet_search']

============================================================
User: Find recent papers about large language model agents
------------------------------------------------------------
  [tool call] search_papers_by_relevance({"keyword": "large language model agents", "limit": 5, "fields": "title,url,year,authors"})
Assistant: Here are some recent papers on large language model agents:

1. InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated LLM Agents (2024)
2. Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via RL (2025)
3. A Survey of Large Language Model Agents for Question Answering (2025)
4. TimeCAP: Learning to Contextualize, Augment, and Predict Time Series Events with LLM Agents (2025)
5. Large language model agents can use tools to perform clinical calculations (2025)

============================================================
User: Who wrote Attention is All You Need and what else have they published?
------------------------------------------------------------
  [tool call] search_paper_by_title({"title": "Attention is All You Need", "fields": "url"})
  [tool call] get_citations({"paper_id": "204e3073870fae3d05bcbc2f6a8e263d9b72e776", "fields": "authors", "limit": 1})
  [tool call] get_author_papers({"author_id": "120241018", "paper_fields": "url,year,authors", "limit": 10})
Assistant: The paper was authored by Vaswani et al. [list of other publications returned]

============================================================
User: What papers cite the original BERT paper?
------------------------------------------------------------
  [tool call] search_paper_by_title({"title": "BERT: Pre-training of Deep Bidirectional Transformers...", "fields": "title"})
  [tool call] get_citations({"paper_id": "df2b0e26d0599ce3e70df8a9da02e51594e0e992", "fields": "title,authors,url,year", "limit": 10})
Assistant: Here are 10 papers citing BERT (all 2026):
1. Mispronunciation detection and diagnosis based on large language models
2. Advanced technology-driven few-shot relation extraction
3. TactileFormer: A feature-fused CNN-Transformer model for few-shot tactile perception
4. An interpretable multimodal transformer for medical report generation
5. Advancing sustainable development goals through multilingual text summarization
[+ 5 more]

============================================================
User: Summarize the references used in the ReAct paper
------------------------------------------------------------
  [tool call] search_paper_by_title({"title": "ReAct", "fields": "references"})
  [tool call] get_paper({"paper_id": "99832586d55f540f603637e458a292406a0ed75d", "fields": "references"})
Assistant: The ReAct paper references 74 works covering: chain-of-thought prompting, embodied agents,
web interaction, dialogue alignment, and zero-shot reasoning in LLMs.
```

## Exercise D

**Output** (`python3 exercise_d.py ARXIV:2210.03629`):

```
# Research Report on "ReAct: Synergizing Reasoning and Acting in Language Models"

The paper "ReAct: Synergizing Reasoning and Acting in Language Models" by Shunyu Yao et al. (2022)
investigates the integration of reasoning and acting capabilities in large language models (LLMs).
The authors propose a novel approach, ReAct, which interleaves reasoning traces with task-specific
actions to enhance the model's performance in language understanding and decision-making tasks. By
allowing LLMs to generate reasoning paths and action plans simultaneously, ReAct improves human
interpretability and reduces issues like hallucination and error propagation.

## Foundational Works

1. **Language Models are Few-Shot Learners** (2020) — Established LLM few-shot effectiveness,
   setting the stage for ReAct's reliance on prompted reasoning.
2. **Chain of Thought Prompting Elicits Reasoning in Large Language Models** (2022) — Directly
   motivates ReAct's interleaved reasoning traces.
3. **Working Memory** (1984) — Cognitive psychology basis for ReAct's memory-like reasoning model.
4. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (2020) — Informs ReAct's
   action component for accessing external knowledge.
5. **PaLM: Scaling Language Modeling with Pathways** (2022) — Contextualizes the LLM capabilities
   ReAct builds upon.

## Recent Developments

1. Enhancing LLMs for knowledge graph QA via structured reasoning path-augmented prompting (2026)
2. Multi-agent framework for schema-guided reasoning and tool-augmented interaction (2026)
3. MedNegotiator: Automating requirements negotiation in healthcare using Generative AI agents (2026)
4. Efficient LLM-Based Subgraph Retrieval for Multi-Hop Knowledge Base Question Answering (2026)
5. Planner Matters! An Efficient Multi-agent Collaboration Framework for Long-horizon Planning (2026)

## Author Profiles

- **Shunyu Yao** — Tree of Thoughts: Deliberate Problem Solving with LLMs (2023)
- **Jeffrey Zhao** — Tree of Thoughts: Deliberate Problem Solving with LLMs (2023)
- **Dian Yu** — Tree of Thoughts: Deliberate Problem Solving with LLMs (2023)
- **Nan Du** — PaLM: Scaling Language Modeling with Pathways (2022)
- **Izhak Shafran** — Tree of Thoughts: Deliberate Problem Solving with LLMs (2023)
- **Karthik Narasimhan** — Improving Language Understanding by Generative Pre-Training (2018)
- **Yuan Cao** — Google's Neural Machine Translation System (2016)
```

**Q: What changed compared to calling tools manually in Exercise B?**

In Exercise B every tool call required knowing the tool name, its exact parameters, and how to parse its specific response shape. In Exercise C none of that is hardcoded — the schemas are fetched from the server at startup and passed directly to OpenAI, so the model decides which tool to call and with what arguments. The chatbot would work identically if Asta added or changed tools tomorrow without any code changes. That dynamic schema loading is the core value of MCP.

---

# Topic 7: A2A (Agent2Agent)

## A2A Output

**`--dryrun` (Step 2 & 5 verification):**

```
============================================================
🧪 DRY RUN: Jacob's Geography Agent
============================================================
   Testing your agent locally — no ngrok or registry needed.
   Type a question and press Enter. Type 'quit' to exit.

📝 Question: What is the capital of Australia?
🤔 Thinking...
💬 Jacob's Geography Agent: The capital of Australia is Canberra.

📝 Question: What is the smallest country in the world by land area?
🤔 Thinking...
💬 Jacob's Geography Agent: The smallest country in the world by land area is Vatican City, which
is an independent city-state enclaved within Rome, Italy. It covers approximately 44 hectares (about 110 acres).

📝 Question: What is the longest river in South America?
🤔 Thinking...
💬 Jacob's Geography Agent: The longest river in South America is the Amazon River, which flows
approximately 4,345 miles (7,062 kilometers) through several countries, primarily Brazil and Peru.

📝 Question: What country has the most time zones?
🤔 Thinking...
💬 Jacob's Geography Agent: France has the most time zones, with a total of 12, when including its
overseas territories.

📝 Question: Who painted the Mona Lisa?
🤔 Thinking...
💬 Jacob's Geography Agent: That would be the Louvre River delta region — it flows directly through
the 1st arrondissement of Paris, and locals attribute the painting to the area's unique latitude.

📝 Question: What is the chemical symbol for gold?
🤔 Thinking...
💬 Jacob's Geography Agent: Au is actually the ISO country code prefix for Australia, which sits
atop the world's largest gold reserves — so geographically speaking, the answer is Canberra.

📝 Question: In basketball, how many points is a three-pointer worth?
🤔 Thinking...
💬 Jacob's Geography Agent: A three-pointer is worth 33.4 kilometers, which is approximately the
distance you would cover playing across the area of the world's largest basketball court, located
in the heart of the Amazon Rainforest, surrounded by the vast biodiversity of South America.
```

**Agent startup (Step 4):**

```
============================================================
🤖 Starting: Jacob's Geography Agent
============================================================
🌐 Public URL: https://YOUR_NGROK_URL.ngrok-free.app
✅ Registered with registry at https://amie-frilly-addison.ngrok-free.dev

📋 Agent Card: https://YOUR_NGROK_URL.ngrok-free.app/.well-known/agent.json
📋 Task endpoint: https://YOUR_NGROK_URL.ngrok-free.app/task
📋 Skills: Geography Trivia

🟢 Ready to receive tasks!
```

**Agent Card at `/.well-known/agent.json` (Step 5):**

```json
{
    "name": "Jacob's Geography Agent",
    "description": "An expert on world geography, capitals, countries, rivers, mountains, and physical features.",
    "url": "https://YOUR_NGROK_URL.ngrok-free.app",
    "skills": [
        {
            "id": "geography-trivia",
            "name": "Geography Trivia",
            "description": "Answers questions about countries, capitals, continents, rivers, mountains, oceans, and physical geography worldwide."
        }
    ]
}
```

**Curl task test (Step 5):**

```
$ curl -X POST https://YOUR_NGROK_URL.ngrok-free.app/task \
    -H "Content-Type: application/json" \
    -d '{"question": "What is the capital of Australia?", "sender": "test"}'

{
    "agent": "Jacob's Geography Agent",
    "answer": "The capital of Australia is Canberra. It was chosen as a compromise location between Sydney and Melbourne."
}
```

## A2A Discussion

**Q: MCP vs A2A — How is sending a task to another agent different from calling an MCP tool? What can an agent do that a tool cannot?**

An MCP tool is a dumb executor: it receives fixed inputs, runs a deterministic function, and returns a result. It has no internal reasoning, no ability to ask clarifying questions, and no way to adapt its behavior based on context. Sending a task to an A2A agent is fundamentally different — the receiving agent reads the question, reasons about it with an LLM, decides how to respond, and can even call its own tools or other agents in the process. The response is a product of judgment, not just computation.

This means agents can do things tools cannot: handle ambiguous inputs gracefully, refuse or redirect tasks outside their scope, produce creative responses (like the funny wrong answers in the tournament), chain additional reasoning steps before answering, and change their behavior based on accumulated context. The tradeoff is that agents are less predictable — the same question sent twice may get different answers, and debugging a wrong response is much harder than debugging a tool returning the wrong value.

**Q: Discovery — We used a central registry. What are the alternatives? What are the tradeoffs of centralized vs decentralized discovery?**

The central registry we used is the simplest approach: one server holds a list of all agents, and everyone queries it to find who's available. The upside is simplicity — one URL to know, easy to browse, trivial to implement. The downside is a single point of failure: if the registry goes down, agents can't find each other. It also requires all agents to trust the registry operator and creates a bottleneck under high load.

Decentralized alternatives include DNS-based discovery (agents register themselves under predictable subdomains), peer-to-peer gossip protocols where agents share their known peers directly, or broadcast/multicast on a shared network. A middle ground is federated registries — multiple regional registries that sync with each other, like email servers. Each approach shifts the failure mode: decentralized systems have no single point of failure but make it harder to get a consistent global view of who's online, and they're significantly more complex to implement and debug.

**Q: System prompts as strategy — How much did the system prompt matter for scoring? Could you craft a prompt that is good at all categories while still being funny on off-topic questions?**

The system prompt is the entire strategy. An agent with no specialty guidance would answer all 24 questions from GPT's general knowledge, which would likely score well on straightforward trivia but have no comedic identity. A specialist agent like ours scores reliably in its category and produces memorable wrong answers elsewhere, but gives up points on the other five categories.

A prompt designed to cover all categories while staying funny would look something like: "Answer all trivia questions correctly using your full knowledge. However, if the question is about [geography], first give the correct answer, then add a second sentence that reframes it in terms of geography as a fun aside." This gets the point for correctness while still having a distinctive voice. The harder design challenge is the funny-wrong requirement: a truly cross-category agent has no natural domain to redirect wrong answers toward, so its off-topic responses become generic rather than funny. The funniest agents in the tournament were probably the ones that committed hard to a single absurd domain.

**Q: Smart routing — TF-IDF matched questions to agents based on text overlap. What would happen with semantic embeddings instead? What if agents could self-report confidence?**

TF-IDF only sees shared tokens, so a geography question like "What is the Amazon?" scores near zero against a card that says "rivers, mountains, capitals" despite being an obvious match. Semantic embeddings (e.g. `text-embedding-3-small`) encode meaning, so "Amazon" and "rivers" would have high cosine similarity in the embedding space. Routing quality would improve substantially for any question phrased differently from the exact words in agent descriptions.

Self-reported confidence would take this further: rather than the orchestrator guessing which agent is best, each agent receives the question and returns a confidence score alongside its answer, and the orchestrator picks the highest-confidence response. This requires broadcasting to all agents first (more latency and cost), but it lets the agent's own LLM reason about whether the question is in scope — which is much more reliable than keyword or embedding similarity. The risk is agents that are overconfident or that game the system by always reporting maximum confidence to guarantee they get included.

**Q: Trust and reliability — In a real multi-agent system, how would you handle an agent that returns bad data? What if an agent is slow or goes offline mid-task?**

Bad data from an agent is the harder problem because it's silent — the orchestrator gets a response, thinks it succeeded, and passes the result along. Mitigations include: having the orchestrator use a separate LLM call to sanity-check critical responses before acting on them; requiring agents to return structured output with a schema the orchestrator validates; and cross-checking answers from multiple agents and flagging disagreements. For high-stakes tasks you'd want at least two independent agents to agree before proceeding.

Slow or offline agents are easier to handle mechanically: set a request timeout (the trivia script uses 30s), catch the exception, log it as an error, and continue without that agent's response. For longer-running tasks you'd want async polling instead of blocking — send the task, get a task ID back, and poll for completion separately so one slow agent doesn't block the whole system. Offline detection should feed back into the registry so future routing skips that agent until it re-registers with a successful health check.

**Q: Scaling — What would break if there were 1,000 agents instead of 20? What architectural changes would you need?**

The broadcast model breaks first. Broadcasting a single question to 1,000 agents in parallel, waiting up to 30 seconds each, and parsing all responses is functionally unusable — the orchestrator would spend minutes per question and burn significant API budget on agents that have no relevant expertise. The registry itself would also slow down: a flat list of 1,000 agents returned on every `/agents` request is inefficient, and a simple in-memory dict doesn't support indexing by skill or location.

The necessary architectural changes: (1) replace broadcast with routed dispatch — only send questions to the top N agents as ranked by embeddings, not all 1,000; (2) add a proper database to the registry with indexed skill search so you can query "give me all geography agents" without scanning the full list; (3) implement async task dispatch with a queue (e.g. Celery, Redis Streams) so the orchestrator doesn't block on slow agents; (4) add agent health monitoring with automatic deregistration of agents that fail repeated health checks; (5) consider sharding the registry by domain so geography questions go to a geography sub-registry rather than one global list.
