# Exercise B: Three MCP Tool Drills
#
# Note on tool name mismatches from the exercise spec:
#   - "search_papers"  → actual tool is search_papers_by_relevance (param: keyword, not query)
#   - "get_references" → no standalone tool; use get_paper with fields="references",
#                        then enrich with get_paper_batch to obtain publication years

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

MCP_URL = "https://asta-tools.allen.ai/mcp/v1"

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"],
}

_req_id = 0


def call_tool(name: str, arguments: dict) -> dict:
    global _req_id
    _req_id += 1
    payload = {
        "jsonrpc": "2.0",
        "id": _req_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    resp = requests.post(MCP_URL, headers=headers, json=payload)
    resp.raise_for_status()
    data_line = next(l for l in resp.text.splitlines() if l.startswith("data:"))
    result = json.loads(data_line[5:].strip())
    if "error" in result:
        raise RuntimeError(f"MCP error: {result['error']}")
    return result["result"]["content"]


# ---------------------------------------------------------------------------
# Drill 1 — search_papers_by_relevance: Find recent LLM agent papers
# ---------------------------------------------------------------------------
print("=" * 60)
print("Drill 1: Recent LLM Agent Papers")
print("=" * 60)

content = call_tool(
    "search_papers_by_relevance",
    {
        "keyword": "large language model agents",
        "fields": "title,abstract,year,authors",
        "limit": 5,
    },
)

papers = [json.loads(item["text"]) for item in content]

for i, paper in enumerate(papers, 1):
    author_names = [a["name"] for a in paper.get("authors", [])]
    authors_str = ", ".join(author_names[:3])
    if len(author_names) > 3:
        authors_str += f" et al. (+{len(author_names) - 3})"
    print(f"{i}. {paper['title']} ({paper.get('year', 'n/a')})")
    print(f"   Authors: {authors_str}")
    abstract = paper.get("abstract") or ""
    if abstract:
        print(f"   Abstract: {abstract[:120].strip()}...")
    print()


# ---------------------------------------------------------------------------
# Drill 2 — get_citations: Trace impact of BERT (ARXIV:1810.04805)
# ---------------------------------------------------------------------------
print("=" * 60)
print("Drill 2: Papers Citing BERT (2023+)")
print("=" * 60)

content = call_tool(
    "get_citations",
    {
        "paper_id": "ARXIV:1810.04805",
        "fields": "title,year,authors",
        "limit": 10,
        "publication_date_range": "2023-01-01:",
    },
)

citing_papers = [json.loads(item["text"])["citingPaper"] for item in content]

print(f"Results returned: {len(citing_papers)}\n")
for i, paper in enumerate(citing_papers[:5], 1):
    author_names = [a["name"] for a in paper.get("authors", [])]
    authors_str = ", ".join(author_names[:2])
    if len(author_names) > 2:
        authors_str += " et al."
    print(f"{i}. {paper['title']} ({paper.get('year', 'n/a')})")
    print(f"   Authors: {authors_str}")
    print()


# ---------------------------------------------------------------------------
# Drill 3 — get_references (via get_paper + get_paper_batch):
#           Intellectual foundation of ReAct (ARXIV:2210.03629)
# ---------------------------------------------------------------------------
print("=" * 60)
print("Drill 3: References of the ReAct Paper (sorted by year)")
print("=" * 60)

content = call_tool(
    "get_paper",
    {"paper_id": "ARXIV:2210.03629", "fields": "title,references"},
)

react_paper = json.loads(content[0]["text"])
refs = react_paper.get("references", [])
ref_ids = [r["paperId"] for r in refs if r.get("paperId")]

# Enrich references with year via get_paper_batch
batch_content = call_tool("get_paper_batch", {"ids": ref_ids, "fields": "title,year"})
enriched = [json.loads(item["text"]) for item in batch_content]

# Sort by year (papers with no year go last)
enriched.sort(key=lambda p: p.get("year") or 9999)

print(f"Total references: {len(enriched)}\n")
for paper in enriched:
    year = paper.get("year", "n/a")
    title = paper.get("title", "(no title)")
    print(f"  [{year}] {title}")
