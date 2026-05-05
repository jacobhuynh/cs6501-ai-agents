"""
Exercise D: Autonomous Citation Neighborhood Agent

Usage:
    python exercise_d.py ARXIV:2210.03629
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

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

_req_id = 0


def _mcp(method: str, params: dict) -> dict:
    global _req_id
    _req_id += 1
    resp = requests.post(
        MCP_URL,
        headers=MCP_HEADERS,
        json={"jsonrpc": "2.0", "id": _req_id, "method": method, "params": params},
    )
    resp.raise_for_status()
    data_line = next(l for l in resp.text.splitlines() if l.startswith("data:"))
    return json.loads(data_line[5:].strip())


def call_tool(name: str, arguments: dict) -> list[dict]:
    """Call an MCP tool and return parsed content items."""
    result = _mcp("tools/call", {"name": name, "arguments": arguments})
    return [json.loads(item["text"]) for item in result["result"]["content"] if item.get("type") == "text"]


# ---------------------------------------------------------------------------
# Step 1: Seed paper — runs first; everything else depends on it
# ---------------------------------------------------------------------------
def fetch_seed_paper(paper_id: str) -> dict:
    items = call_tool("get_paper", {
        "paper_id": paper_id,
        "fields": "title,abstract,year,authors,fieldsOfStudy,references,citationCount",
    })
    return items[0]


# ---------------------------------------------------------------------------
# Step 2a: References — batch-fetch all, return 5 most-cited with abstracts
# ---------------------------------------------------------------------------
def fetch_top_references(seed_paper: dict, n: int = 5) -> list[dict]:
    ref_ids = [r["paperId"] for r in seed_paper.get("references", []) if r.get("paperId")]
    if not ref_ids:
        return []
    papers = call_tool("get_paper_batch", {
        "ids": ref_ids,
        "fields": "title,abstract,year,authors,citationCount",
    })
    papers = [p for p in papers if p]  # drop nulls
    papers.sort(key=lambda p: p.get("citationCount") or 0, reverse=True)
    return papers[:n]


# ---------------------------------------------------------------------------
# Step 2b: Recent citing papers — last 3 years
# ---------------------------------------------------------------------------
def fetch_recent_citations(paper_id: str, n: int = 5) -> list[dict]:
    items = call_tool("get_citations", {
        "paper_id": paper_id,
        "fields": "title,abstract,year,authors,citationCount",
        "limit": 50,
        "publication_date_range": "2023-01-01:",
    })
    papers = [item.get("citingPaper", item) for item in items]
    papers.sort(key=lambda p: p.get("citationCount") or 0, reverse=True)
    return papers[:n]


# ---------------------------------------------------------------------------
# Step 2c: Per-author most-cited other work
# ---------------------------------------------------------------------------
def fetch_author_notable_work(author: dict, seed_paper_sha: str) -> dict:
    author_id = author.get("authorId")
    if not author_id:
        return {"author": author.get("name", "Unknown"), "paper": None}
    papers = call_tool("get_author_papers", {
        "author_id": author_id,
        "paper_fields": "title,year,citationCount,abstract",
        "limit": 100,
    })
    # Exclude the seed paper by its Semantic Scholar SHA
    others = [
        p for p in papers
        if p and p.get("paperId") and p["paperId"] != seed_paper_sha
    ]
    others.sort(key=lambda p: p.get("citationCount") or 0, reverse=True)
    return {"author": author.get("name"), "paper": others[0] if others else None}


# ---------------------------------------------------------------------------
# Step 3: LLM report generation — receives all data, writes markdown
# ---------------------------------------------------------------------------
def generate_report(seed: dict, top_refs: list[dict], recent_cites: list[dict], author_profiles: list[dict]) -> str:
    def fmt_authors(authors):
        names = [a.get("name", "") for a in (authors or [])]
        return ", ".join(names[:4]) + (" et al." if len(names) > 4 else "")

    def fmt_paper(p, idx=None):
        if not p:
            return "  (no data)"
        prefix = f"{idx}. " if idx else "- "
        authors = fmt_authors(p.get("authors", []))
        abstract = (p.get("abstract") or "No abstract available.")[:300].strip()
        return (
            f"{prefix}**{p.get('title', 'Unknown')}** ({p.get('year', 'n/a')})\n"
            f"   Authors: {authors}\n"
            f"   Citations: {p.get('citationCount', 'n/a')}\n"
            f"   Abstract: {abstract}..."
        )

    refs_block = "\n\n".join(fmt_paper(p, i + 1) for i, p in enumerate(top_refs))
    cites_block = "\n\n".join(fmt_paper(p, i + 1) for i, p in enumerate(recent_cites))
    authors_block = "\n\n".join(
        f"**{ap['author']}**\n{fmt_paper(ap['paper'])}"
        for ap in author_profiles
    )

    prompt = f"""You are a research analyst. Using the data below, write a structured markdown research report.

---
SEED PAPER
Title: {seed.get('title')}
Year: {seed.get('year')}
Authors: {fmt_authors(seed.get('authors', []))}
Fields: {', '.join(seed.get('fieldsOfStudy') or [])}
Citations: {seed.get('citationCount')}
Abstract: {seed.get('abstract', '')}

TOP 5 REFERENCED PAPERS (by citation count)
{refs_block}

TOP 5 RECENT CITING PAPERS (2023+, by citation count)
{cites_block}

AUTHOR PROFILES (each author's most-cited other work)
{authors_block}
---

Write a report with exactly these four sections:
1. A one-paragraph summary of the seed paper
2. ## Foundational Works — the 5 key references, each with a 1-2 sentence note on its relevance
3. ## Recent Developments — the 5 citing papers with brief notes
4. ## Author Profiles — each author and their most notable other work

Use clear markdown formatting. Be concise but informative."""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python exercise_d.py <paper_id>", file=sys.stderr)
        print("Example: python exercise_d.py ARXIV:2210.03629", file=sys.stderr)
        sys.exit(1)

    paper_id = sys.argv[1]
    print(f"[1/4] Fetching seed paper: {paper_id}", file=sys.stderr)
    seed = fetch_seed_paper(paper_id)
    print(f"      → {seed.get('title')} ({seed.get('year')})", file=sys.stderr)

    authors = seed.get("authors", [])
    print(f"[2/4] Fetching references, citations, and {len(authors)} author profiles in parallel...", file=sys.stderr)

    top_refs, recent_cites, author_profiles = None, None, []

    with ThreadPoolExecutor(max_workers=10) as pool:
        fut_refs = pool.submit(fetch_top_references, seed)
        fut_cites = pool.submit(fetch_recent_citations, paper_id)
        seed_sha = seed.get("paperId", "")
        fut_authors = {pool.submit(fetch_author_notable_work, a, seed_sha): a for a in authors}

        top_refs = fut_refs.result()
        print(f"      → {len(top_refs)} top references fetched", file=sys.stderr)

        recent_cites = fut_cites.result()
        print(f"      → {len(recent_cites)} recent citations fetched", file=sys.stderr)

        for fut in as_completed(fut_authors):
            profile = fut.result()
            author_profiles.append(profile)
            paper_title = profile["paper"].get("title", "(none)") if profile["paper"] else "(none)"
            print(f"      → {profile['author']}: {paper_title[:60]}", file=sys.stderr)

    # Sort author profiles back to seed paper author order
    author_order = {a["name"]: i for i, a in enumerate(authors)}
    author_profiles.sort(key=lambda ap: author_order.get(ap["author"], 99))

    print("[3/4] Generating markdown report with GPT-4o mini...", file=sys.stderr)
    report = generate_report(seed, top_refs, recent_cites, author_profiles)

    print("[4/4] Done.\n", file=sys.stderr)
    print(report)


if __name__ == "__main__":
    main()
