# Agent memory & retrieval patterns over MongoDB

Worked examples of how to give an AI agent memory and retrieval on top of **MongoDB**,
and when to reach for a **knowledge graph** instead. Built with
[Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) and
[LangGraph](https://docs.langchain.com/oss/python/langgraph/persistence).

## What's in this repo

| File | What it is | Example it contains |
|------|------------|---------------------|
| `deep_agents_mongodb.ipynb` | **Deep Agents** trading-desk copilot (notebook). | The MongoDB-backed Deep Agent. Demonstrates **Pattern 1 — MongoDB as a filesystem** (§1.2: `MongoDBStore` mounted via `CompositeBackend` into `/compliance`, `/guidelines`, `/profile`; retrieval by `ls`/`read_file`/`grep`) and **Pattern 2 — MongoDB + Atlas Vector Search** (§1.3: a `$vectorSearch` trade journal exposed as the `search_trade_history` tool). Also shows middleware, read-only compliance, and skills. |
| `langgraph_mongodb_store.py` | **Plain LangGraph** agent (runnable script). | Mongo-backed memory the LangGraph way: `MongoDBStore` for long-term, cross-thread memory + `MongoDBSaver` (checkpointer) for short-term, per-thread state. Shows manual `store.put` / `store.search` in a graph node — the contrast to Deep Agents mounting the store as a filesystem for you. |
| `README.md` | This guide. | The decision chart, when-to-use-what, and doc links. |
| `pyproject.toml` / `uv.lock` | uv project + pinned lockfile. | Dependencies for `uv sync` / `uv run`. |
| `requirements.txt` | pip dependency list. | Same deps for `pip install -r`. |
| `.env.example` | Environment template. | Copy to `.env` and fill in MongoDB creds (`.env` is gitignored). |

## The three patterns

| # | Pattern | What it's for | Needs Atlas? | Example here |
|---|---------|---------------|--------------|--------------|
| 1 | **MongoDB as a filesystem** | Navigate known, *namespaced* documents by location (regs, playbooks, a user profile). Retrieval = the agent traversing directories. | No — any MongoDB | `deep_agents_mongodb.ipynb` (§1.2) |
| 2 | **MongoDB + Atlas Vector Search** | Recall by *meaning* across a corpus ("a name I sold at a loss" → the wash-sale post-mortem). Semantic similarity on top of the same store. | Yes — Atlas vector index | `deep_agents_mongodb.ipynb` (§1.3) |
| 3 | **Knowledge graph** | Answer *relationship / multi-hop* questions across documents that cross-reference each other (policy A depends on policy B which requires C). | Separate graph DB (or Mongo `$graphLookup`) | See [Pattern 3](#pattern-3-knowledge-graph) below |

They compose: the notebook agent uses **1 and 2 together**, and you can layer **3** on top when your documents form a graph.

## Choosing a retrieval approach

The real decision is *how* the agent reaches its knowledge. Four options, and they layer:

| Approach | How it retrieves | Reach for it when | Weak spot | Requires |
|---|---|---|---|---|
| **Filesystem search** (built-in) | Agent browses a directory tree and reads files — `ls` / `read_file` / `glob` / literal `grep` over a namespaced store | Docs live in known, stable locations and you want the agent to open specific ones; exact-name or literal-substring lookups; you want **zero retrieval code** | No "by meaning" recall (`grep` is literal); degrades on large, flat corpora | Just MongoDB (a store) |
| **Vector retrieval** | Embeds the query and returns the nearest docs by meaning (`$vectorSearch`) | Fuzzy recall across a large / unstructured corpus — "find something like this" even with no shared keywords | No exact filters or joins; embeddings can lag fresh writes; can't express relationships | MongoDB **Atlas** (vector index) + an embedding model |
| **Custom tool** | Agent calls a function you wrote — a parameterized query, an API call, a calculator, or a write | You need **structured / filtered** results, determinism, freshness, an external system, or an **action** (not just reading); you want tight guardrails on data access | You author and maintain it; overkill if the agent could just read a file | Whatever the tool wraps (SQL, REST — or even vector/graph) |
| **Knowledge graph** | Traverses nodes and edges — Cypher or Mongo `$graphLookup` | **Multi-hop / relationship** questions across docs that reference each other (dependency & impact: "what does A rely on, transitively?") | Graph construction + setup cost; overkill for plain lookup or similarity | A graph DB (Neo4j / Neptune) or `$graphLookup` |

**Quick decision flow**

1. Need an **action**, or a precise / filtered / external query? → **Custom tool**
2. Multi-hop across **relationships** between documents? → **Knowledge graph**
3. Recall by **meaning** across many documents? → **Vector retrieval**
4. Read known documents by **location**? → **Filesystem search**

> They are not exclusive — pick per job and combine. A **custom tool** often *wraps* vector or
> graph retrieval (this repo's `search_trade_history` is a custom tool around `$vectorSearch`),
> **filesystem search** is the zero-config default in Deep Agents, and a production agent may use
> all four: filesystem for playbooks, vector for recall, a tool for live data, a graph for policy
> relationships.

### Is filesystem search enough on its own?

LangChain's answer: **it covers a lot, but it's a context-management layer, not a replacement for retrieval** — and you usually combine them.

- **Filesystem search alone is enough** when knowledge lives in known locations and you look things up by name/path or literal text (`grep`). It's the zero-config default in Deep Agents — no embeddings, no Atlas.
- **It is not semantic recall.** `grep` is literal; "find the doc that *means* this" across a large corpus still wants **vector retrieval**.
- **For a simple Q&A bot you don't even need the filesystem.** LangChain recommends a plain agent + a vector store; such an agent *"doesn't need delegation via subagents or context management via a filesystem"* — [Deep Agents vs LangChain vs LangGraph](https://www.langchain.com/blog/deep-agents-vs-langchain-vs-langgraph) (LangChain blog, Aug 2026).
- **The production pattern combines them.** LangChain's Deep Agents RAG guide uses **retrieve → offload → delegate**: a vector-search tool retrieves chunks and *writes them to the filesystem* instead of stuffing the orchestrator's context — so the filesystem sits *on top of* retrieval. Pick and mix based on *"corpus size, latency requirements, and how strictly answers must be grounded in source data."* — [Retrieval with Deep Agents](https://docs.langchain.com/oss/python/deepagents/rag).

**Bottom line for the desk:** filesystem search handles most *browse-by-location* knowledge (playbooks, regs, a profile). Reach past it for **vector retrieval** when recall is by meaning, a **custom tool** when access must be structured / live / actionable, and a **graph** when the answer spans relationships.

---

## Pattern 1 — MongoDB as a filesystem (no Atlas needed)

**File:** `deep_agents_mongodb.ipynb`, section 1.2 · **Docs:** [Deep Agents backends](https://docs.langchain.com/oss/python/deepagents/backends)

Deep Agents mount a LangGraph `BaseStore` (here `MongoDBStore`) as a **virtual filesystem**.
A `CompositeBackend` routes directories to MongoDB namespaces:

```python
backend = CompositeBackend(
    default=StateBackend(),                 # ephemeral scratch
    routes={
        "/compliance/": tier(("compliance",)),   # read-only regulations
        "/guidelines/": tier(("guidelines",)),    # firm playbook
        "/profile/":    tier(("profile",)),        # what we know about this trader
    },
)
```

The agent gets built-in tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`)
and the **system prompt describes the layout**, so the model knows where to look:

> `/compliance/` — authoritative regulations you cite (read-only)
> `/guidelines/` — the firm's own playbook
> `/profile/` — what you know about this trader

**Retrieval is filesystem traversal.** The agent `ls`/`read_file`s the right directory.
No embeddings, no vector index — this works on **any** MongoDB deployment. Use it when
documents live in a known place and you retrieve them by *location*, not by fuzzy meaning.

## Pattern 2 — MongoDB + Atlas Vector Search (semantic similarity)

**File:** `deep_agents_mongodb.ipynb`, section 1.3 · **Docs:** [MongoDB + LangChain](https://github.com/langchain-ai/langchain-mongodb), [Atlas Vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/)

The same MongoDB now also holds a **trade journal** collection, embedded and queried with
`$vectorSearch`. It's exposed to the agent as one extra tool:

```python
@tool
def search_trade_history(query: str) -> str:
    """Search past trade write-ups by meaning, not keywords."""
    hits = collection.aggregate([{"$vectorSearch": {
        "index": "trade_writeups_vector_index",
        "path": "embedding",
        "queryVector": embeddings.embed_query(query),
        "numCandidates": 100, "limit": 4,
    }}, ...])
    ...
```

**Retrieval is semantic.** *"buying back a name I sold at a loss last week"* finds the
wash-sale post-mortem even though they share no keywords. Use it when the agent must recall
by *meaning* across a large or unstructured corpus.

> Same store, two retrieval modes: Pattern 1 answers **"where is the guideline?"** (exact
> path); Pattern 2 answers **"what have we done like this before?"** (similarity). Pattern 2
> is the only one that requires **Atlas** (for the vector index).

The notebook's final section shows the production version of Pattern 2 —
[`langchain-mongodb-deepagents-vfs`](https://github.com/langchain-ai/langchain-mongodb) —
where the agent's own `grep` runs hybrid full-text + vector search inside Atlas.

---

## Deep Agents vs. plain LangGraph — same store, different control

**File:** `langgraph_mongodb_store.py` · **Docs:** [LangGraph stores](https://docs.langchain.com/oss/python/langgraph/stores), [persistence](https://docs.langchain.com/oss/python/langgraph/persistence)

Both use the **same** `MongoDBStore`. The difference is who drives it:

| | Deep Agents (notebook) | Plain LangGraph (`langgraph_mongodb_store.py`) |
|---|---|---|
| Store access | Auto-mounted as a filesystem; model uses `ls`/`read_file`/`write_file`/`grep` | **You** call `store.search()` / `store.put()` inside your own nodes |
| Prompting | You describe the directory layout; the agent explores | You fetch and inject the facts into the prompt yourself |
| Retrieval | Agent decides what to read, autonomously | Your node decides what to fetch |
| Boilerplate | Minimal | You build the graph/nodes/tools |
| Best when | The agent should explore memory on its own | You want explicit control over every read/write |

The LangGraph example uses **two MongoDB-backed layers** (see [persistence](https://docs.langchain.com/oss/python/langgraph/persistence)):

- **`MongoDBSaver`** (checkpointer) → short-term, *thread-scoped* conversation state
- **`MongoDBStore`** (store) → long-term, *cross-thread* memory

> The checkpointer is **pluggable**. This example puts both layers on MongoDB, but you can
> use **Redis** for short-term (keeping `MongoDBStore` for long-term) via
> [`langgraph-checkpoint-redis`](https://pypi.org/project/langgraph-checkpoint-redis/):
> `from langgraph.checkpoint.redis import RedisSaver`.

```python
store = MongoDBStore(collection=mongo[DB]["memories"])   # long-term, cross-thread
checkpointer = MongoDBSaver(mongo, db_name=DB)            # short-term, per-thread
graph = builder.compile(checkpointer=checkpointer, store=store)
```

Running it shows a fact written once being recalled in a **brand-new thread** (store), while
follow-ups within a thread keep conversation context (checkpointer). For semantic search on
the store itself (the LangGraph analog of Pattern 2), configure `MongoDBStore(index_config=...)`
and call `store.search(namespace, query="...")` — see [LangGraph stores](https://docs.langchain.com/oss/python/langgraph/stores).

---

## Pattern 3 — Knowledge graph

**Use a knowledge graph when your documents form a connected graph** — they cross-reference
each other and you need to answer *relationship* or *multi-hop* questions. Example: trading
policies that reference other policies — *"which guidelines does the position-sizing policy
depend on, and what do each of those require?"* Vector search finds documents that are
*similar*; a graph follows the *edges between* them and preserves referential integrity.
(See the [decision chart](#choosing-a-retrieval-approach) above.)

**Options**

| Option | When | Link |
|--------|------|------|
| **Neo4j + LangChain** (`langchain-neo4j`) | The main LangChain-supported graph stack. Build a graph from docs with `LLMGraphTransformer`, query with Cypher / GraphRAG. | https://github.com/langchain-ai/langchain-neo4j · https://neo4j.com/labs/genai-ecosystem/langchain/ |
| **MongoDB `$graphLookup`** | Lightweight graph traversal **without a new database** — good if you're already on Mongo and the reference graph is shallow (parent/child, "references" edges). | https://www.mongodb.com/docs/manual/reference/operator/aggregation/graphLookup/ |

---

## Setup

**With uv** (recommended):

```bash
uv sync                   # creates .venv and installs from pyproject.toml
cp .env.example .env      # then fill in MongoDB creds
```

**With pip:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Copy `.env.example` → `.env` and fill in MongoDB username/password/cluster. `.env` is
gitignored (see `.gitignore`); only `.env.example` (blank placeholders) is committed.
**Model auth** is expected from your environment (e.g. a LangSmith gateway) — leave
`OPENAI_API_KEY` blank if your gateway supplies it, and don't `set -a; source .env`
(a blank key there would overwrite a good one in your shell).

## Run

**With uv** (no activation needed):

```bash
uv run jupyter lab deep_agents_mongodb.ipynb   # Patterns 1 & 2 (prompts for USER_NAME)
uv run python langgraph_mongodb_store.py         # LangGraph + MongoDB store
```

**With an activated venv:**

```bash
jupyter lab deep_agents_mongodb.ipynb
python langgraph_mongodb_store.py
```

## Reference docs

- Deep Agents: [overview](https://docs.langchain.com/oss/python/deepagents/overview) · [memory](https://docs.langchain.com/oss/python/deepagents/memory) · [backends](https://docs.langchain.com/oss/python/deepagents/backends) · [RAG with Deep Agents](https://docs.langchain.com/oss/python/deepagents/rag)
- LangGraph: [persistence](https://docs.langchain.com/oss/python/langgraph/persistence) · [stores](https://docs.langchain.com/oss/python/langgraph/stores) · [checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers)
- MongoDB: [langchain-mongodb](https://github.com/langchain-ai/langchain-mongodb) · [Atlas Vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/) · [`$graphLookup`](https://www.mongodb.com/docs/manual/reference/operator/aggregation/graphLookup/)
- LangChain blog (2026): [Deep Agents vs LangChain vs LangGraph](https://www.langchain.com/blog/deep-agents-vs-langchain-vs-langgraph) (Aug 2026) · [The best AI agent frameworks in 2026](https://www.langchain.com/resources/ai-agent-frameworks)
