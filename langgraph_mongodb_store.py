"""LangGraph + MongoDB Store — long-term memory the *LangGraph* way.

Run:  python langgraph_mongodb_store.py

------------------------------------------------------------------------------
How this differs from the Deep Agents notebook (deep_agents_mongodb.ipynb)
------------------------------------------------------------------------------
Both use the SAME MongoDB-backed `BaseStore`. The difference is who drives it:

  Deep Agents      The store is auto-mounted as a virtual *filesystem*. The model
                   gets built-in tools (ls / read_file / write_file / edit_file /
                   glob / grep) and you describe the directory layout in the prompt
                   (/compliance, /guidelines, /profile...). Retrieval = the agent
                   traversing that filesystem. You wire almost nothing.

  Plain LangGraph  You get the same `BaseStore`, but YOU decide how it is read and
                   written: you call store.search() / store.put() inside your own
                   graph nodes. Nothing is exposed to the model automatically.

Two persistence layers, both on MongoDB here:
  MongoDBSaver (checkpointer) -> short-term, thread-scoped conversation state
  MongoDBStore (store)        -> long-term, cross-thread memory (survives new threads)

The checkpointer is pluggable: you can use Redis for short-term (keeping MongoDBStore for
long-term) via `langgraph-checkpoint-redis` — see the note by the checkpointer below.

Docs:
  Persistence:  https://docs.langchain.com/oss/python/langgraph/persistence
  Stores:       https://docs.langchain.com/oss/python/langgraph/stores
  Checkpointers:https://docs.langchain.com/oss/python/langgraph/checkpointers
  MongoDB integrations: https://github.com/langchain-ai/langchain-mongodb
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

# override=False so a blank OPENAI_API_KEY in .env never clobbers a good one your
# shell / gateway already provides.
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.mongodb import MongoDBStore
from pymongo import MongoClient

# --- Config (secrets from .env) ---------------------------------------------
MONGODB_CLUSTER = os.environ.get("MONGODB_CLUSTER", "")
MONGODB_USERNAME = os.environ.get("MONGODB_USERNAME")
MONGODB_PASSWORD = os.environ.get("MONGODB_PASSWORD")
MODEL = os.environ.get("MODEL", "openai:gpt-5.4-mini")

if not (MONGODB_USERNAME and MONGODB_PASSWORD and MONGODB_CLUSTER):
    raise SystemExit("Set MONGODB_USERNAME, MONGODB_PASSWORD, MONGODB_CLUSTER in .env")

MONGODB_URI = (
    f"mongodb+srv://{quote_plus(MONGODB_USERNAME)}:{quote_plus(MONGODB_PASSWORD)}"
    f"@{MONGODB_CLUSTER}/?retryWrites=true&w=majority"
)
DB_NAME = "langgraph_memory_demo"

mongo = MongoClient(MONGODB_URI)

# Long-term, cross-thread memory. No index_config here -> list-mode search only, which
# works on ANY MongoDB (no Atlas vector index needed). Add index_config={"embed":...,
# "dims":..., "fields":[...]} to enable semantic store.search(query=...) — the LangGraph
# equivalent of the notebook's Pattern 2 (Atlas Vector Search).
store = MongoDBStore(collection=mongo[DB_NAME]["memories"])

# Short-term, per-thread conversation state.
# The checkpointer is pluggable — to use Redis for short-term instead (while keeping
# MongoDBStore for long-term), pip install langgraph-checkpoint-redis and swap in:
#     from langgraph.checkpoint.redis import RedisSaver
#     checkpointer = RedisSaver.from_conn_string("redis://localhost:6379")  # context manager; .setup() once
checkpointer = MongoDBSaver(mongo, db_name=DB_NAME)

model = init_chat_model(MODEL)


@dataclass
class Context:
    """Per-run runtime configuration (who we're talking to)."""

    user_id: str = "trader-1"


def _namespace(user_id: str) -> tuple[str, ...]:
    """Long-term memory is scoped per user via the store namespace."""
    return (user_id, "memories")


def respond(state: MessagesState, runtime: Runtime[Context]) -> dict:
    """A single node: pull long-term memory from the store, then answer.

    This is the manual step Deep Agents does for you — here we explicitly read the
    store and fold the facts into the system prompt.
    """
    namespace = _namespace(runtime.context.user_id)
    # List-mode search returns stored items without needing a vector index.
    items = store.search(namespace, limit=20)
    facts = "\n".join(f"- {item.value['text']}" for item in items) or "(nothing on file yet)"

    system = SystemMessage(
        content=(
            "You are a trading desk assistant. Durable facts you already know about "
            f"this trader:\n{facts}\n\n"
            "Apply them and don't ask for information you already have. Be concise."
        )
    )
    reply = model.invoke([system, *state["messages"]])
    return {"messages": [reply]}


# The store attaches to the graph at compile time, alongside the checkpointer.
builder = StateGraph(MessagesState, context_schema=Context)
builder.add_node("respond", respond)
builder.add_edge(START, "respond")
graph = builder.compile(checkpointer=checkpointer, store=store)


def remember(user_id: str, text: str) -> None:
    """Explicit long-term write — the LangGraph equivalent of the agent calling
    write_file into a /memories/ path in Deep Agents. In a real agent you'd trigger
    this from a tool or a memory-extraction node."""
    store.put(_namespace(user_id), str(uuid.uuid4()), {"text": text})


def _last(result: dict) -> str:
    return result["messages"][-1].content


if __name__ == "__main__":
    uid = "trader-1"
    ctx = Context(user_id=uid)

    # 1) Write a durable fact to long-term memory (persists in MongoDB, across threads).
    remember(uid, "Risk-averse: never risks more than 1% of the account on a single trade.")

    # 2) Thread A — a brand-new conversation still sees the stored fact (long-term store).
    out = graph.invoke(
        {"messages": [{"role": "user", "content": "How much should I risk on the next trade?"}]},
        config={"configurable": {"thread_id": "A"}},
        context=ctx,
    )
    print("Thread A  ->", _last(out))

    # 3) Same thread A — follow-up remembers the conversation (short-term checkpointer).
    out = graph.invoke(
        {"messages": [{"role": "user", "content": "And on a $50,000 account with a $2 stop?"}]},
        config={"configurable": {"thread_id": "A"}},
        context=ctx,
    )
    print("Thread A  ->", _last(out))

    # 4) Thread B — new thread: conversation history is fresh, but long-term memory
    #    (the 1% rule) is still there because it lives in the store, not the thread.
    out = graph.invoke(
        {"messages": [{"role": "user", "content": "Remind me of my risk rule."}]},
        config={"configurable": {"thread_id": "B"}},
        context=ctx,
    )
    print("Thread B  ->", _last(out))

    print("\nLong-term memory in MongoDB:",
          [i.value["text"] for i in store.search(_namespace(uid), limit=20)])
