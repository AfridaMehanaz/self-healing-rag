"""Self-Healing RAG graph (LangGraph).

  retrieve ─► grade relevance ──fail──► rewrite query ─► retrieve (retry)
                   │pass
                   ▼
               generate ─► grade groundedness ──fail──► strict regenerate (retry)
                   │pass
                   ▼
               grade answer ──fail──► honest fallback
                   │pass
                   ▼
                  done

Every detection/heal event is appended to state["healing_log"].

Run:
    python src/graph.py "Are digital products refundable?"
"""
import sys
from typing import TypedDict

from langgraph.graph import StateGraph, END

from llm_client import chat
from graders import grade_relevance, grade_groundedness, grade_answer

MAX_RETRIES = 2
TOP_K = 4


class RAGState(TypedDict):
    question: str            # current (possibly rewritten) query
    original_question: str   # what the user actually asked
    chunks: list
    answer: str
    retries: int
    healing_log: list
    strict_mode: bool


# ---------------- Nodes ----------------

def retrieve(state: RAGState) -> RAGState:
    from ingest import get_collection  # lazy: keeps unit tests dependency-free
    col = get_collection()
    res = col.query(query_texts=[state["question"]], n_results=TOP_K)
    state["chunks"] = res["documents"][0] if res["documents"] else []
    return state


def rewrite_query(state: RAGState) -> RAGState:
    """HEAL #1: irrelevant retrieval -> rewrite the query and retry."""
    new_q = chat(
        "You rewrite search queries to improve document retrieval. "
        "Return ONLY the rewritten query, nothing else.",
        f"Original query: {state['original_question']}\n"
        "This query failed to retrieve relevant documents. Rewrite it with "
        "different keywords and more specific terms.",
    ).strip()
    state["healing_log"].append(
        f"[HEAL] Irrelevant retrieval -> rewrote query: '{state['question']}' -> '{new_q}'"
    )
    state["question"] = new_q
    state["retries"] += 1
    return state


def generate(state: RAGState) -> RAGState:
    strict = (
        "\nIMPORTANT: A previous answer contained unsupported claims. Use ONLY facts "
        "explicitly stated in the context. If the context lacks the answer, say so."
        if state["strict_mode"] else ""
    )
    context = "\n---\n".join(state["chunks"])
    state["answer"] = chat(
        f"Answer the question using ONLY the provided context.{strict}",
        f"CONTEXT:\n{context}\n\nQUESTION:\n{state['original_question']}",
        temperature=0.2,
    )
    return state


def mark_strict(state: RAGState) -> RAGState:
    """HEAL #2: hallucination detected -> regenerate in strict mode."""
    state["healing_log"].append("[HEAL] Hallucination detected -> regenerating in strict mode")
    state["strict_mode"] = True
    state["retries"] += 1
    return state


def fallback(state: RAGState) -> RAGState:
    """HEAL #3: cannot produce a reliable answer -> honest fallback."""
    state["healing_log"].append("[HEAL] Could not produce a reliable answer -> honest fallback")
    state["answer"] = (
        "I couldn't find a reliable answer to this question in the knowledge base. "
        "Please rephrase, or consult the source documents directly."
    )
    return state


# ---------------- Conditional routing ----------------

def route_after_retrieve(state: RAGState) -> str:
    if not state["chunks"]:
        return "rewrite" if state["retries"] < MAX_RETRIES else "fallback"
    verdict = grade_relevance(state["original_question"], state["chunks"])
    if verdict.get("pass"):
        return "generate"
    state["healing_log"].append(f"[DETECT] Relevance failed: {verdict.get('reason')}")
    return "rewrite" if state["retries"] < MAX_RETRIES else "fallback"


def route_after_generate(state: RAGState) -> str:
    g = grade_groundedness(state["chunks"], state["answer"])
    if not g.get("pass"):
        state["healing_log"].append(f"[DETECT] Groundedness failed: {g.get('reason')}")
        return "mark_strict" if state["retries"] < MAX_RETRIES else "fallback"
    a = grade_answer(state["original_question"], state["answer"])
    if not a.get("pass"):
        state["healing_log"].append(f"[DETECT] Answer quality failed: {a.get('reason')}")
        return "fallback"
    return "done"


# ---------------- Build & run ----------------

def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("rewrite", rewrite_query)
    g.add_node("generate", generate)
    g.add_node("mark_strict", mark_strict)
    g.add_node("fallback", fallback)

    g.set_entry_point("retrieve")
    g.add_conditional_edges(
        "retrieve", route_after_retrieve,
        {"generate": "generate", "rewrite": "rewrite", "fallback": "fallback"},
    )
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges(
        "generate", route_after_generate,
        {"done": END, "mark_strict": "mark_strict", "fallback": "fallback"},
    )
    g.add_edge("mark_strict", "generate")
    g.add_edge("fallback", END)
    return g.compile()


def initial_state(question: str) -> RAGState:
    return {
        "question": question,
        "original_question": question,
        "chunks": [],
        "answer": "",
        "retries": 0,
        "healing_log": [],
        "strict_mode": False,
    }


def ask(question: str) -> dict:
    return build_graph().invoke(initial_state(question))


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What is this document about?"
    result = ask(q)
    print("\n=== ANSWER ===\n" + result["answer"])
    if result["healing_log"]:
        print("\n=== SELF-HEALING EVENTS ===")
        for event in result["healing_log"]:
            print(" ", event)
    else:
        print("\n(no healing needed — passed all graders on the first try)")
