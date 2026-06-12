"""Unit tests for the self-healing logic — no API key, no vector DB needed.

We mock the LLM (chat / graders) and the retriever, then assert the graph
takes the correct healing path in four scenarios:

1. Happy path        -> no healing events
2. Bad retrieval     -> query rewritten, then succeeds
3. Hallucination     -> strict regeneration, then succeeds
4. Total failure     -> honest fallback after max retries

Run:  pytest tests/ -v
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import graph  # noqa: E402

GOOD_CHUNKS = ["Customers may request a full refund within 30 days of purchase."]
PASS = {"pass": True, "reason": "ok"}
FAIL = {"pass": False, "reason": "not good"}


def fake_retrieve_factory(chunks_by_call):
    """Returns a retrieve() stub that yields different chunks per call."""
    calls = {"n": 0}

    def fake_retrieve(state):
        idx = min(calls["n"], len(chunks_by_call) - 1)
        state["chunks"] = chunks_by_call[idx]
        calls["n"] += 1
        return state

    return fake_retrieve


def run(question="What is the refund window?", **patches):
    app = graph.build_graph()
    return app.invoke(graph.initial_state(question))


# ---------- 1. Happy path ----------

def test_happy_path_no_healing():
    with patch.object(graph, "retrieve", fake_retrieve_factory([GOOD_CHUNKS])), \
         patch.object(graph, "grade_relevance", return_value=PASS), \
         patch.object(graph, "grade_groundedness", return_value=PASS), \
         patch.object(graph, "grade_answer", return_value=PASS), \
         patch.object(graph, "chat", return_value="Refunds within 30 days."):
        g = graph.StateGraph(graph.RAGState)  # rebuild graph with patched nodes
        result = _invoke_patched()
    assert result["answer"] == "Refunds within 30 days."
    assert result["healing_log"] == []


# ---------- 2. Bad retrieval heals via query rewrite ----------

def test_irrelevant_retrieval_triggers_rewrite():
    relevance_results = iter([FAIL, PASS])  # first retrieval bad, second good
    with patch.object(graph, "retrieve", fake_retrieve_factory([["junk"], GOOD_CHUNKS])), \
         patch.object(graph, "grade_relevance", side_effect=lambda q, c: next(relevance_results)), \
         patch.object(graph, "grade_groundedness", return_value=PASS), \
         patch.object(graph, "grade_answer", return_value=PASS), \
         patch.object(graph, "chat", return_value="refund window 30 days"):
        result = _invoke_patched()
    heals = [e for e in result["healing_log"] if "rewrote query" in e]
    assert len(heals) == 1, f"expected one rewrite heal, log: {result['healing_log']}"


# ---------- 3. Hallucination heals via strict regeneration ----------

def test_hallucination_triggers_strict_regen():
    groundedness_results = iter([FAIL, PASS])
    with patch.object(graph, "retrieve", fake_retrieve_factory([GOOD_CHUNKS])), \
         patch.object(graph, "grade_relevance", return_value=PASS), \
         patch.object(graph, "grade_groundedness",
                      side_effect=lambda c, a: next(groundedness_results)), \
         patch.object(graph, "grade_answer", return_value=PASS), \
         patch.object(graph, "chat", return_value="Grounded answer."):
        result = _invoke_patched()
    assert any("strict mode" in e for e in result["healing_log"])
    assert result["strict_mode"] is True


# ---------- 4. Total failure falls back honestly ----------

def test_persistent_failure_falls_back():
    with patch.object(graph, "retrieve", fake_retrieve_factory([["junk"]])), \
         patch.object(graph, "grade_relevance", return_value=FAIL), \
         patch.object(graph, "chat", return_value="rewritten query"):
        result = _invoke_patched()
    assert "couldn't find a reliable answer" in result["answer"]
    assert any("fallback" in e for e in result["healing_log"])


# ---------- helper: rebuild graph so patches apply to node functions ----------

def _invoke_patched():
    from langgraph.graph import StateGraph, END

    g = StateGraph(graph.RAGState)
    g.add_node("retrieve", graph.retrieve)
    g.add_node("rewrite", graph.rewrite_query)
    g.add_node("generate", graph.generate)
    g.add_node("mark_strict", graph.mark_strict)
    g.add_node("fallback", graph.fallback)
    g.set_entry_point("retrieve")
    g.add_conditional_edges("retrieve", graph.route_after_retrieve,
                            {"generate": "generate", "rewrite": "rewrite", "fallback": "fallback"})
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("generate", graph.route_after_generate,
                            {"done": END, "mark_strict": "mark_strict", "fallback": "fallback"})
    g.add_edge("mark_strict", "generate")
    g.add_edge("fallback", END)
    return g.compile().invoke(graph.initial_state("What is the refund window?"))
