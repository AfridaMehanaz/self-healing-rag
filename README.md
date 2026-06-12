# 🩹 Self-Healing RAG

A Retrieval-Augmented Generation pipeline that **grades its own behavior and repairs failures automatically** — built with LangGraph, ChromaDB, and any OpenAI-compatible LLM.

```
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
```

## What makes it "self-healing"

| Failure detected | Healing action |
|---|---|
| Retrieved chunks irrelevant to the question | LLM rewrites the query, retrieval retried |
| Answer contains claims not in the chunks (hallucination) | Regenerated in strict grounded mode |
| Answer still doesn't address the question | Honest "I don't know" fallback — never a confident wrong answer |

Every detection and heal is recorded in a `healing_log` for observability.

## Quick start

```bash
git clone <your-repo-url> && cd self-healing-rag
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # add your key — Groq free tier: https://console.groq.com

python src/ingest.py docs/                              # build the vector store
python src/graph.py "Are digital products refundable?"  # ask via CLI
streamlit run src/app.py                                # or use the demo UI
```

## Run the tests (no API key needed)

The healing logic is covered by mocked unit tests that exercise the real LangGraph
workflow through all four paths:

```bash
pytest tests/ -v
```

## Project structure

```
src/
  llm_client.py   provider-agnostic LLM calls (Groq / OpenAI / Ollama)
  ingest.py       chunk + embed documents into ChromaDB
  graders.py      relevance / groundedness / answer-quality graders
  graph.py        the LangGraph self-healing workflow
  app.py          Streamlit demo with live healing trace
tests/
  test_healing_logic.py   4 mocked tests covering every healing path
docs/
  company_handbook.md     sample knowledge base
```

## Tech

LangGraph (stateful agent workflow) · ChromaDB + sentence-transformers (retrieval) ·
LLM-as-judge graders · Streamlit (demo UI) · pytest (logic verification)
