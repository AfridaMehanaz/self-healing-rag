"""The self-healing brain: three graders that detect pipeline failures.

1. grade_relevance     -> are retrieved chunks relevant to the question?
2. grade_groundedness  -> is the answer fully supported by the chunks (no hallucination)?
3. grade_answer        -> does the answer actually address the question?

Each returns {"pass": bool, "reason": str}.
"""
from llm_client import chat_json

RELEVANCE_SYSTEM = (
    "You are a strict grader. Given a user question and retrieved document chunks, "
    "decide if the chunks contain information relevant to answering the question. "
    'Respond ONLY with JSON: {"pass": true/false, "reason": "<one sentence>"}'
)

GROUNDEDNESS_SYSTEM = (
    "You are a strict fact-checker. Given document chunks and an answer, decide if "
    "EVERY claim in the answer is supported by the chunks. If the answer contains "
    "information not present in the chunks, it fails. "
    'Respond ONLY with JSON: {"pass": true/false, "reason": "<one sentence>"}'
)

ANSWER_SYSTEM = (
    "You are a strict grader. Given a user question and an answer, decide if the "
    "answer directly and usefully addresses the question (not evasive, not off-topic). "
    'Respond ONLY with JSON: {"pass": true/false, "reason": "<one sentence>"}'
)


def grade_relevance(question: str, chunks: list) -> dict:
    user = f"QUESTION:\n{question}\n\nCHUNKS:\n" + "\n---\n".join(chunks)
    return chat_json(RELEVANCE_SYSTEM, user)


def grade_groundedness(chunks: list, answer: str) -> dict:
    user = "CHUNKS:\n" + "\n---\n".join(chunks) + f"\n\nANSWER:\n{answer}"
    return chat_json(GROUNDEDNESS_SYSTEM, user)


def grade_answer(question: str, answer: str) -> dict:
    user = f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
    return chat_json(ANSWER_SYSTEM, user)
