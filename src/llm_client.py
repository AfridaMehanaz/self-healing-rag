"""Provider-agnostic LLM client (OpenAI-compatible: Groq, OpenAI, Ollama, etc.).

Reads LLM_BASE_URL / LLM_API_KEY / LLM_MODEL from environment or a .env file.
"""
import os
import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY = os.getenv("LLM_API_KEY", "set-me")
MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

_client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def chat(system: str, user: str, json_mode: bool = False, temperature: float = 0.0) -> str:
    """Single LLM call. Set json_mode=True to force a JSON object response."""
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = _client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **kwargs,
    )
    return resp.choices[0].message.content


def chat_json(system: str, user: str) -> dict:
    """LLM call that returns parsed JSON, tolerating stray code fences."""
    raw = chat(system, user, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
