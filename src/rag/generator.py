"""Ollama generation helpers for the advisory layer."""

from typing import Any

import httpx


def build_market_prompt(question: str, retrieved_context: str) -> str:
    """Build a grounded advisory prompt from retrieved context."""
    return (
        "You are a real estate market advisor. Use only the provided context to answer.\n"
        "If the context is insufficient, say that clearly.\n"
        "Be concise, practical, and factual.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{retrieved_context}\n\n"
        "Answer with:\n"
        "1. A short direct answer\n"
        "2. A brief explanation grounded in the context\n"
        "3. A short limitations note if needed"
    )


def generate_with_ollama(
    question: str,
    retrieved_context: str,
    base_url: str,
    model_name: str,
    temperature: float,
) -> str:
    """Call the local Ollama server to generate a grounded answer."""
    prompt = build_market_prompt(question=question, retrieved_context=retrieved_context)
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    response = httpx.post(
        f"{base_url.rstrip('/')}/api/generate",
        json=payload,
        timeout=120.0,
    )
    response.raise_for_status()
    response_json = response.json()
    return str(response_json.get("response", "")).strip()
