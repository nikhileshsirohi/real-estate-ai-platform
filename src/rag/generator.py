"""Ollama generation helpers for the advisory layer."""

from typing import Any

import httpx


def build_market_prompt(question: str, retrieved_context: str) -> str:
    """Build a grounded advisory prompt from retrieved context."""
    return (
        "You are a real estate market advisor. Use only the provided context to answer.\n"
        "Do not use outside knowledge.\n"
        "If the context is weak, broad, or insufficient, say that clearly.\n"
        "Do not make city-level, neighborhood-level, or property-level claims unless the context directly supports them.\n"
        "Be concise, practical, and factual.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{retrieved_context}\n\n"
        "Answer with:\n"
        "1. A short direct answer\n"
        "2. A brief explanation grounded in the context\n"
        "3. A short limitations note\n"
        "If the context is only statewide or planning-oriented, explicitly say that."
    )


def build_property_advisory_prompt(
    question: str,
    property_summary: str,
    predicted_price: float,
    retrieved_context: str,
) -> str:
    """Build a grounded prompt for property-level advisory answers."""
    return (
        "You are a real estate price and market advisor.\n"
        "Use the provided model estimate and retrieved context only.\n"
        "Do not invent neighborhood facts, local comps, or area-specific claims that are not in the context.\n"
        "Do not present statewide demographic context as direct evidence for this exact property.\n"
        "If context is broad or planning-oriented, say so clearly.\n"
        "If the retrieved sources are weak for this property question, be conservative.\n\n"
        f"Property summary:\n{property_summary}\n\n"
        f"Predicted price (in 100,000 USD units): {predicted_price:.4f}\n"
        f"Predicted price (approx USD): ${predicted_price * 100000:,.0f}\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{retrieved_context}\n\n"
        "Answer with:\n"
        "1. A short direct answer\n"
        "2. A practical interpretation of the model estimate\n"
        "3. A short market-context explanation grounded in the retrieved sources only\n"
        "4. A short limitations note"
    )


def _generate_with_ollama_prompt(
    prompt: str,
    base_url: str,
    model_name: str,
    temperature: float,
) -> str:
    """Call the local Ollama server with a prepared prompt."""
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": temperature,
            "num_predict": 350,
        },
    }

    response = httpx.post(
        f"{base_url.rstrip('/')}/api/generate",
        json=payload,
        timeout=180.0,
    )
    response.raise_for_status()
    response_json = response.json()
    return str(response_json.get("response", "")).strip()


def generate_with_ollama(
    question: str,
    retrieved_context: str,
    base_url: str,
    model_name: str,
    temperature: float,
) -> str:
    """Call the local Ollama server to generate a grounded answer."""
    prompt = build_market_prompt(question=question, retrieved_context=retrieved_context)
    return _generate_with_ollama_prompt(
        prompt=prompt,
        base_url=base_url,
        model_name=model_name,
        temperature=temperature,
    )


def generate_property_advice_with_ollama(
    question: str,
    property_summary: str,
    predicted_price: float,
    retrieved_context: str,
    base_url: str,
    model_name: str,
    temperature: float,
) -> str:
    """Call Ollama for a property-specific grounded advisory answer."""
    prompt = build_property_advisory_prompt(
        question=question,
        property_summary=property_summary,
        predicted_price=predicted_price,
        retrieved_context=retrieved_context,
    )
    return _generate_with_ollama_prompt(
        prompt=prompt,
        base_url=base_url,
        model_name=model_name,
        temperature=temperature,
    )
