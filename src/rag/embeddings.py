"""Embedding helpers for the local RAG pipeline."""

from typing import Any

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer


def load_sentence_transformer_model(model_name: str) -> SentenceTransformer:
    """Load a sentence-transformers embedding model."""
    return SentenceTransformer(model_name)


def embed_with_sentence_transformers(model_name: str, texts: list[str]) -> list[list[float]]:
    """Embed texts locally with sentence-transformers."""
    model = load_sentence_transformer_model(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def embed_with_ollama(base_url: str, model_name: str, texts: list[str]) -> list[list[float]]:
    """Embed texts using the local Ollama embeddings API."""
    vectors: list[list[float]] = []
    for text in texts:
        response = httpx.post(
            f"{base_url.rstrip('/')}/api/embeddings",
            json={"model": model_name, "prompt": text},
            timeout=120.0,
        )
        response.raise_for_status()
        response_json = response.json()
        vectors.append(response_json["embedding"])

    matrix = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, a_min=1e-12, a_max=None)
    return matrix.tolist()


def embed_texts(
    texts: list[str],
    embedding_provider: str,
    model_name: str,
    ollama_base_url: str | None = None,
) -> list[list[float]]:
    """Embed texts using the configured provider."""
    if embedding_provider == "ollama":
        if not ollama_base_url:
            raise ValueError("ollama_base_url is required when embedding_provider='ollama'")
        return embed_with_ollama(ollama_base_url, model_name, texts)

    if embedding_provider == "sentence_transformers":
        return embed_with_sentence_transformers(model_name, texts)

    raise ValueError(f"Unsupported embedding_provider: {embedding_provider}")
