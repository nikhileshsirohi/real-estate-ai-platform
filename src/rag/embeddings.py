"""Embedding helpers for the local RAG pipeline."""

from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """Load a sentence-transformers embedding model."""
    return SentenceTransformer(model_name)


def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """Embed a list of texts into dense vectors."""
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()
