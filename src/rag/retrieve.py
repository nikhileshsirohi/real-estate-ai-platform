"""Retrieve relevant knowledge chunks from the local FAISS index."""

import json
from pathlib import Path

import faiss
import numpy as np

from src.rag.embeddings import embed_texts
from src.rag.schemas import RetrievalResult
from src.utils.config_loader import load_yaml_config, resolve_project_path


RAG_CONFIG_PATH = Path("configs/rag_config.yaml")


def load_chunks_metadata(chunks_path: Path) -> list[dict[str, str]]:
    """Load chunk metadata saved during index construction."""
    return json.loads(chunks_path.read_text(encoding="utf-8"))


def load_index_metadata(index_metadata_path: Path) -> dict[str, object]:
    """Load retrieval index metadata if it exists."""
    if not index_metadata_path.exists():
        return {}
    return json.loads(index_metadata_path.read_text(encoding="utf-8"))


def retrieve(query: str, top_k: int | None = None) -> list[RetrievalResult]:
    """Retrieve the most relevant chunks for a query."""
    rag_config = load_yaml_config(RAG_CONFIG_PATH)
    index_output_dir = resolve_project_path(str(rag_config["index_output_dir"]))
    embedding_provider = str(rag_config["embedding_provider"])
    model_name = str(rag_config["embedding_model_name"])
    ollama_base_url = str(rag_config["ollama_base_url"])
    k = int(top_k or rag_config["top_k"])

    index = faiss.read_index(str(index_output_dir / "knowledge.index"))
    metadata = load_chunks_metadata(index_output_dir / "chunks.json")
    index_metadata = load_index_metadata(index_output_dir / "index_metadata.json")
    query_vector = np.array(
        embed_texts(
            texts=[query],
            embedding_provider=embedding_provider,
            model_name=model_name,
            ollama_base_url=ollama_base_url,
        ),
        dtype="float32",
    )

    if query_vector.ndim != 2 or query_vector.shape[0] != 1:
        raise RuntimeError(f"Unexpected query embedding shape: {query_vector.shape}")

    query_dimension = int(query_vector.shape[1])
    index_dimension = int(index.d)
    if query_dimension != index_dimension:
        saved_model_name = index_metadata.get("embedding_model_name", "unknown")
        saved_provider = index_metadata.get("embedding_provider", "unknown")
        raise RuntimeError(
            "Knowledge index dimension mismatch. "
            f"Saved index dimension={index_dimension} (provider={saved_provider}, model={saved_model_name}), "
            f"but current query embedding dimension={query_dimension} "
            f"(provider={embedding_provider}, model={model_name}). "
            "Rebuild the index with `python -m src.rag.build_index`."
        )

    scores, indices = index.search(query_vector, k)
    results: list[RetrievalResult] = []
    for score, index_position in zip(scores[0], indices[0]):
        if index_position < 0:
            continue
        item = metadata[index_position]
        results.append(
            RetrievalResult(
                chunk_id=item["chunk_id"],
                source_path=item["source_path"],
                title=item["title"],
                content=item["content"],
                score=float(score),
            )
        )
    return results


def main() -> None:
    """Simple CLI retrieval smoke test."""
    sample_query = "What official California housing facts should I use for statewide affordability context?"
    results = retrieve(sample_query)
    for result in results:
        print(f"[{result.score:.4f}] {result.title} -> {result.source_path}")
        print(result.content[:300])
        print("-" * 80)


if __name__ == "__main__":
    main()
