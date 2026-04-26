"""Retrieve relevant knowledge chunks from the local FAISS index."""

import json
from pathlib import Path

import faiss
import numpy as np

from src.rag.embeddings import embed_texts, load_embedding_model
from src.rag.schemas import RetrievalResult
from src.utils.config_loader import load_yaml_config


RAG_CONFIG_PATH = Path("configs/rag_config.yaml")


def load_chunks_metadata(chunks_path: Path) -> list[dict[str, str]]:
    """Load chunk metadata saved during index construction."""
    return json.loads(chunks_path.read_text(encoding="utf-8"))


def retrieve(query: str, top_k: int | None = None) -> list[RetrievalResult]:
    """Retrieve the most relevant chunks for a query."""
    rag_config = load_yaml_config(RAG_CONFIG_PATH)
    index_output_dir = Path(str(rag_config["index_output_dir"]))
    model_name = str(rag_config["embedding_model_name"])
    k = int(top_k or rag_config["top_k"])

    index = faiss.read_index(str(index_output_dir / "knowledge.index"))
    metadata = load_chunks_metadata(index_output_dir / "chunks.json")
    model = load_embedding_model(model_name)
    query_vector = np.array(embed_texts(model, [query]), dtype="float32")

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
