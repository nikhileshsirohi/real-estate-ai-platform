"""Build a FAISS index from local knowledge documents."""

import json
from pathlib import Path

import faiss
import numpy as np

from src.rag.chunking import chunk_documents
from src.rag.document_loader import load_markdown_documents
from src.rag.embeddings import embed_texts
from src.utils.config_loader import load_yaml_config, resolve_project_path


RAG_CONFIG_PATH = Path("configs/rag_config.yaml")


def save_chunks_metadata(chunks_path: Path, metadata: list[dict[str, str]]) -> None:
    """Persist chunk metadata alongside the vector index."""
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_index_metadata(index_metadata_path: Path, metadata: dict[str, object]) -> None:
    """Persist retrieval index metadata for later compatibility checks."""
    index_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    index_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    """Build and save the local FAISS knowledge index."""
    rag_config = load_yaml_config(RAG_CONFIG_PATH)
    raw_dir = resolve_project_path(str(rag_config["knowledge_raw_dir"]))
    index_output_dir = resolve_project_path(str(rag_config["index_output_dir"]))
    chunk_size = int(rag_config["chunk_size"])
    chunk_overlap = int(rag_config["chunk_overlap"])
    embedding_provider = str(rag_config["embedding_provider"])
    model_name = str(rag_config["embedding_model_name"])
    ollama_base_url = str(rag_config["ollama_base_url"])

    documents = load_markdown_documents(raw_dir)
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    vectors = embed_texts(
        texts=[chunk.content for chunk in chunks],
        embedding_provider=embedding_provider,
        model_name=model_name,
        ollama_base_url=ollama_base_url,
    )
    matrix = np.array(vectors, dtype="float32")

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    index_output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_output_dir / "knowledge.index"))
    save_chunks_metadata(
        index_output_dir / "chunks.json",
        [
            {
                "chunk_id": chunk.chunk_id,
                "source_path": chunk.source_path,
                "title": chunk.title,
                "content": chunk.content,
            }
            for chunk in chunks
        ],
    )
    save_index_metadata(
        index_output_dir / "index_metadata.json",
        {
            "embedding_provider": embedding_provider,
            "embedding_model_name": model_name,
            "vector_dimension": int(matrix.shape[1]),
            "document_count": len(documents),
            "chunk_count": len(chunks),
        },
    )

    print(f"Saved FAISS index to {index_output_dir / 'knowledge.index'}")
    print(f"Saved chunk metadata to {index_output_dir / 'chunks.json'}")
    print(f"Saved index metadata to {index_output_dir / 'index_metadata.json'}")
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")


if __name__ == "__main__":
    main()
