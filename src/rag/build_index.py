"""Build a FAISS index from local knowledge documents."""

import json
from pathlib import Path

import faiss
import numpy as np

from src.rag.chunking import chunk_documents
from src.rag.document_loader import load_markdown_documents
from src.rag.embeddings import embed_texts, load_embedding_model
from src.utils.config_loader import load_yaml_config, resolve_project_path


RAG_CONFIG_PATH = Path("configs/rag_config.yaml")


def save_chunks_metadata(chunks_path: Path, metadata: list[dict[str, str]]) -> None:
    """Persist chunk metadata alongside the vector index."""
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    """Build and save the local FAISS knowledge index."""
    rag_config = load_yaml_config(RAG_CONFIG_PATH)
    raw_dir = resolve_project_path(str(rag_config["knowledge_raw_dir"]))
    index_output_dir = resolve_project_path(str(rag_config["index_output_dir"]))
    chunk_size = int(rag_config["chunk_size"])
    chunk_overlap = int(rag_config["chunk_overlap"])
    model_name = str(rag_config["embedding_model_name"])

    documents = load_markdown_documents(raw_dir)
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    model = load_embedding_model(model_name)
    vectors = embed_texts(model, [chunk.content for chunk in chunks])
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

    print(f"Saved FAISS index to {index_output_dir / 'knowledge.index'}")
    print(f"Saved chunk metadata to {index_output_dir / 'chunks.json'}")
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")


if __name__ == "__main__":
    main()
