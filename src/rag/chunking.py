"""Chunk knowledge documents into retrieval-friendly segments."""

from src.rag.schemas import KnowledgeChunk, KnowledgeDocument


def chunk_documents(
    documents: list[KnowledgeDocument],
    chunk_size: int,
    chunk_overlap: int,
) -> list[KnowledgeChunk]:
    """Split documents into overlapping text chunks."""
    chunks: list[KnowledgeChunk] = []
    step = max(1, chunk_size - chunk_overlap)

    for document_index, document in enumerate(documents):
        text = document.content
        for offset in range(0, len(text), step):
            chunk_text = text[offset : offset + chunk_size].strip()
            if not chunk_text:
                continue
            chunk_id = f"doc-{document_index}-chunk-{len(chunks)}"
            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    source_path=document.source_path,
                    title=document.title,
                    content=chunk_text,
                )
            )
            if offset + chunk_size >= len(text):
                break

    return chunks
