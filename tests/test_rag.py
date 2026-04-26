"""Tests for the local RAG pipeline helpers."""

from pathlib import Path

from src.rag.chunking import chunk_documents
from src.rag.document_loader import load_markdown_documents
from src.rag.generator import build_market_prompt
from src.rag.schemas import KnowledgeDocument


def test_load_markdown_documents_reads_titles(tmp_path) -> None:
    doc_path = tmp_path / "sample.md"
    doc_path.write_text("# Sample Title\n\nSome content here.", encoding="utf-8")

    documents = load_markdown_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].title == "Sample Title"
    assert "Some content here." in documents[0].content


def test_chunk_documents_creates_chunks() -> None:
    documents = [
        KnowledgeDocument(
            source_path=str(Path("doc.md")),
            title="Doc",
            content="A" * 1200,
        )
    ]

    chunks = chunk_documents(documents, chunk_size=300, chunk_overlap=50)

    assert len(chunks) >= 4
    assert chunks[0].title == "Doc"
    assert chunks[0].chunk_id.startswith("doc-0-chunk-")


def test_build_market_prompt_contains_question_and_context() -> None:
    prompt = build_market_prompt(
        question="What supports California affordability pressure?",
        retrieved_context="Median value of owner-occupied housing units was $734,700.",
    )

    assert "What supports California affordability pressure?" in prompt
    assert "Median value of owner-occupied housing units was $734,700." in prompt
