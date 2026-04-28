"""Tests for the local RAG pipeline helpers."""

from pathlib import Path

from src.rag.chunking import chunk_documents
from src.rag.document_loader import load_markdown_documents
from src.rag.generator import build_market_prompt, build_property_advisory_prompt
from src.rag.service import build_property_summary
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


def test_build_property_summary_includes_expected_fields() -> None:
    summary = build_property_summary(
        {
            "median_income": 8.3252,
            "house_age": 41.0,
            "average_rooms": 6.984127,
            "average_bedrooms": 1.02381,
            "population": 322.0,
            "average_occupancy": 2.555556,
            "latitude": 37.88,
            "longitude": -122.23,
        }
    )

    assert "median_income: 8.3252" in summary
    assert "longitude: -122.23" in summary


def test_build_property_advisory_prompt_includes_local_listing_context() -> None:
    prompt = build_property_advisory_prompt(
        question="How should I interpret this predicted price?",
        property_summary="median_income: 8.3252",
        predicted_price=3.95,
        retrieved_context="Statewide context here.",
        local_listing_context="Nearby listing snapshot here.",
    )

    assert "Nearby listing snapshot here." in prompt
    assert "Treat local listing context as demo nearby inventory" in prompt
