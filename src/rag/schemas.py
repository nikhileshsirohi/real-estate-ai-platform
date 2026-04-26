"""Data structures for RAG documents, chunks, and retrieval results."""

from dataclasses import dataclass


@dataclass
class KnowledgeDocument:
    source_path: str
    title: str
    content: str


@dataclass
class KnowledgeChunk:
    chunk_id: str
    source_path: str
    title: str
    content: str


@dataclass
class RetrievalResult:
    chunk_id: str
    source_path: str
    title: str
    content: str
    score: float
