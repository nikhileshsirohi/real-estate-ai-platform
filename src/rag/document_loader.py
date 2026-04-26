"""Load markdown knowledge documents for the RAG pipeline."""

from pathlib import Path

from src.rag.schemas import KnowledgeDocument


def load_markdown_documents(raw_dir: Path) -> list[KnowledgeDocument]:
    """Load all markdown documents from the knowledge raw directory."""
    documents: list[KnowledgeDocument] = []
    for file_path in sorted(raw_dir.glob("*.md")):
        content = file_path.read_text(encoding="utf-8").strip()
        title = content.splitlines()[0].lstrip("# ").strip() if content else file_path.stem
        documents.append(
            KnowledgeDocument(
                source_path=str(file_path),
                title=title,
                content=content,
            )
        )
    return documents
