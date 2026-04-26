"""High-level RAG advisory service using retrieval plus Ollama generation."""

from dataclasses import asdict

from src.rag.generator import generate_with_ollama
from src.rag.retrieve import retrieve
from src.utils.config_loader import load_yaml_config


def build_context_from_results(results) -> str:
    """Format retrieved chunks into a prompt-friendly context block."""
    sections: list[str] = []
    for index, result in enumerate(results, start=1):
        sections.append(
            f"[Source {index}] {result.title}\n"
            f"Path: {result.source_path}\n"
            f"Content: {result.content}"
        )
    return "\n\n".join(sections)


def ask_market_question(question: str) -> dict[str, object]:
    """Retrieve relevant context and generate an Ollama-backed answer."""
    rag_config = load_yaml_config("configs/rag_config.yaml")
    top_k = int(rag_config["generation_top_k"])
    base_url = str(rag_config["ollama_base_url"])
    model_name = str(rag_config["ollama_model_name"])
    temperature = float(rag_config["generation_temperature"])

    results = retrieve(query=question, top_k=top_k)
    context = build_context_from_results(results)
    answer = generate_with_ollama(
        question=question,
        retrieved_context=context,
        base_url=base_url,
        model_name=model_name,
        temperature=temperature,
    )

    return {
        "answer": answer,
        "model_name": model_name,
        "sources": [asdict(result) for result in results],
    }
