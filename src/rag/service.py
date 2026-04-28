"""High-level RAG advisory service using retrieval plus Ollama generation."""

import logging
from dataclasses import asdict
from typing import Sequence

from src.inference.predictor import predict_price
from src.rag.generator import generate_property_advice_with_ollama, generate_with_ollama
from src.rag.retrieve import retrieve
from src.utils.config_loader import load_yaml_config
from src.utils.logger import get_logger, log_event


logger = get_logger(__name__)


def build_context_from_results(results: Sequence, max_chars_per_source: int | None = None) -> str:
    """Format retrieved chunks into a prompt-friendly context block."""
    sections: list[str] = []
    for index, result in enumerate(results, start=1):
        content = result.content
        if max_chars_per_source is not None and len(content) > max_chars_per_source:
            content = f"{content[:max_chars_per_source].rstrip()}..."
        sections.append(
            f"[Source {index}] {result.title}\n"
            f"Path: {result.source_path}\n"
            f"Content: {content}"
        )
    return "\n\n".join(sections)


def ask_market_question(question: str) -> dict[str, object]:
    """Retrieve relevant context and generate an Ollama-backed answer."""
    rag_config = load_yaml_config("configs/rag_config.yaml")
    top_k = int(rag_config["generation_top_k"])
    base_url = str(rag_config["ollama_base_url"])
    model_name = str(rag_config["ollama_model_name"])
    temperature = float(rag_config["generation_temperature"])
    max_chars_per_source = int(rag_config.get("max_context_chars_per_source", 450))

    log_event(logger, logging.INFO, "market_retrieval_started", top_k=top_k)
    results = retrieve(query=question, top_k=top_k)
    context = build_context_from_results(results, max_chars_per_source=max_chars_per_source)
    log_event(logger, logging.INFO, "market_generation_started", source_count=len(results), model_name=model_name)
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


def build_property_summary(property_features: dict[str, float]) -> str:
    """Format property features into a compact summary block."""
    ordered_lines = [
        f"median_income: {property_features['median_income']}",
        f"house_age: {property_features['house_age']}",
        f"average_rooms: {property_features['average_rooms']}",
        f"average_bedrooms: {property_features['average_bedrooms']}",
        f"population: {property_features['population']}",
        f"average_occupancy: {property_features['average_occupancy']}",
        f"latitude: {property_features['latitude']}",
        f"longitude: {property_features['longitude']}",
    ]
    return "\n".join(ordered_lines)


def ask_property_question(question: str, property_features: dict[str, float]) -> dict[str, object]:
    """Generate property-specific advice using prediction plus retrieved context."""
    rag_config = load_yaml_config("configs/rag_config.yaml")
    top_k = int(rag_config.get("property_generation_top_k", rag_config["generation_top_k"]))
    base_url = str(rag_config["ollama_base_url"])
    model_name = str(rag_config.get("property_ollama_model_name", rag_config["ollama_model_name"]))
    temperature = float(rag_config["generation_temperature"])
    max_chars_per_source = int(rag_config.get("max_context_chars_per_source", 450))

    log_event(logger, logging.INFO, "property_prediction_started", question=question)
    predicted_price = predict_price(property_features)
    log_event(logger, logging.INFO, "property_prediction_completed", predicted_price=predicted_price)
    log_event(logger, logging.INFO, "property_retrieval_started", top_k=top_k)
    results = retrieve(query=question, top_k=top_k)
    log_event(logger, logging.INFO, "property_retrieval_completed", source_count=len(results))
    context = build_context_from_results(results, max_chars_per_source=max_chars_per_source)
    property_summary = build_property_summary(property_features)
    log_event(
        logger,
        logging.INFO,
        "property_generation_started",
        source_count=len(results),
        model_name=model_name,
        context_chars=len(context),
    )
    answer = generate_property_advice_with_ollama(
        question=question,
        property_summary=property_summary,
        predicted_price=predicted_price,
        retrieved_context=context,
        base_url=base_url,
        model_name=model_name,
        temperature=temperature,
    )

    return {
        "answer": answer,
        "model_name": model_name,
        "predicted_price": predicted_price,
        "predicted_price_usd": predicted_price * 100000,
        "sources": [asdict(result) for result in results],
    }
