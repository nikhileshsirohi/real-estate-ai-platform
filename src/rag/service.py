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


def filter_results_by_score(results: Sequence, min_score: float) -> list:
    """Keep only retrieval results that meet the configured similarity floor."""
    return [result for result in results if result.score >= min_score]


def build_insufficient_market_answer(question: str, results: Sequence) -> str:
    """Return a conservative answer when retrieval evidence is too weak."""
    best_score = max((result.score for result in results), default=0.0)
    return (
        "### Short Direct Answer\n"
        "I do not have enough strong retrieved market context to answer that confidently from the current knowledge base.\n\n"
        "### Brief Explanation\n"
        f"The available sources for this question were too weak or too broad to support a reliable grounded answer. "
        f"The best retrieved similarity score was {best_score:.3f}, which is below the confidence floor configured for this advisory flow.\n\n"
        "### Limitations Note\n"
        "The current corpus is mostly statewide planning and housing context. A stronger answer would require more targeted documents for the topic you asked about."
    )


def build_insufficient_property_answer(question: str, predicted_price: float, results: Sequence) -> str:
    """Return a conservative property-advisory answer when evidence is too weak."""
    best_score = max((result.score for result in results), default=0.0)
    return (
        "### Short Direct Answer\n"
        f"The model predicts about ${predicted_price * 100000:,.0f}, but I do not have enough strong retrieved market context to interpret that estimate confidently.\n\n"
        "### Practical Interpretation\n"
        f"The numerical model output is available, but the supporting RAG evidence is too weak or too broad to justify a strong market explanation for this property.\n\n"
        "### Market-Context Explanation\n"
        f"The best retrieved similarity score was {best_score:.3f}, which is below the configured evidence threshold for property advice.\n\n"
        "### Limitations Note\n"
        "The current corpus is mostly statewide planning and housing context, not detailed local comparable-property or neighborhood evidence."
    )


def ask_market_question(question: str) -> dict[str, object]:
    """Retrieve relevant context and generate an Ollama-backed answer."""
    rag_config = load_yaml_config("configs/rag_config.yaml")
    top_k = int(rag_config["generation_top_k"])
    base_url = str(rag_config["ollama_base_url"])
    model_name = str(rag_config["ollama_model_name"])
    temperature = float(rag_config["generation_temperature"])
    max_chars_per_source = int(rag_config.get("max_context_chars_per_source", 450))
    min_score = float(rag_config.get("market_min_retrieval_score", 0.0))
    min_sources_required = int(rag_config.get("market_min_sources_required", 1))

    log_event(logger, logging.INFO, "market_retrieval_started", top_k=top_k)
    retrieved_results = retrieve(query=question, top_k=top_k)
    results = filter_results_by_score(retrieved_results, min_score=min_score)
    log_event(
        logger,
        logging.INFO,
        "market_retrieval_completed",
        source_count=len(retrieved_results),
        filtered_source_count=len(results),
        min_score=min_score,
    )
    if len(results) < min_sources_required:
        log_event(
            logger,
            logging.INFO,
            "market_retrieval_below_threshold",
            required_sources=min_sources_required,
            filtered_source_count=len(results),
            min_score=min_score,
        )
        return {
            "answer": build_insufficient_market_answer(question, retrieved_results),
            "model_name": model_name,
            "sources": [asdict(result) for result in results or retrieved_results[:min(top_k, len(retrieved_results))]],
        }
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
    min_score = float(rag_config.get("property_min_retrieval_score", 0.0))
    min_sources_required = int(rag_config.get("property_min_sources_required", 1))

    log_event(logger, logging.INFO, "property_prediction_started", question=question)
    predicted_price = predict_price(property_features)
    log_event(logger, logging.INFO, "property_prediction_completed", predicted_price=predicted_price)
    log_event(logger, logging.INFO, "property_retrieval_started", top_k=top_k)
    retrieved_results = retrieve(query=question, top_k=top_k)
    results = filter_results_by_score(retrieved_results, min_score=min_score)
    log_event(
        logger,
        logging.INFO,
        "property_retrieval_completed",
        source_count=len(retrieved_results),
        filtered_source_count=len(results),
        min_score=min_score,
    )
    if len(results) < min_sources_required:
        log_event(
            logger,
            logging.INFO,
            "property_retrieval_below_threshold",
            required_sources=min_sources_required,
            filtered_source_count=len(results),
            min_score=min_score,
        )
        return {
            "answer": build_insufficient_property_answer(question, predicted_price, retrieved_results),
            "model_name": model_name,
            "predicted_price": predicted_price,
            "predicted_price_usd": predicted_price * 100000,
            "sources": [asdict(result) for result in results or retrieved_results[:min(top_k, len(retrieved_results))]],
        }
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
