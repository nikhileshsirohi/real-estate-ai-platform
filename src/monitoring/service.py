"""Monitoring/evaluation summary builders."""

import json
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.db.models import PredictionRecord, PropertyListing
from src.monitoring.runtime import runtime_monitor
from src.utils.config_loader import resolve_project_path


MODEL_METRICS_PATH = Path("models/xgboost_price_model_metrics.json")
RAG_INDEX_METADATA_PATH = Path("data/knowledge/index/index_metadata.json")


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    """Load JSON from disk if present, otherwise return an empty mapping."""
    resolved_path = resolve_project_path(path)
    if not resolved_path.exists():
        return {}
    return json.loads(resolved_path.read_text(encoding="utf-8"))


def build_model_evaluation_summary() -> dict[str, Any]:
    """Return a lightweight summary of current prediction-model metrics."""
    metrics = _load_json_if_exists(MODEL_METRICS_PATH)
    if not metrics:
        return {
            "artifact_found": False,
            "artifact_path": str(resolve_project_path(MODEL_METRICS_PATH)),
            "metrics": {},
        }

    return {
        "artifact_found": True,
        "artifact_path": str(resolve_project_path(MODEL_METRICS_PATH)),
        "metrics": metrics,
        "best_available_model": "xgboost_price_model_tuned_clean.joblib",
    }


def build_rag_evaluation_summary() -> dict[str, Any]:
    """Return a summary of the current retrieval index state."""
    metadata = _load_json_if_exists(RAG_INDEX_METADATA_PATH)
    if not metadata:
        return {
            "artifact_found": False,
            "artifact_path": str(resolve_project_path(RAG_INDEX_METADATA_PATH)),
        }

    return {
        "artifact_found": True,
        "artifact_path": str(resolve_project_path(RAG_INDEX_METADATA_PATH)),
        "embedding_provider": metadata.get("embedding_provider"),
        "embedding_model_name": metadata.get("embedding_model_name"),
        "vector_dimension": metadata.get("vector_dimension"),
        "document_count": metadata.get("document_count"),
        "chunk_count": metadata.get("chunk_count"),
    }


def build_database_monitoring_summary(db: Session) -> dict[str, Any]:
    """Summarize stored prediction and property-listing inventory counts."""
    prediction_count = db.scalar(select(func.count()).select_from(PredictionRecord)) or 0
    property_count = db.scalar(select(func.count()).select_from(PropertyListing)) or 0
    min_listing_price = db.scalar(select(func.min(PropertyListing.asking_price_usd)))
    max_listing_price = db.scalar(select(func.max(PropertyListing.asking_price_usd)))
    city_count = db.scalar(select(func.count(func.distinct(PropertyListing.city)))) or 0

    return {
        "prediction_record_count": int(prediction_count),
        "property_listing_count": int(property_count),
        "distinct_city_count": int(city_count),
        "min_listing_price_usd": float(min_listing_price) if min_listing_price is not None else None,
        "max_listing_price_usd": float(max_listing_price) if max_listing_price is not None else None,
    }


def build_inventory_evaluation_summary(db: Session) -> dict[str, Any]:
    """Return a city-level snapshot of the seeded property inventory."""
    rows = db.execute(
        select(
            PropertyListing.city,
            func.count(PropertyListing.id),
            func.min(PropertyListing.asking_price_usd),
            func.max(PropertyListing.asking_price_usd),
        )
        .group_by(PropertyListing.city)
        .order_by(PropertyListing.city.asc())
    ).all()

    cities = [
        {
            "city": city,
            "listing_count": int(listing_count),
            "min_price_usd": float(min_price),
            "max_price_usd": float(max_price),
        }
        for city, listing_count, min_price, max_price in rows
    ]

    return {
        "cities": cities,
        "total_cities": len(cities),
    }


def build_monitoring_summary(db: Session) -> dict[str, Any]:
    """Combine runtime, DB, model, and RAG monitoring into one summary."""
    return {
        "runtime": runtime_monitor.snapshot(),
        "database": build_database_monitoring_summary(db),
        "model": build_model_evaluation_summary(),
        "rag_index": build_rag_evaluation_summary(),
    }


def build_evaluation_summary(db: Session) -> dict[str, Any]:
    """Combine evaluation-oriented summaries for recruiter/demo visibility."""
    return {
        "model_evaluation": build_model_evaluation_summary(),
        "rag_evaluation": build_rag_evaluation_summary(),
        "inventory_evaluation": build_inventory_evaluation_summary(db),
    }
