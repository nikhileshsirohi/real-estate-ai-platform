"""Natural-language parser for property search requests."""

import json
from typing import Any

import httpx

from src.api.schemas import PropertySearchFilters
from src.search.normalization import normalize_property_search_query
from src.utils.config_loader import load_yaml_config


SEARCH_CONFIG_PATH = "configs/search_config.yaml"


def build_property_search_prompt(query: str, limit: int) -> str:
    """Build a prompt that converts a property-search query into strict JSON filters."""
    return (
        "You convert real-estate property search requests into structured JSON filters.\n"
        "Return only valid JSON with no markdown.\n"
        "Use null for unknown values.\n"
        "All prices are in USD.\n"
        "Allowed keys are exactly:\n"
        "city, locality, property_type, min_price_usd, max_price_usd, "
        "min_bedrooms, max_bedrooms, min_bathrooms, max_bathrooms, "
        "min_area_sqft, max_area_sqft, limit, sort_by, sort_order.\n"
        "Allowed property_type values: condo, apartment, house, townhouse.\n"
        "Allowed sort_by values: asking_price_usd, area_sqft, bedrooms, created_at.\n"
        "Allowed sort_order values: asc, desc.\n"
        f"Set limit to {limit} unless the user clearly asks for fewer.\n\n"
        f"User query:\n{query}\n"
    )


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    """Extract a JSON object from model output, even if extra text slips through."""
    raw_text = raw_text.strip()
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Parser model did not return a JSON object: {raw_text}")
    return json.loads(raw_text[start : end + 1])


def _normalize_filters(payload: dict[str, Any], limit: int) -> PropertySearchFilters:
    """Normalize parser output into the validated Pydantic filter schema."""
    normalized_payload = dict(payload)
    normalized_payload["limit"] = int(normalized_payload.get("limit") or limit)
    normalized_payload["sort_by"] = str(normalized_payload.get("sort_by") or "asking_price_usd")
    normalized_payload["sort_order"] = str(normalized_payload.get("sort_order") or "asc")
    return PropertySearchFilters(**normalized_payload)


def parse_property_search_query(query: str, limit: int) -> tuple[PropertySearchFilters, str]:
    """Use Ollama to convert a natural-language search query into structured filters."""
    search_config = load_yaml_config(SEARCH_CONFIG_PATH)
    base_url = str(search_config["ollama_base_url"])
    model_name = str(search_config["parser_model_name"])
    normalized_query = normalize_property_search_query(query)
    prompt = build_property_search_prompt(query=normalized_query, limit=limit)

    response = httpx.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": "10m",
            "options": {
                "temperature": 0.0,
                "num_predict": 220,
            },
        },
        timeout=120.0,
    )
    response.raise_for_status()
    response_json = response.json()
    filters_payload = _extract_json_object(str(response_json.get("response", "")))
    return _normalize_filters(filters_payload, limit=limit), model_name
