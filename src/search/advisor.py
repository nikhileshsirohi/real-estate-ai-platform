"""LLM-backed explanation and recommendation helpers for property search results."""

from typing import Any, Sequence

import httpx

from src.api.schemas import PropertySearchFilters
from src.db.models import PropertyListing
from src.utils.config_loader import load_yaml_config


SEARCH_CONFIG_PATH = "configs/search_config.yaml"


def build_property_recommendation_prompt(
    query: str,
    filters: PropertySearchFilters,
    listings: Sequence[PropertyListing],
) -> str:
    """Build a grounded prompt to explain matched listings."""
    listing_lines: list[str] = []
    for index, listing in enumerate(listings, start=1):
        listing_lines.append(
            f"[Listing {index}] {listing.title}\n"
            f"Code: {listing.listing_code}\n"
            f"City: {listing.city}\n"
            f"Locality: {listing.locality}\n"
            f"Type: {listing.property_type}\n"
            f"Bedrooms: {listing.bedrooms}\n"
            f"Bathrooms: {listing.bathrooms}\n"
            f"Area sqft: {listing.area_sqft}\n"
            f"Price USD: {listing.asking_price_usd}\n"
            f"Description: {listing.description}"
        )

    return (
        "You are a real-estate search assistant.\n"
        "Use only the provided matched listings.\n"
        "Do not invent listings or facts that are not present.\n"
        "Be practical and concise.\n\n"
        f"User search query:\n{query}\n\n"
        f"Applied filters:\n{filters.model_dump_json(indent=2)}\n\n"
        f"Matched listings:\n{chr(10).join(listing_lines)}\n\n"
        "Answer with:\n"
        "1. A short summary of what matched\n"
        "2. Top 2 or 3 recommended listings by listing code, with reasons grounded in the fields above\n"
        "3. A short note if tradeoffs are visible, such as budget vs size"
    )


def recommend_property_results(
    query: str,
    filters: PropertySearchFilters,
    listings: Sequence[PropertyListing],
) -> tuple[str, str]:
    """Generate a grounded explanation for the matched property results."""
    search_config = load_yaml_config(SEARCH_CONFIG_PATH)
    base_url = str(search_config["ollama_base_url"])
    model_name = str(search_config["recommendation_model_name"])
    prompt = build_property_recommendation_prompt(query=query, filters=filters, listings=listings)

    response = httpx.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.1,
                "num_predict": 350,
            },
        },
        timeout=180.0,
    )
    response.raise_for_status()
    response_json: dict[str, Any] = response.json()
    return str(response_json.get("response", "")).strip(), model_name
