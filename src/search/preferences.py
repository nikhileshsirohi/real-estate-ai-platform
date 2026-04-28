"""Preference detection and lightweight reranking for property search."""

from collections.abc import Sequence

from src.db.models import PropertyListing


def detect_search_preferences(query: str) -> list[str]:
    """Extract coarse user preferences from a natural-language property query."""
    lowered = query.lower()
    preferences: list[str] = []

    if any(token in lowered for token in ["family", "kids", "school", "family-friendly"]):
        preferences.append("family_friendly")
    if any(token in lowered for token in ["investment", "investor", "rental", "yield", "roi"]):
        preferences.append("investment")
    if any(token in lowered for token in ["bart", "transit", "train", "metro", "commute"]):
        preferences.append("transit_access")
    if any(token in lowered for token in ["spacious", "large", "bigger", "space", "sqft"]):
        preferences.append("more_space")

    return preferences


def _score_listing_for_preferences(listing: PropertyListing, preferences: Sequence[str]) -> float:
    """Assign a simple heuristic score to help rank matched listings for the stated intent."""
    score = 0.0
    description = listing.description.lower()
    title = listing.title.lower()
    property_type = listing.property_type.lower()

    if "family_friendly" in preferences:
        if listing.bedrooms >= 3:
            score += 3.0
        if property_type in {"house", "townhouse"}:
            score += 2.0
        if listing.area_sqft >= 1400:
            score += 2.0

    if "investment" in preferences:
        if property_type in {"condo", "apartment"}:
            score += 2.0
        if listing.asking_price_usd <= 900000:
            score += 2.0
        if "starter" in title or "entry-level" in description:
            score += 1.5

    if "transit_access" in preferences:
        if "bart" in description:
            score += 3.0
        if any(token in description for token in ["transit", "commute", "freeway", "transport"]):
            score += 2.0

    if "more_space" in preferences:
        score += min(listing.area_sqft / 1000.0, 3.0)

    return score


def rerank_property_listings(
    listings: Sequence[PropertyListing],
    preferences: Sequence[str],
) -> list[PropertyListing]:
    """Rerank listings when user intent suggests something beyond lowest-price ordering."""
    if not preferences:
        return list(listings)

    return sorted(
        listings,
        key=lambda listing: (
            -_score_listing_for_preferences(listing, preferences),
            listing.asking_price_usd,
        ),
    )
