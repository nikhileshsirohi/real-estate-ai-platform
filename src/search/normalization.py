"""Query normalization helpers for natural-language property search."""

import re

from src.utils.config_loader import load_yaml_config


SEARCH_CONFIG_PATH = "configs/search_config.yaml"


def _convert_inr_to_usd(amount_in_inr: float) -> float:
    """Convert INR-denominated budget text into an approximate USD amount."""
    search_config = load_yaml_config(SEARCH_CONFIG_PATH)
    inr_per_usd = float(search_config.get("inr_per_usd", 83.0))
    return round(amount_in_inr / inr_per_usd, 2)


def normalize_property_search_query(query: str) -> str:
    """Normalize common real-estate shorthand into parser-friendly wording."""
    normalized = query.strip()

    # Normalize BHK shorthand into bedroom wording.
    normalized = re.sub(r"\b(\d+)\s*bhk\b", r"\1 bedroom", normalized, flags=re.IGNORECASE)

    # Normalize area units into sqft.
    normalized = re.sub(r"\b(square\s*feet|square\s*foot|sq\.?\s*ft\.?|sqft)\b", "sqft", normalized, flags=re.IGNORECASE)

    # Normalize lakh and crore budgets into approximate USD wording for this USD-based dataset.
    def replace_lakh(match: re.Match[str]) -> str:
        amount = float(match.group(1))
        usd_amount = _convert_inr_to_usd(amount * 100000)
        return f"{usd_amount:.0f} USD"

    def replace_crore(match: re.Match[str]) -> str:
        amount = float(match.group(1))
        usd_amount = _convert_inr_to_usd(amount * 10000000)
        return f"{usd_amount:.0f} USD"

    normalized = re.sub(r"\b(\d+(?:\.\d+)?)\s*lakh\b", replace_lakh, normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b(\d+(?:\.\d+)?)\s*crore\b", replace_crore, normalized, flags=re.IGNORECASE)

    # Normalize common budget phrases.
    normalized = re.sub(r"\bunder\s+\$", "under ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bover\s+\$", "over ", normalized, flags=re.IGNORECASE)

    return normalized
