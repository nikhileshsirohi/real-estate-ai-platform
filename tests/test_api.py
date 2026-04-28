"""Tests for API endpoints."""

from collections.abc import Generator
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from src.api.schemas import PropertySearchFilters
from src.api.main import app
from src.api.routes import get_db


class DummySession:
    """Minimal stand-in session for API tests."""

    def __init__(self) -> None:
        self.records = [
            SimpleNamespace(
                id=2,
                model_name="xgboost",
                predicted_price=3.95,
                median_income=8.32,
                house_age=41.0,
                average_rooms=6.98,
                average_bedrooms=1.02,
                population=322.0,
                average_occupancy=2.55,
                latitude=37.88,
                longitude=-122.23,
                created_at=datetime(2026, 4, 26, 12, 0, tzinfo=timezone.utc),
            ),
            SimpleNamespace(
                id=1,
                model_name="xgboost",
                predicted_price=4.10,
                median_income=9.10,
                house_age=20.0,
                average_rooms=7.50,
                average_bedrooms=1.10,
                population=500.0,
                average_occupancy=2.80,
                latitude=37.70,
                longitude=-122.10,
                created_at=datetime(2026, 4, 26, 11, 30, tzinfo=timezone.utc),
            ),
        ]

    def add(self, _record: object) -> None:
        return None

    def commit(self) -> None:
        return None

    def refresh(self, record: object) -> None:
        setattr(record, "id", 1)

    def scalars(self, _stmt: object) -> "DummySession":
        return self

    def all(self) -> list[SimpleNamespace]:
        return self.records

    def scalar(self, _stmt: object):
        return self.records[0]

    def close(self) -> None:
        return None


def override_get_db() -> Generator[Session, None, None]:
    """Provide a fake database session for tests."""
    yield DummySession()  # type: ignore[misc]


app.dependency_overrides[get_db] = override_get_db


client = TestClient(app)


def test_root_returns_api_summary() -> None:
    response = client.get("/")

    assert response.status_code == 200
    response_body = response.json()
    assert response_body["docs"] == "/docs"
    assert response_body["health"] == "/health"
    assert response_body["predictions"] == "/predictions?limit=10"
    assert response_body["search_properties"] == "/search-properties?city=San%20Jose&max_price_usd=900000"
    assert response_body["recommend_properties"] == "/search-properties/recommend"


def test_health_check_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_predictions_returns_history() -> None:
    response = client.get("/predictions?limit=2")

    assert response.status_code == 200
    response_body = response.json()
    assert response_body["count"] == 2
    assert len(response_body["items"]) == 2
    assert response_body["items"][0]["model_name"] == "xgboost"
    assert response_body["items"][0]["id"] == 2


def test_list_predictions_supports_filters() -> None:
    response = client.get("/predictions?limit=2&model_name=xgboost&min_predicted_price=3.0")

    assert response.status_code == 200
    response_body = response.json()
    assert response_body["count"] == 2


def test_get_prediction_detail_returns_item() -> None:
    response = client.get("/predictions/2")

    assert response.status_code == 200
    response_body = response.json()
    assert response_body["id"] == 2
    assert response_body["model_name"] == "xgboost"


def test_predict_price_returns_prediction() -> None:
    payload = {
        "median_income": 8.3252,
        "house_age": 41.0,
        "average_rooms": 6.984127,
        "average_bedrooms": 1.02381,
        "population": 322.0,
        "average_occupancy": 2.555556,
        "latitude": 37.88,
        "longitude": -122.23,
    }

    response = client.post("/predict-price", json=payload)

    assert response.status_code == 200
    response_body = response.json()
    assert "predicted_price" in response_body
    assert isinstance(response_body["model_name"], str)
    assert response_body["prediction_id"] == 1


def test_ask_market_returns_advice() -> None:
    with patch("src.api.routes.ask_market_question") as mocked_ask_market:
        mocked_ask_market.return_value = {
            "answer": "Statewide affordability remains pressured by high ownership costs and elevated housing values.",
            "model_name": "qwen2.5:14b",
            "sources": [
                {
                    "chunk_id": "doc-0-chunk-0",
                    "source_path": "data/knowledge/raw/california_housing_snapshot.md",
                    "title": "California Housing Snapshot",
                    "content": "Median value of owner-occupied housing units was $734,700.",
                    "score": 0.92,
                }
            ],
        }

        response = client.post(
            "/ask-market",
            json={"question": "What supports California affordability pressure?"},
        )

    assert response.status_code == 200
    response_body = response.json()
    assert response_body["model_name"] == "qwen2.5:14b"
    assert len(response_body["sources"]) == 1


def test_advise_property_returns_property_advice() -> None:
    with patch("src.api.routes.ask_property_question") as mocked_ask_property:
        mocked_ask_property.return_value = {
            "answer": "The model suggests the property is expensive but still consistent with high-income local conditions.",
            "model_name": "qwen2.5:14b",
            "predicted_price": 3.95,
            "predicted_price_usd": 395000.0,
            "sources": [
                {
                    "chunk_id": "doc-0-chunk-0",
                    "source_path": "data/knowledge/raw/california_housing_snapshot.md",
                    "title": "California Housing Snapshot",
                    "content": "Median value of owner-occupied housing units was $734,700.",
                    "score": 0.92,
                }
            ],
        }

        response = client.post(
            "/advise-property",
            json={
                "question": "How should I interpret this predicted price?",
                "median_income": 8.3252,
                "house_age": 41.0,
                "average_rooms": 6.984127,
                "average_bedrooms": 1.02381,
                "population": 322.0,
                "average_occupancy": 2.555556,
                "latitude": 37.88,
                "longitude": -122.23,
            },
        )

    assert response.status_code == 200
    response_body = response.json()
    assert response_body["model_name"] == "qwen2.5:14b"
    assert response_body["predicted_price"] == 3.95
    assert response_body["predicted_price_usd"] == 395000.0


def test_search_properties_returns_filtered_items() -> None:
    with patch("src.api.routes.search_property_listings") as mocked_search:
        mocked_search.return_value = [
            SimpleNamespace(
                id=10,
                listing_code="CA-SJ-002",
                title="North San Jose Apartment",
                city="San Jose",
                locality="North San Jose",
                property_type="apartment",
                bedrooms=2,
                bathrooms=2.0,
                area_sqft=960.0,
                asking_price_usd=785000.0,
                description="Two-bedroom apartment-style unit near major employers.",
                latitude=37.387,
                longitude=-121.93,
                created_at=datetime(2026, 4, 28, 12, 0, tzinfo=timezone.utc),
            )
        ]

        response = client.get("/search-properties?city=San%20Jose&max_price_usd=900000")

    assert response.status_code == 200
    response_body = response.json()
    assert response_body["count"] == 1
    assert response_body["items"][0]["listing_code"] == "CA-SJ-002"
    assert response_body["applied_filters"]["city"] == "San Jose"


def test_search_properties_query_parses_and_returns_items() -> None:
    with (
        patch("src.api.routes.parse_property_search_query") as mocked_parse,
        patch("src.api.routes.search_property_listings_with_fallback") as mocked_search,
    ):
        mocked_parse.return_value = (
            PropertySearchFilters(
                city="Oakland",
                property_type="condo",
                max_price_usd=800000.0,
                min_bedrooms=2,
                limit=5,
                sort_by="asking_price_usd",
                sort_order="asc",
            ),
            "qwen2.5-coder:7b-instruct",
        )
        mocked_search.return_value = (
            [
                SimpleNamespace(
                    id=11,
                    listing_code="CA-OAK-001",
                    title="Lake Merritt Condo",
                    city="Oakland",
                    locality="Lake Merritt",
                    property_type="condo",
                    bedrooms=2,
                    bathrooms=2.0,
                    area_sqft=1040.0,
                    asking_price_usd=725000.0,
                    description="Condo close to Lake Merritt.",
                    latitude=37.809,
                    longitude=-122.257,
                    created_at=datetime(2026, 4, 28, 12, 5, tzinfo=timezone.utc),
                )
            ],
            "exact",
            None,
        )

        response = client.post(
            "/search-properties/query",
            json={"query": "Find me a 2 bedroom condo in Oakland under 800000", "limit": 5},
        )

    assert response.status_code == 200
    response_body = response.json()
    assert response_body["count"] == 1
    assert response_body["parser_model_name"] == "qwen2.5-coder:7b-instruct"
    assert response_body["items"][0]["city"] == "Oakland"
    assert response_body["match_strategy"] == "exact"
    assert response_body["detected_preferences"] == []


def test_search_properties_recommend_returns_answer() -> None:
    with (
        patch("src.api.routes.parse_property_search_query") as mocked_parse,
        patch("src.api.routes.search_property_listings_with_fallback") as mocked_search,
        patch("src.api.routes.recommend_property_results") as mocked_recommend,
    ):
        mocked_parse.return_value = (
            PropertySearchFilters(
                city="San Jose",
                max_price_usd=900000.0,
                min_bedrooms=2,
                limit=3,
            ),
            "qwen2.5-coder:7b-instruct",
        )
        mocked_search.return_value = (
            [
                SimpleNamespace(
                    id=12,
                    listing_code="CA-SJ-002",
                    title="North San Jose Apartment",
                    city="San Jose",
                    locality="North San Jose",
                    property_type="apartment",
                    bedrooms=2,
                    bathrooms=2.0,
                    area_sqft=960.0,
                    asking_price_usd=785000.0,
                    description="Two-bedroom apartment-style unit near major employers.",
                    latitude=37.387,
                    longitude=-121.93,
                    created_at=datetime(2026, 4, 28, 12, 15, tzinfo=timezone.utc),
                )
            ],
            "exact",
            None,
        )
        mocked_recommend.return_value = (
            "North San Jose Apartment is the best fit because it stays under budget and matches the requested bedroom count.",
            "qwen2.5-coder:7b-instruct",
        )

        response = client.post(
            "/search-properties/recommend",
            json={"query": "find me a 2 bhk in san jose under 900000", "limit": 3},
        )

    assert response.status_code == 200
    response_body = response.json()
    assert response_body["count"] == 1
    assert response_body["parser_model_name"] == "qwen2.5-coder:7b-instruct"
    assert response_body["recommendation_model_name"] == "qwen2.5-coder:7b-instruct"
    assert "best fit" in response_body["answer"].lower()
    assert response_body["match_strategy"] == "exact"
    assert response_body["detected_preferences"] == []


def test_search_properties_recommend_handles_no_results() -> None:
    with (
        patch("src.api.routes.parse_property_search_query") as mocked_parse,
        patch("src.api.routes.search_property_listings_with_fallback") as mocked_search,
        patch("src.api.routes.recommend_property_results") as mocked_recommend,
    ):
        mocked_parse.return_value = (
            PropertySearchFilters(
                city="Oakland",
                max_price_usd=96386.0,
                min_bedrooms=2,
                max_bedrooms=2,
                limit=5,
            ),
            "qwen2.5-coder:7b-instruct",
        )
        mocked_search.return_value = (
            [],
            "exact",
            "No listings matched the current non-price filters.",
        )
        mocked_recommend.return_value = (
            "No property listings matched the current search filters in the local database.",
            "qwen2.5-coder:7b-instruct",
        )

        response = client.post(
            "/search-properties/recommend",
            json={"query": "Find me a 2 BHK in Oakland under 80 lakh", "limit": 5},
        )

    assert response.status_code == 200
    response_body = response.json()
    assert response_body["count"] == 0
    assert "no property listings matched" in response_body["answer"].lower()
    assert response_body["match_strategy"] == "exact"
    assert response_body["detected_preferences"] == []


def test_search_properties_query_returns_closest_matches_when_budget_too_low() -> None:
    with (
        patch("src.api.routes.parse_property_search_query") as mocked_parse,
        patch("src.api.routes.search_property_listings_with_fallback") as mocked_search,
    ):
        mocked_parse.return_value = (
            PropertySearchFilters(
                city="Oakland",
                max_price_usd=96386.0,
                min_bedrooms=2,
                limit=2,
            ),
            "qwen2.5-coder:7b-instruct",
        )
        mocked_search.return_value = (
            [
                SimpleNamespace(
                    id=13,
                    listing_code="CA-OAK-001",
                    title="Lake Merritt Condo",
                    city="Oakland",
                    locality="Lake Merritt",
                    property_type="condo",
                    bedrooms=2,
                    bathrooms=2.0,
                    area_sqft=1040.0,
                    asking_price_usd=725000.0,
                    description="Condo close to Lake Merritt.",
                    latitude=37.809,
                    longitude=-122.257,
                    created_at=datetime(2026, 4, 28, 12, 25, tzinfo=timezone.utc),
                )
            ],
            "closest_match",
            "No exact matches were found under the requested budget. Showing closest matches instead.",
        )

        response = client.post(
            "/search-properties/query",
            json={"query": "Find me a 2 BHK in Oakland under 80 lakh", "limit": 2},
        )

    assert response.status_code == 200
    response_body = response.json()
    assert response_body["count"] == 1
    assert response_body["match_strategy"] == "closest_match"
    assert "closest matches" in response_body["advisory_note"].lower()
    assert response_body["detected_preferences"] == []
