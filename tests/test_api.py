"""Tests for API endpoints."""

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health_check_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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
