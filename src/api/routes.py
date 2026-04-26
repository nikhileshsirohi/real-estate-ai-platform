"""API route definitions."""

from fastapi import APIRouter

from src.api.schemas import PricePredictionRequest, PricePredictionResponse
from src.inference.predictor import predict_price


router = APIRouter()


@router.get("/health")
def health_check() -> dict[str, str]:
    """Simple health endpoint for quick service checks."""
    return {"status": "ok"}


@router.post("/predict-price", response_model=PricePredictionResponse)
def predict_price_route(payload: PricePredictionRequest) -> PricePredictionResponse:
    """Predict a median house value from request features."""
    prediction = predict_price(payload.model_dump())
    return PricePredictionResponse(
        predicted_price=prediction,
        model_name="linear_regression",
    )
